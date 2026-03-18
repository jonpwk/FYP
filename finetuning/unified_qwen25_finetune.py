#!/usr/bin/env python3
"""
Qwen2.5-VL fine-tuning script for Parquet OCR datasets.
Optimized for training convergence and stability.
"""

import os
import sys
import argparse
import gc
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial
from tqdm import tqdm
import logging
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from data_loading_functions import load_parquet_dataframe, validate_required_columns, load_image_as_rgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParquetOCRDataset(Dataset):
    """Dataset class for Parquet OCR files."""

    def __init__(self, parquet_path, image_column="Image", text_column="Text",
                 user_text="Transcribe the Jawi script in this image into Jawi text"):
        self.user_text = user_text

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        required = ["Identifier", image_column, text_column]
        self.df = load_parquet_dataframe(parquet_path)
        validate_required_columns(self.df, required, context="Parquet")
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_data = row[self.image_column]
        image = load_image_as_rgb(image_data)
        assistant_text = str(row[self.text_column]) if pd.notna(row[self.text_column]) else ""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ],
            },
        ]
        return {"messages": messages}


def improved_collate_fn(batch, processor, device):
    """Collate function with loss masking and memory-safe tensor construction."""
    try:
        messages_list = [item["messages"] for item in batch]

        processed_inputs = []
        for messages in messages_list:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            processed_inputs.append(inputs)

        max_length = max(inp.input_ids.shape[1] for inp in processed_inputs)

        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []

        for inp in processed_inputs:
            seq_len = inp.input_ids.shape[1]
            pad_length = max_length - seq_len

            if pad_length > 0:
                input_ids = F.pad(inp.input_ids, (0, pad_length), value=processor.tokenizer.pad_token_id)
                attention_mask = F.pad(inp.attention_mask, (0, pad_length), value=0)
            else:
                input_ids = inp.input_ids
                attention_mask = inp.attention_mask

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

            if hasattr(inp, 'pixel_values') and inp.pixel_values is not None:
                pixel_values_list.append(inp.pixel_values)
            if hasattr(inp, 'image_grid_thw') and inp.image_grid_thw is not None:
                image_grid_thw_list.append(inp.image_grid_thw)

        batched_inputs = {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
        }
        if pixel_values_list:
            batched_inputs["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            batched_inputs["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        # Free intermediate lists before moving to GPU
        del input_ids_list, attention_mask_list, pixel_values_list, image_grid_thw_list, processed_inputs

        batched_inputs = {k: v.to(device) for k, v in batched_inputs.items()}

        # Build labels: mask everything except the assistant response tokens
        labels = batched_inputs["input_ids"].clone()

        for i, messages in enumerate(messages_list):
            assistant_content = messages[1]["content"][0]["text"]
            assistant_tokens = processor.tokenizer.encode(assistant_content, add_special_tokens=False)

            input_ids = batched_inputs["input_ids"][i].tolist()
            label_ids = [-100] * len(input_ids)

            for j in range(len(input_ids) - len(assistant_tokens) + 1):
                if input_ids[j:j + len(assistant_tokens)] == assistant_tokens:
                    label_ids[j:j + len(assistant_tokens)] = assistant_tokens
                    break

            labels[i] = torch.tensor(label_ids, dtype=torch.long)

        labels = labels.to(device)
        return batched_inputs, labels

    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        return None, None


def clear_gpu():
    """Flush the GPU caching allocator.

    synchronize() must precede empty_cache() so that all in-flight CUDA
    kernels have finished releasing their allocations first.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def validate(model, val_loader):
    """Validate the model and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_inputs, labels in tqdm(val_loader, desc="Validating"):
            if batch_inputs is None:
                continue
            try:
                outputs = model(**batch_inputs, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1
            except Exception as e:
                logger.warning(f"Validation batch failed: {e}")
                continue

    avg_loss = total_loss / max(num_batches, 1)
    model.train()
    return avg_loss


def train_model(
    # Data arguments
    train_parquet_path,
    val_parquet_path=None,
    image_column="Image",
    text_column="Text",

    # Model arguments
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    output_dir="./models/qwen25_finetuned",

    # Training arguments
    epochs=3,
    train_batch_size=1,
    val_batch_size=1,
    learning_rate=5e-6,
    warmup_steps=100,
    max_grad_norm=1.0,
    accumulation_steps=8,
    eval_steps=500,
    save_steps=1000,

    # Generation arguments
    user_text="Transcribe the Jawi script in this image into Jawi text",

    # System arguments
    device=None,
):
    """Fine-tune with gradient checkpointing and aggressive memory hygiene."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    config_dict = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "user_text": user_text,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Loading model and processor...")

    # FIX: flush allocator before loading — the parent process may have left
    # fragmented reservations that prevent a clean contiguous model load.
    clear_gpu()
    if device == "cuda":
        logger.info(
            f"GPU memory before model load: "
            f"{torch.cuda.memory_allocated() / 1e9:.2f} GiB allocated, "
            f"{torch.cuda.memory_reserved() / 1e9:.2f} GiB reserved"
        )

    config = AutoConfig.from_pretrained("culturalheritagenus/Jawi-OCR-Qwen-v2")
    config.rope_scaling = {'type': 'default', 'mrope_section': [16, 24, 24], 'rope_type': 'default'}

    # FIX: use bfloat16 consistently (same precision as inference in main.py
    # on CUDA) to avoid an implicit fp32 upcasting that doubles VRAM usage.
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # FIX: enable gradient checkpointing to trade compute for memory.
    # For a 3B model this typically saves ~30-40% activation memory at the
    # cost of ~20% slower backward pass — a worthwhile trade on a 40 GiB GPU.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
        padding_side="right",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    logger.info("Model and processor loaded successfully!")

    train_dataset = ParquetOCRDataset(train_parquet_path, image_column, text_column, user_text)
    val_dataset = None
    if val_parquet_path:
        val_dataset = ParquetOCRDataset(val_parquet_path, image_column, text_column, user_text)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=partial(improved_collate_fn, processor=processor, device=device),
        num_workers=0,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=partial(improved_collate_fn, processor=processor, device=device),
            num_workers=0,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )

    total_steps = epochs * len(train_loader) // accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    model.train()
    global_step = 0
    best_val_loss = float('inf')
    accumulated_loss = 0.0

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Total training steps: {total_steps}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (batch_inputs, labels) in enumerate(progress_bar):
            if batch_inputs is None:
                continue

            try:
                outputs = model(**batch_inputs, labels=labels)
                loss = outputs.loss / accumulation_steps

                loss.backward()
                accumulated_loss += loss.item()

                # FIX: explicitly delete outputs and loss after backward so the
                # computation graph and activation tensors are freed before the
                # next forward pass, not whenever Python GC decides to run.
                del outputs, loss

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # set_to_none=True frees gradient memory immediately
                    global_step += 1

                    current_loss = accumulated_loss * accumulation_steps
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                    })

                    epoch_loss += accumulated_loss
                    accumulated_loss = 0.0

                    # FIX: flush allocator every accumulation step (i.e. after
                    # every real weight update) rather than only every 10 steps.
                    # With gradient checkpointing this is cheap and prevents
                    # fragmentation from building up across batches.
                    clear_gpu()

                    if val_loader and global_step % eval_steps == 0:
                        logger.info(f"\nEvaluating at step {global_step}...")
                        val_loss = validate(model, val_loader)
                        logger.info(f"Validation loss: {val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_dir = os.path.join(output_dir, "best_model")
                            os.makedirs(best_model_dir, exist_ok=True)
                            model.save_pretrained(best_model_dir)
                            processor.save_pretrained(best_model_dir)
                            logger.info(f"New best model saved to {best_model_dir}")

                    if global_step % save_steps == 0:
                        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        model.save_pretrained(checkpoint_dir)
                        processor.save_pretrained(checkpoint_dir)
                        logger.info(f"Checkpoint saved to {checkpoint_dir}")

            except torch.cuda.OutOfMemoryError as oom:
                # On OOM: free everything, zero grads, and skip this batch
                # rather than crashing the whole training run.
                logger.error(f"OOM on batch {batch_idx}, skipping: {oom}")
                del batch_inputs, labels
                if 'outputs' in dir():
                    del outputs
                if 'loss' in dir():
                    del loss
                optimizer.zero_grad(set_to_none=True)
                clear_gpu()
                continue

            except Exception as e:
                logger.error(f"Training batch failed: {e}")
                optimizer.zero_grad(set_to_none=True)
                clear_gpu()
                continue

        logger.info(
            f"Epoch {epoch+1} completed. "
            f"Average loss: {epoch_loss / max(1, len(train_loader) // accumulation_steps):.4f}"
        )

    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL fine-tuning on Parquet OCR data")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./models/qwen25_finetuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--user_text", type=str, default="Transcribe the Jawi script in this image into Jawi text")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.train_data):
        logger.error(f"Training data not found: {args.train_data}")
        sys.exit(1)

    if args.val_data and not os.path.exists(args.val_data):
        logger.warning(f"Validation data not found: {args.val_data}. Continuing without validation.")
        args.val_data = None

    train_model(
        train_parquet_path=args.train_data,
        val_parquet_path=args.val_data,
        image_column="Image",
        text_column="Text",
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.lr,
        accumulation_steps=args.accumulation_steps,
        eval_steps=args.eval_steps,
        user_text=args.user_text,
        device=args.device,
    )


if __name__ == "__main__":
    main()