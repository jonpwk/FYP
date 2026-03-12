#!/usr/bin/env python3
"""
Run Qwen2.5-VL fine-tuning on a local Parquet OCR dataset.

- Input:  model name to fine-tune, training (and optional validation) .parquet file(s)
- Output: saved fine-tuned model (and processor) to the output directory

This script reuses the Qwen2.5 backbone utilities (collate_fn, validate) and adapts
them to your Parquet data format used in finetune_model.py (Image bytes + Text).
"""

import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
from io import BytesIO
from functools import partial
from tqdm import tqdm

# Qwen2.5-VL imports
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

# Reuse the backbone helpers from qwen25_finetune.py
from qwen25_finetune import collate_fn, validate


class ParquetOCRDataset(Dataset):
    """Parquet OCR dataset compatible with Qwen2.5-VL collate (messages-based)."""
    def __init__(self, parquet_path: str, user_text: str = "Transcribe the Jawi script in this image into Jawi text"):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        self.user_text = user_text

        # Validate required columns
        required = ["Identifier", "Image", "Text"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column in parquet: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image from bytes dict: {'bytes': ...}
        image_data = row["Image"]
        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            # Fallback: try file path
            image = Image.open(image_data).convert("RGB")

        assistant_text = str(row["Text"]) if pd.notna(row["Text"]) else ""

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


def run_training(
    model_name: str,
    train_parquet: str,
    output_dir: str,
    val_parquet: str | None = None,
    device: str | None = None,
    user_text: str = "Convert this image to text",
    lr: float = 1e-5,
    train_batch_size: int = 2,
    val_batch_size: int = 1,
    num_epochs: int = 1,
    accumulation_steps: int = 2,
    eval_steps: int = 1000,
):
    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading config!")
    config = AutoConfig.from_pretrained(model_name)
    config.rope_scaling = {'type': 'default', 'mrope_section': [16, 24, 24], 'rope_type': 'default'}

    print(f"Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
        config=config,
    )
    print("Model loaded.")

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        padding_side="right",
        trust_remote_code=True,
    )
    print("Processor loaded.")

    # Datasets
    train_dataset = ParquetOCRDataset(train_parquet, user_text=user_text)
    val_dataset = ParquetOCRDataset(val_parquet, user_text=user_text) if val_parquet else None

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor, device=device),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, processor=processor, device=device),
        )

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    global_step = 0
    total_steps = num_epochs * (len(train_loader) // max(1, accumulation_steps))
    progress_bar = tqdm(total=total_steps if total_steps > 0 else None, desc="Training")

    print("Starting training...")
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if progress_bar.total is not None:
                    progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{(loss.item() * accumulation_steps):.4f}"})

                # Periodic evaluation and checkpointing
                if val_loader is not None and (global_step % eval_steps == 0):
                    print(f"\nEvaluating at step {global_step}...")
                    avg_val_loss = validate(model, val_loader)
                    print(f"Validation loss: {avg_val_loss:.4f}")

                    save_dir = os.path.join(output_dir, f"checkpoint-step-{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)
                    print(f"Saved checkpoint to {save_dir}")
                    model.train()

    progress_bar.close()

    # Save final artifacts
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nTraining complete. Model saved to: {output_dir}")



def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL fine-tuning on Parquet OCR data")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to fine-tune (e.g., Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training .parquet file")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation .parquet file (optional)")
    parser.add_argument("--output_dir", type=str, default="./models/qwen25_finetuned", help="Directory to save the fine-tuned model")

    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluation/checkpoint frequency (in optimizer steps)")
    parser.add_argument("--user_text", type=str, default="Convert this image to text", help="Instruction used in the chat template")
    parser.add_argument("--device", type=str, default=None, help="Device map (cuda|cpu); defaults to cuda if available")

    args = parser.parse_args()

    # Basic checks
    if not os.path.exists(args.train_data):
        print(f"Error: training data not found: {args.train_data}")
        sys.exit(1)
    if args.val_data and not os.path.exists(args.val_data):
        print(f"Warning: validation data not found: {args.val_data}. Continuing without validation.")
        args.val_data = None

    run_training(
        model_name=args.model_name,
        train_parquet=args.train_data,
        output_dir=args.output_dir,
        val_parquet=args.val_data,
        device=args.device,
        user_text=args.user_text,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        eval_steps=args.eval_steps,
    )


if __name__ == "__main__":
    main()
