#!/usr/bin/env python3
"""
Qwen2.5-VL fine-tuning script for Parquet OCR datasets.
Optimized for training convergence and stability.
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from io import BytesIO
from functools import partial
from tqdm import tqdm
import logging
import json
from pathlib import Path

# Transformers imports
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParquetOCRDataset(Dataset):
    """Dataset class for Parquet OCR files."""
    
    def __init__(self, parquet_path, image_column="Image", text_column="Text", 
                 user_text="Transcribe the Jawi script in this image into Jawi text"):
        self.user_text = user_text
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        required = ["Identifier", image_column, text_column]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column in parquet: {col}")
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Handle parquet image data
        image_data = row[self.image_column]
        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            image = Image.open(image_data).convert("RGB")
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
    """Improved collate function with better error handling and loss masking."""
    try:
        messages_list = [item["messages"] for item in batch]
        
        # Process each item individually to handle variable image sizes better
        processed_inputs = []
        for messages in messages_list:
            # Apply chat template the same way as evaluate_model.py
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,  # False for training, True for inference
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            processed_inputs.append(inputs)
        
        # Pad sequences to same length
        max_length = max(inp.input_ids.shape[1] for inp in processed_inputs)
        
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        
        for inp in processed_inputs:
            # Pad input_ids and attention_mask
            seq_len = inp.input_ids.shape[1]
            pad_length = max_length - seq_len
            
            if pad_length > 0:
                # Pad on the right
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
        
        # Stack tensors
        batched_inputs = {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
        }
        
        if pixel_values_list:
            batched_inputs["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            batched_inputs["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
        
        # Move to device (same as evaluate_model.py does with .to(self.model.device))
        batched_inputs = {k: v.to(device) for k, v in batched_inputs.items()}
        
        # Create labels for loss computation
        labels = batched_inputs["input_ids"].clone()
        
        # Find assistant tokens and mask everything else
        for i, messages in enumerate(messages_list):
            # Simple approach: find where assistant content starts
            assistant_content = messages[1]["content"][0]["text"]
            assistant_tokens = processor.tokenizer.encode(assistant_content, add_special_tokens=False)
            
            # Mask all tokens except assistant content
            input_ids = batched_inputs["input_ids"][i].tolist()
            label_ids = [-100] * len(input_ids)
            
            # Find assistant content in the tokenized sequence
            for j in range(len(input_ids) - len(assistant_tokens) + 1):
                if input_ids[j:j+len(assistant_tokens)] == assistant_tokens:
                    # Found assistant content, set labels
                    label_ids[j:j+len(assistant_tokens)] = assistant_tokens
                    break
            
            labels[i] = torch.tensor(label_ids, dtype=torch.long)
        
        labels = labels.to(device)
        
        return batched_inputs, labels
        
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        # Return a simple fallback
        return None, None


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
    train_batch_size=1,  # Reduced for memory stability
    val_batch_size=1,
    learning_rate=5e-6,  # Lower learning rate
    warmup_steps=100,
    max_grad_norm=1.0,
    accumulation_steps=8,  # Increased to compensate for smaller batch
    eval_steps=500,
    save_steps=1000,
    
    # Generation arguments
    user_text="Transcribe the Jawi script in this image into Jawi text",
    
    # System arguments
    device=None,
):
    """Improved training function with better convergence."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training config
    config_dict = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "user_text": user_text,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Load model and processor
    logger.info("Loading model and processor...")
    
    # Clear GPU cache first
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared before loading model")
    
    # Load config and fix missing rope_scaling (same as evaluate_model.py)
    config = AutoConfig.from_pretrained("culturalheritagenus/Jawi-OCR-Qwen-v2")
    config.rope_scaling = {'type': 'default', 'mrope_section': [16, 24, 24], 'rope_type': 'default'}
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    logger.info("Model and processor loaded successfully!")
    
    # Prepare datasets
    train_dataset = ParquetOCRDataset(
        train_parquet_path, image_column, text_column, user_text
    )
    val_dataset = None
    if val_parquet_path:
        val_dataset = ParquetOCRDataset(
            val_parquet_path, image_column, text_column, user_text
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=partial(improved_collate_fn, processor=processor, device=device),
        num_workers=0,  # Avoid multiprocessing issues
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
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01,
        eps=1e-8
    )
    
    total_steps = epochs * len(train_loader) // accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
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
                # Forward pass
                outputs = model(**batch_inputs, labels=labels)
                loss = outputs.loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                accumulated_loss += loss.item()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Log progress
                    current_loss = accumulated_loss * accumulation_steps
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
                    
                    epoch_loss += accumulated_loss
                    accumulated_loss = 0.0
                    
                    # Clear cache periodically to prevent fragmentation
                    if device == "cuda" and global_step % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    # Evaluation
                    if val_loader and global_step % eval_steps == 0:
                        logger.info(f"\nEvaluating at step {global_step}...")
                        val_loss = validate(model, val_loader)
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_dir = os.path.join(output_dir, "best_model")
                            os.makedirs(best_model_dir, exist_ok=True)
                            model.save_pretrained(best_model_dir)
                            processor.save_pretrained(best_model_dir)
                            logger.info(f"New best model saved to {best_model_dir}")
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        model.save_pretrained(checkpoint_dir)
                        processor.save_pretrained(checkpoint_dir)
                        logger.info(f"Checkpoint saved to {checkpoint_dir}")
                        
            except Exception as e:
                logger.error(f"Training batch failed: {e}")
                continue
        
        logger.info(f"Epoch {epoch+1} completed. Average loss: {epoch_loss / max(1, len(train_loader) // accumulation_steps):.4f}")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL fine-tuning on Parquet OCR data")
    
    # Data arguments
    parser.add_argument("--model_name", type=str, required=True, 
                       help="Model name to fine-tune (e.g., Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--train_data", type=str, required=True, 
                       help="Path to training .parquet file")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Path to validation .parquet file (optional)")
    parser.add_argument("--output_dir", type=str, default="./models/qwen25_finetuned",
                       help="Directory to save the fine-tuned model")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=1,
                       help="Validation batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Evaluation/checkpoint frequency (in optimizer steps)")
    parser.add_argument("--user_text", type=str, 
                       default="Convert this image to text",
                       help="Instruction used in the chat template")
    parser.add_argument("--device", type=str, default=None,
                       help="Device map (cuda|cpu); defaults to cuda if available")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.train_data):
        logger.error(f"Training data not found: {args.train_data}")
        sys.exit(1)
    
    if args.val_data and not os.path.exists(args.val_data):
        logger.warning(f"Validation data not found: {args.val_data}. Continuing without validation.")
        args.val_data = None
    
    # Run training
    train_model(
        train_parquet_path=args.train_data,
        val_parquet_path=args.val_data,
        image_column="Image",  # Hardcoded since we only handle parquet files
        text_column="Text",    # Hardcoded since we only handle parquet files
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.lr,  # Changed from learning_rate to lr
        accumulation_steps=args.accumulation_steps,
        eval_steps=args.eval_steps,
        user_text=args.user_text,
        device=args.device,
    )


if __name__ == "__main__":
    main()