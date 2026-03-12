#!/usr/bin/env python3
"""
Fine-tuning script for Qwen Vision-Language model on OCR data.
Supports fine-tuning culturalheritagenus/qwen-for-jawi-v1 and other Qwen2-VL models.

This script fine-tunes the model using LoRA (Low-Rank Adaptation) for efficient training
on CUDA-enabled GPUs with proper handling of Jawi/Arabic OCR data.
"""

import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments,
    DataCollator
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO
import time
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import argparse
from performance_metrics import normalize_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRDataCollator:
    """Custom data collator for OCR vision-language tasks."""
    processor: Any
    
    def __call__(self, batch):
        """Process a batch of samples."""
        input_ids = []
        attention_masks = []
        labels = []
        images = []
        
        for sample in batch:
            input_ids.append(sample['input_ids'])
            attention_masks.append(sample['attention_mask'])
            labels.append(sample['labels'])
            if 'images' in sample:
                images.append(sample['images'])
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i in range(len(input_ids)):
            # Pad input_ids
            pad_length = max_len - len(input_ids[i])
            padded_input_ids.append(input_ids[i].tolist() + [self.processor.tokenizer.pad_token_id] * pad_length)
            
            # Pad attention_masks
            padded_attention_masks.append(attention_masks[i].tolist() + [0] * pad_length)
            
            # Pad labels (use -100 for padding tokens to ignore in loss)
            padded_labels.append(labels[i].tolist() + [-100] * pad_length)
        
        result = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long)
        }
        
        if images:
            result['images'] = images
            
        return result

class OCRDataset(Dataset):
    """Dataset class for OCR fine-tuning."""
    
    def __init__(self, data_path: str, processor: Any, max_length: int = 512, 
                 instruction: str = "Convert this image to text"):
        """
        Initialize the OCR dataset.
        
        Args:
            data_path: Path to the parquet file
            processor: Qwen2VL processor
            max_length: Maximum sequence length
            instruction: Instruction text for the model
        """
        self.processor = processor
        self.max_length = max_length
        self.instruction = instruction
        
        # Load data
        logger.info(f"Loading data from: {data_path}")
        self.df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(self.df)} samples")
        
        # Validate data format
        required_columns = ['Identifier', 'Image', 'Text']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        row = self.df.iloc[idx]
        
        try:
            # Load image from bytes
            image_data = row['Image']
            image_bytes = image_data['bytes']
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Get ground truth text
            ground_truth = str(row['Text']).strip()
            
            # Normalize text for consistency
            ground_truth = normalize_text(ground_truth)
            
            # Prepare conversation format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": self.instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ground_truth}
                    ]
                }
            ]
            
            # Process with the processor
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Process vision information
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Tokenize
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            # Prepare labels (same as input_ids for causal LM)
            input_ids = inputs['input_ids'].squeeze(0)
            labels = input_ids.clone()
            
            # Find the assistant response start to only compute loss on the answer
            # This is a simplified approach - in practice you might want more sophisticated masking
            assistant_start = text.find("assistant\n")
            if assistant_start != -1:
                # Tokenize just the part before assistant response
                user_text = text[:assistant_start + len("assistant\n")]
                user_inputs = self.processor.tokenizer(
                    user_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                user_length = len(user_inputs['input_ids'].squeeze(0))
                
                # Mask the user part in labels (set to -100 to ignore in loss)
                labels[:user_length] = -100
            
            result = {
                'input_ids': input_ids,
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': labels,
            }
            
            # Add image if present
            if image_inputs:
                result['images'] = image_inputs
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return a dummy sample in case of error
            dummy_text = f"{self.instruction}\n\nDummy text"
            dummy_inputs = self.processor.tokenizer(
                dummy_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                'input_ids': dummy_inputs['input_ids'].squeeze(0),
                'attention_mask': dummy_inputs['attention_mask'].squeeze(0),
                'labels': dummy_inputs['input_ids'].squeeze(0),
            }

class ModelFineTuner:
    """Fine-tuning class for Qwen2-VL models."""
    
    def __init__(self, model_name: str, use_lora: bool = True):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name: Name of the model to fine-tune
            use_lora: Whether to use LoRA for efficient fine-tuning
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            logger.warning("CUDA not available. Fine-tuning on CPU will be very slow.")
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, torch_dtype: torch.dtype = torch.bfloat16):
        """Load the model and processor."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        logger.info("✓ Processor loaded")
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Prepare model for training
        if self.use_lora:
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,  # Rank
                lora_alpha=32,  # Alpha parameter
                lora_dropout=0.1,  # Dropout probability
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"
                ],  # Target modules for LoRA
                bias="none",
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logger.info("✓ LoRA applied to model")
        
        logger.info("✓ Model loaded and prepared for training")
    
    def create_datasets(self, train_data_path: str, val_data_path: Optional[str] = None,
                       max_length: int = 512, instruction: str = "Convert this image to text"):
        """Create training and validation datasets."""
        
        # Create training dataset
        train_dataset = OCRDataset(
            train_data_path, 
            self.processor, 
            max_length, 
            instruction
        )
        
        # Create validation dataset if provided
        val_dataset = None
        if val_data_path:
            val_dataset = OCRDataset(
                val_data_path, 
                self.processor, 
                max_length, 
                instruction
            )
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None,
              output_dir: str = "./fine_tuned_model",
              num_epochs: int = 3,
              batch_size: int = 2,
              learning_rate: float = 5e-5,
              warmup_steps: int = 100,
              logging_steps: int = 10,
              eval_steps: int = 100,
              save_steps: int = 500,
              gradient_accumulation_steps: int = 4):
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            logging_steps: Steps between logging
            eval_steps: Steps between evaluation
            save_steps: Steps between saving checkpoints
            gradient_accumulation_steps: Gradient accumulation steps
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps if val_dataset else None,
            eval_strategy="steps" if val_dataset else "no",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard for now
            dataloader_num_workers=2,
            bf16=True if self.device == "cuda" else False,
            remove_unused_columns=False,  # Important for vision models
        )
        
        # Data collator
        data_collator = OCRDataCollator(processor=self.processor)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting training...")
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/3600:.2f} hours")
        
        # Save the final model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "model_name": self.model_name,
            "use_lora": self.use_lora,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_time_hours": training_time / 3600,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
        }
        
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Model saved to: {output_dir}")
        return trainer

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL model for OCR")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, 
                       default="culturalheritagenus/qwen-for-jawi-v1",
                       help="Model name to fine-tune")
    parser.add_argument("--train_data", type=str, 
                       default="/Users/jon/Documents/FYP/OCR Data/data/train-00000-of-00001-82ad548e2f991d3f.parquet",
                       help="Path to training data parquet file")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Path to validation data parquet file")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_qwen_ocr",
                       help="Output directory for the fine-tuned model")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--instruction", type=str, default="Convert this image to text",
                       help="Instruction text for the model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a CUDA-enabled GPU.")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("QWEN2-VL OCR FINE-TUNING")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Validation data: {args.val_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Use LoRA: {args.use_lora}")
    logger.info("="*60)
    
    # Initialize fine-tuner
    fine_tuner = ModelFineTuner(args.model_name, args.use_lora)
    
    # Load model
    fine_tuner.load_model()
    
    # Create datasets
    train_dataset, val_dataset = fine_tuner.create_datasets(
        args.train_data, 
        args.val_data, 
        args.max_length, 
        args.instruction
    )
    
    # Start training
    trainer = fine_tuner.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
