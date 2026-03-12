#!/usr/bin/env python3
"""
Simple script to run fine-tuning with predefined configurations.
Usage examples:
    python run_finetune.py --config basic
    python run_finetune.py --config advanced
    python run_finetune.py --model "culturalheritagenus/qwen-for-jawi-v1" --data "train-00000-of-00001-82ad548e2f991d3f.parquet"
"""

import subprocess
import sys
import os
import argparse

# Predefined configurations
CONFIGS = {
    "basic": {
        "model_name": "culturalheritagenus/qwen-for-jawi-v1",
        "train_data": "/Users/jon/Documents/FYP/OCR Data/data/train-00000-of-00001-82ad548e2f991d3f.parquet",
        "val_data": "/Users/jon/Documents/FYP/OCR Data/data/validation-00000-of-00001-d4437dd0d8db37e8.parquet",
        "num_epochs": 2,
        "batch_size": 1,
        "learning_rate": 3e-5,
        "output_dir": "./models/qwen_ocr_basic",
        "gradient_accumulation_steps": 8
    },
    "advanced": {
        "model_name": "culturalheritagenus/qwen-for-jawi-v1",
        "train_data": "/Users/jon/Documents/FYP/OCR Data/data/train-00000-of-00001-82ad548e2f991d3f.parquet",
        "val_data": "/Users/jon/Documents/FYP/OCR Data/data/validation-00000-of-00001-d4437dd0d8db37e8.parquet",
        "num_epochs": 5,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "output_dir": "./models/qwen_ocr_advanced",
        "gradient_accumulation_steps": 4,
        "max_length": 768
    }
}

def run_finetune(config):
    """Run the fine-tuning script with the given configuration."""
    
    # Build command
    cmd = [
        sys.executable, "finetune_model.py",
        "--model_name", config["model_name"],
        "--train_data", config["train_data"],
        "--output_dir", config["output_dir"],
        "--num_epochs", str(config["num_epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--gradient_accumulation_steps", str(config["gradient_accumulation_steps"])
    ]
    
    # Add LoRA flag if not disabled
    if not config.get("no_lora", False):
        cmd.append("--use_lora")
    
    # Add optional parameters
    if "val_data" in config and config["val_data"]:
        cmd.extend(["--val_data", config["val_data"]])
    
    if "max_length" in config:
        cmd.extend(["--max_length", str(config["max_length"])])
    
    if "instruction" in config:
        cmd.extend(["--instruction", config["instruction"]])
    
    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("Fine-tuning completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Fine-tuning failed with error code: {e.returncode}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="Run Qwen OCR fine-tuning with predefined configurations")
    
    # Configuration options
    parser.add_argument("--config", type=str, choices=list(CONFIGS.keys()),
                       help="Use predefined configuration")
    
    # Direct parameter options
    parser.add_argument("--model", type=str,
                       help="Model name (overrides config)")
    parser.add_argument("--data", type=str,
                       help="Training data path (overrides config)")
    parser.add_argument("--output", type=str,
                       help="Output directory (overrides config)")
    parser.add_argument("--epochs", type=int,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float,
                       help="Learning rate (overrides config)")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA (use full fine-tuning instead)")
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.config:
        config = CONFIGS[args.config].copy()
        print(f"Using predefined configuration: {args.config}")
    else:
        config = CONFIGS["basic"].copy()  # Default to basic config
        print("Using default basic configuration")
    
    # Override with command line arguments
    if args.model:
        config["model_name"] = args.model
    if args.data:
        config["train_data"] = args.data
    if args.output:
        config["output_dir"] = args.output
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.no_lora:
        config["no_lora"] = True
    
    # Display configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if training data exists
    if not os.path.exists(config["train_data"]):
        print(f"Error: Training data file not found: {config['train_data']}")
        sys.exit(1)
    
    # Check if validation data exists (if specified)
    if "val_data" in config and config["val_data"] and not os.path.exists(config["val_data"]):
        print(f"Warning: Validation data file not found: {config['val_data']}")
        config["val_data"] = None
    
    # Run fine-tuning
    return run_finetune(config)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)