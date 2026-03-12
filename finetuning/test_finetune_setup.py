#!/usr/bin/env python3
"""
Test script to verify fine-tuning setup and data loading.
Run this before starting actual fine-tuning to catch any issues early.
"""

import os
import sys
import torch
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("❌ CUDA is not available")
        return False

def test_model_loading(model_name: str):
    """Test model and processor loading."""
    print(f"\nTesting model loading: {model_name}")
    
    try:
        # Load processor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        print("✓ Processor loaded")
        
        # Load model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("✓ Model loaded")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def test_data_loading(data_path: str, processor, max_samples: int = 3):
    """Test data loading and processing."""
    print(f"\nTesting data loading: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        # Load data
        df = pd.read_parquet(data_path)
        print(f"✓ Data loaded: {len(df)} samples")
        
        # Check columns
        required_columns = ['Identifier', 'Image', 'Text']
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Missing column: {col}")
                return False
        print("✓ All required columns present")
        
        # Test processing a few samples
        print(f"\nTesting sample processing (first {max_samples} samples):")
        
        for i in range(min(max_samples, len(df))):
            row = df.iloc[i]
            
            try:
                # Load image
                image_data = row['Image']
                image_bytes = image_data['bytes']
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
                
                # Get text
                text = str(row['Text']).strip()
                
                print(f"  Sample {i+1}:")
                print(f"    Identifier: {row['Identifier']}")
                print(f"    Image size: {image.size}")
                print(f"    Text length: {len(text)} chars")
                print(f"    Text preview: {repr(text[:50])}...")
                
                # Test processor
                if processor:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": "Convert this image to text"},
                            ],
                        }
                    ]
                    
                    processed_text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = processor(
                        text=[processed_text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=False,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    
                    print(f"    Processed input shape: {inputs['input_ids'].shape}")
                    
            except Exception as e:
                print(f"    ❌ Error processing sample {i+1}: {e}")
                return False
        
        print("✓ Sample processing successful")
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage."""
    print(f"\nTesting memory usage...")
    
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get memory info
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved: {memory_reserved:.2f} GB")
        print(f"  Total: {memory_total:.2f} GB")
        print(f"  Available: {memory_total - memory_reserved:.2f} GB")
        
        if memory_total - memory_reserved < 4.0:
            print("⚠️  Warning: Low GPU memory. Consider reducing batch size.")
        else:
            print("✓ Sufficient GPU memory available")
    else:
        print("  CPU mode - memory usage will be system RAM")

def main():
    """Main test function."""
    print("="*60)
    print("QWEN OCR FINE-TUNING SETUP TEST")
    print("="*60)
    
    # Default paths
    model_name = "culturalheritagenus/qwen-for-jawi-v1"
    train_data_path = "/Users/jon/Documents/FYP/OCR Data/data/train-00000-of-00001-82ad548e2f991d3f.parquet"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    if len(sys.argv) > 2:
        train_data_path = sys.argv[2]
    
    print(f"Model: {model_name}")
    print(f"Data: {train_data_path}")
    print()
    
    # Run tests
    cuda_ok = test_cuda()
    model, processor = test_model_loading(model_name)
    data_ok = test_data_loading(train_data_path, processor)
    
    if cuda_ok:
        test_memory_usage()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"CUDA: {'✓' if cuda_ok else '❌'}")
    print(f"Model Loading: {'✓' if model is not None else '❌'}")
    print(f"Data Loading: {'✓' if data_ok else '❌'}")
    
    if cuda_ok and model is not None and data_ok:
        print("\n🎉 All tests passed! Ready for fine-tuning.")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues before fine-tuning.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)