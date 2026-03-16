import torch
import numpy as np
import random
import os
import sys
import pandas as pd
import logging
import subprocess
import tempfile
import gc
from pathlib import Path
from datasets import concatenate_datasets
import io
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from OCR_model_functions import OCRModelFunctions
from helper_functions import calculate_confidence_from_scores, build_special_token_ids
from data_loading_functions import load_parquet_data_with_fallback, decode_parquet_image_example

# ====================================================
# CONFIG
# ====================================================

MODEL_NAME = "culturalheritagenus/Jawi-OCR-Qwen-v2"  # change if needed
TRAIN_PATH = "../data_v4/train"
TEST_PATH = "../data_v4/test"
VALIDATION_PATH = "../data_v4/validation"

INITIAL_LABEL_FRACTION = 0.1
ACQUISITION_FRACTION = 0.1
NUM_ITERATIONS = 1
UNCERTAINTY_BATCH_SIZE = 2  # Batch size for uncertainty computation
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ocr_model(model_name):
    """Load OCR model through shared abstraction."""
    ocr = OCRModelFunctions(model_name=model_name, max_new_tokens=64)
    ocr.load()
    return ocr


# ====================================================
# LOAD MODEL
# ====================================================

ocr = load_ocr_model(MODEL_NAME)

# ====================================================
# LOAD PARQUET DATA
# ====================================================

train_data = load_parquet_data_with_fallback(TRAIN_PATH)

# Apply image decoding to the loaded data
print("Decoding images...")
dataset = train_data.map(decode_parquet_image_example, desc="Decoding images")

# Keep only required fields
print("Cleaning dataset columns...")
dataset = dataset.remove_columns(
    [col for col in dataset.column_names if col not in ["image", "text", "id"]]
)

# Debug: Print first 5 rows after decoding
print("\nDEBUG: First 5 rows after image decoding:")
for i in range(min(5, len(dataset))):
    sample = dataset[i]
    print(f"Row {i}:")
    print(f"  ID: {sample['id']}")
    print(f"  Text: {sample['text'][:50]}..." if len(sample['text']) > 50 else f"  Text: {sample['text']}")
    print(f"  Image type: {type(sample['image'])}")
    print(f"  Image size: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")
    print(f"  Image mode: {sample['image'].mode if hasattr(sample['image'], 'mode') else 'N/A'}")
    print()

# ====================================================
# ACTIVE LEARNING SETUP
# ====================================================

# Use the loaded training data as the pool for active learning
train_pool = dataset

n_initial = int(len(train_pool) * INITIAL_LABEL_FRACTION)

shuffled = train_pool.shuffle(seed=SEED)
labeled_set = shuffled.select(range(n_initial))
unlabeled_pool = shuffled.select(range(n_initial, len(train_pool)))

# ====================================================
# DATA SAVING FOR EXTERNAL TRAINING
# ====================================================

def save_dataset_to_parquet(dataset, filepath):
    """Save HuggingFace dataset to parquet format for external training script."""
    # Convert to pandas DataFrame
    data_list = []
    for item in dataset:
        # Convert PIL image to bytes for parquet storage
        img_buffer = io.BytesIO()
        item["image"].save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        data_list.append({
            "Identifier": item["id"],
            "Image": {"bytes": img_bytes},
            "Text": item["text"]
        })
    
    df = pd.DataFrame(data_list)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {len(dataset)} samples to {filepath}")

def call_training_script(train_parquet_path, iteration, model_name):
    """Call the unified_qwen25_finetune.py script for training."""
    output_dir = f"./active_learning_models/iteration_{iteration}"
    
    # Prepare arguments for the training script
    cmd = [
        "python", "../finetuning/unified_qwen25_finetune.py",
        "--model_name", model_name,
        "--train_data", train_parquet_path,
        "--output_dir", output_dir,
        "--epochs", "1",
        "--train_batch_size", "1",
        "--lr", "5e-6",
        "--accumulation_steps", "8",  # Increased to compensate for smaller batch
        "--user_text", "Recognize the text in the image."
    ]
    
    logger.info(f"Running training script: {' '.join(cmd)}")
    
    # Set memory management environment variables
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error reporting
    
    try:
        # Run the training script - inherit stdout/stderr for slurm logging
        result = subprocess.run(cmd, check=True, env=env)
        logger.info("Training completed successfully")
        return os.path.join(output_dir, "final_model")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training script failed with exit code: {e.returncode}")
        raise

# ====================================================
# UNCERTAINTY (CONFIDENCE-BASED)
# ====================================================

def compute_uncertainty(ocr, dataset, batch_size=8):
    """Compute uncertainty scores based on model confidence with TRUE batch processing."""
    model = ocr.model
    processor = ocr.processor
    model.eval()
    
    scores = []
    confidences = []
    error_count = 0
    success_count = 0
    
    print(f"Computing uncertainty for {len(dataset)} samples using TRUE batch_size={batch_size}...")
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Computing uncertainty"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_samples = [dataset[i] for i in range(batch_start, batch_end)]
        
        try:
            # Prepare all inputs for the batch
            batch_inputs = []
            
            for sample in batch_samples:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": sample["image"]},
                            {"type": "text", "text": "Transcribe the Jawi script in this image into Jawi text"}
                        ]
                    },
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                
                batch_inputs.append(inputs)
            
            # Pad all inputs to same length for TRUE batch processing
            max_length = max(inp.input_ids.shape[1] for inp in batch_inputs)
            
            input_ids_list = []
            attention_mask_list = []
            pixel_values_list = []
            image_grid_thw_list = []
            
            for inp in batch_inputs:
                seq_len = inp.input_ids.shape[1]
                pad_length = max_length - seq_len
                
                if pad_length > 0:
                    input_ids = torch.nn.functional.pad(inp.input_ids, (pad_length, 0), value=processor.tokenizer.pad_token_id)
                    attention_mask = torch.nn.functional.pad(inp.attention_mask, (pad_length, 0), value=0)
                else:
                    input_ids = inp.input_ids
                    attention_mask = inp.attention_mask
                
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                
                if hasattr(inp, 'pixel_values') and inp.pixel_values is not None:
                    pixel_values_list.append(inp.pixel_values)
                if hasattr(inp, 'image_grid_thw') and inp.image_grid_thw is not None:
                    image_grid_thw_list.append(inp.image_grid_thw)
            
            # Create TRUE batched input
            batched_inputs = {
                "input_ids": torch.cat(input_ids_list, dim=0).to(device),
                "attention_mask": torch.cat(attention_mask_list, dim=0).to(device),
            }
            
            if pixel_values_list:
                batched_inputs["pixel_values"] = torch.cat(pixel_values_list, dim=0).to(device)
            if image_grid_thw_list:
                batched_inputs["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0).to(device)
            
            # SINGLE model call for entire batch
            with torch.no_grad():
                gen_out = model.generate(
                    **batched_inputs,
                    max_new_tokens=64,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            # Extract individual results from batched output
            input_lens = [inp.input_ids.shape[1] for inp in batch_inputs]
            batch_confidences = []
            
            # Full special token filtering
            special_ids = build_special_token_ids(processor.tokenizer)
            
            # Calculate individual confidence from batch results
            for i, input_len in enumerate(input_lens):
                # Extract generated tokens for this sample
                gen_token_ids = gen_out.sequences[i][input_len:]
                generated_tokens_list = gen_token_ids.tolist() if hasattr(gen_token_ids, 'tolist') else list(gen_token_ids)
                
                # Extract scores for this sample from each generation step
                sample_scores = []
                for step_idx in range(len(gen_out.scores)):
                    if step_idx < len(generated_tokens_list):
                        # Get logits for sample i at step step_idx
                        step_logits = gen_out.scores[step_idx][i:i+1]  # [1, vocab_size]
                        sample_scores.append(step_logits)
                
                # Calculate confidence for this individual sample
                confidence = calculate_confidence_from_scores(
                    sample_scores,
                    generated_tokens_list,
                    ignore_token_ids=special_ids
                )
                
                batch_confidences.append(confidence)
            
            # Add results for this batch
            for conf in batch_confidences:
                confidences.append(conf)
                uncertainty = 1.0 - conf
                scores.append(uncertainty)
            
            success_count += len(batch_samples)
            
            # Memory cleanup
            del batched_inputs
            del gen_out
            clear_gpu()
                
        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")
            error_count += len(batch_samples)
            # Add default values for failed batch
            for _ in range(len(batch_samples)):
                scores.append(1.0)  # High uncertainty for failed samples
                confidences.append(0.0)
            
            # Clean up on error
            if device == "cuda":
                torch.cuda.empty_cache()
    
    print(f"DEBUG: Processed {success_count} samples successfully, {error_count} samples failed")
    print(f"DEBUG: Final uncertainty range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"DEBUG: Final confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
    
    model.train()
    return np.array(scores), np.array(confidences)

def clear_gpu():
    """Aggressively clear GPU memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

# ====================================================
# ACTIVE LEARNING LOOP
# ====================================================

# Create output directory for active learning models
os.makedirs("./active_learning_models", exist_ok=True)

# Keep track of current model path
current_model_path = MODEL_NAME

for iteration in range(NUM_ITERATIONS):
    print(f"\n===== ITERATION {iteration} =====")

    # Save current labeled dataset to parquet for training
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        train_parquet_path = tmp_file.name
    
    save_dataset_to_parquet(labeled_set, train_parquet_path)
    
    try:
        # Free up GPU memory before training
        logger.info("Clearing GPU memory before training...")

        try:
            del ocr.model
        except Exception:
            pass

        clear_gpu()
        
        # Train model using external script
        trained_model_path = call_training_script(train_parquet_path, iteration, current_model_path)
        
        # Load the newly trained model
        ocr = load_ocr_model(trained_model_path)
        clear_gpu()
        # Update current model path for next iteration
        current_model_path = trained_model_path
        
    finally:
        # Clean up temporary file
        if os.path.exists(train_parquet_path):
            os.unlink(train_parquet_path)

    if len(unlabeled_pool) == 0:
        break

    # Compute uncertainty and confidence scores with batch processing
    uncertainties, confidences = compute_uncertainty(
        ocr,
        unlabeled_pool,
        batch_size=UNCERTAINTY_BATCH_SIZE
    )  # Increased batch size

    # DEBUG: Print uncertainty and confidence statistics
    print(f"DEBUG: Uncertainty stats - min: {uncertainties.min():.4f}, max: {uncertainties.max():.4f}, mean: {uncertainties.mean():.4f}")
    print(f"DEBUG: Confidence stats - min: {confidences.min():.4f}, max: {confidences.max():.4f}, mean: {confidences.mean():.4f}")
    print(f"DEBUG: Number of samples with uncertainty = 1.0: {(uncertainties == 1.0).sum()}/{len(uncertainties)}")

    n_acquire = int(len(unlabeled_pool) * ACQUISITION_FRACTION)
    acquire_indices = np.argsort(-uncertainties)[:n_acquire]

    new_samples = unlabeled_pool.select(acquire_indices.tolist())

    labeled_set = concatenate_datasets([labeled_set, new_samples])

    remaining_indices = list(set(range(len(unlabeled_pool))) - set(acquire_indices))
    unlabeled_pool = unlabeled_pool.select(remaining_indices)

    print(f"Added {len(new_samples)} samples.")
    print(f"Mean uncertainty selected: {uncertainties[acquire_indices].mean():.4f}")
    print(f"Total labeled samples: {len(labeled_set)}")
    print(f"Remaining unlabeled samples: {len(unlabeled_pool)}")

print("Active learning complete.")