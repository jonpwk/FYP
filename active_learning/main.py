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
import argparse
from pathlib import Path
from datasets import concatenate_datasets
import io
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helper_functions import calculate_confidence_from_scores, build_special_token_ids
from data_loading_functions import load_parquet_data_with_fallback, decode_parquet_image_example
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

# ====================================================
# CLI
# ====================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["active", "random"],
        default="active",
        help="Acquisition strategy to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=0,
        help="Run identifier for repeated runs with the same seed. Mainly useful for random strategy.",
    )
    return parser.parse_args()


args = parse_args()

# ====================================================
# CONFIG
# ====================================================

MODEL_NAME = "culturalheritagenus/Jawi-OCR-Qwen-v2"
TRAIN_PATH = "./data_v4/train"
TEST_PATH = "./data_v4/test"
VALIDATION_PATH = "./data_v4/validation"

INITIAL_LABEL_FRACTION = 0.1
ACQUISITION_FRACTION = 0.1
NUM_ITERATIONS = 5
UNCERTAINTY_BATCH_SIZE = 2
MAX_IMAGE_SIDE = 1024

ACQUISITION_STRATEGY = args.strategy
SEED = args.seed
RUN_ID = args.run_id

if ACQUISITION_STRATEGY == "random":
    EXPERIMENT_NAME = f"{ACQUISITION_STRATEGY}_seed_{SEED}_run_{RUN_ID}"
else:
    EXPERIMENT_NAME = f"{ACQUISITION_STRATEGY}_seed_{SEED}"

OUTPUT_DIR = f"./active_learning_models/{EXPERIMENT_NAME}"
SELECTION_TRACKING_CSV = os.path.join(OUTPUT_DIR, "selected_samples_tracking.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================================
# MODEL LOAD / UNLOAD HELPERS
# ====================================================

PROCESSOR_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_DTYPE = torch.float16 if device.type == "cuda" else torch.float32


def log_cuda_memory(prefix: str):
    if torch.cuda.is_available():
        logger.info(
            f"{prefix} | "
            f"allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB, "
            f"max_allocated={torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )


def load_model_and_processor(model_name):
    """Load model and processor as plain variables with a consistent dtype/device path."""
    config = AutoConfig.from_pretrained(model_name)

    config.rope_scaling = {
        "type": "default",
        "mrope_section": [16, 24, 24],
        "rope_type": "default",
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log_cuda_memory(f"Before loading model from {model_name}")

    loaded_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        config=config,
        torch_dtype=MODEL_DTYPE,
        trust_remote_code=True,
    )

    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    loaded_processor = AutoProcessor.from_pretrained(
        PROCESSOR_MODEL,
        trust_remote_code=True,
        padding_side="left",
    )

    if loaded_processor.tokenizer.pad_token is None:
        loaded_processor.tokenizer.pad_token = loaded_processor.tokenizer.eos_token
    loaded_processor.tokenizer.padding_side = "left"

    logger.info(f"Model loaded from {model_name} on {device} with dtype={MODEL_DTYPE}")

    if torch.cuda.is_available():
        log_cuda_memory(f"After loading model from {model_name}")

    return loaded_model, loaded_processor


def clear_gpu():
    """Flush the GPU caching allocator."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ====================================================
# TRACKING HELPERS
# ====================================================

def append_selected_samples_to_csv(records, filepath):
    """Append selected sample metadata to a CSV file."""
    if not records:
        return

    df = pd.DataFrame(records)
    write_header = not os.path.exists(filepath)
    df.to_csv(filepath, mode="a", header=write_header, index=False)
    print(f"Selection tracking updated: {filepath}")


def build_seed_tracking_records(dataset, strategy):
    """Create tracking records for the initial seed set."""
    records = []
    for sample in dataset:
        records.append(
            {
                "iteration": -1,
                "sample_id": sample["id"],
                "strategy": strategy,
                "selection_type": "initial_seed",
                "uncertainty": np.nan,
                "confidence": np.nan,
            }
        )
    return records


def build_acquisition_tracking_records(
    unlabeled_pool,
    acquire_indices,
    iteration,
    strategy,
    uncertainties=None,
    confidences=None,
):
    """Create tracking records for newly acquired samples."""
    records = []

    for pool_idx in acquire_indices.tolist():
        sample = unlabeled_pool[int(pool_idx)]

        if strategy == "active":
            record = {
                "iteration": iteration,
                "sample_id": sample["id"],
                "strategy": strategy,
                "selection_type": "acquired",
                "uncertainty": float(uncertainties[pool_idx]),
                "confidence": float(confidences[pool_idx]),
            }
        else:
            record = {
                "iteration": iteration,
                "sample_id": sample["id"],
                "strategy": strategy,
                "selection_type": "acquired",
                "uncertainty": np.nan,
                "confidence": np.nan,
            }

        records.append(record)

    return records


# ====================================================
# LOAD MODEL
# ====================================================

model, processor = load_model_and_processor(MODEL_NAME)

# ====================================================
# LOAD PARQUET DATA
# ====================================================

train_data = load_parquet_data_with_fallback(TRAIN_PATH)

print("Decoding images...")
dataset = train_data.map(decode_parquet_image_example, desc="Decoding images")

print("Cleaning dataset columns...")
dataset = dataset.remove_columns(
    [col for col in dataset.column_names if col not in ["image", "text", "id"]]
)

print("\nDEBUG: First 5 rows after image decoding:")
for i in range(min(5, len(dataset))):
    sample = dataset[i]
    print(f"Row {i}:")
    print(f"  ID: {sample['id']}")
    print(
        f"  Text: {sample['text'][:50]}..."
        if len(sample['text']) > 50
        else f"  Text: {sample['text']}"
    )
    print(f"  Image type: {type(sample['image'])}")
    print(
        f"  Image size: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}"
    )
    print(
        f"  Image mode: {sample['image'].mode if hasattr(sample['image'], 'mode') else 'N/A'}"
    )
    print()

# ====================================================
# ACTIVE LEARNING SETUP
# ====================================================

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
    data_list = []
    for item in dataset:
        img_buffer = io.BytesIO()
        image = item["image"]
        if hasattr(image, "copy") and hasattr(image, "thumbnail"):
            image = image.copy()
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        data_list.append(
            {
                "Identifier": item["id"],
                "Image": {"bytes": img_bytes},
                "Text": item["text"],
            }
        )
    df = pd.DataFrame(data_list)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def call_training_script(train_parquet_path, iteration, model_name):
    """Call the unified_qwen25_finetune.py script for training."""
    output_dir = f"{OUTPUT_DIR}/iteration_{iteration}"

    cmd = [
        "python",
        "./unified_qwen25_finetune.py",
        "--model_name",
        model_name,
        "--train_data",
        train_parquet_path,
        "--output_dir",
        output_dir,
        "--epochs",
        "1",
        "--train_batch_size",
        "1",
        "--lr",
        "5e-6",
        "--accumulation_steps",
        "8",
        "--user_text",
        "Recognize the text in the image.",
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
    ]

    logger.info(f"Running training script: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        subprocess.run(cmd, check=True, env=env)
        logger.info("Training completed successfully")
        return os.path.join(output_dir, "final_model")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training script failed with exit code: {e.returncode}")
        raise


# ====================================================
# UNCERTAINTY (CONFIDENCE-BASED)
# ====================================================

def clean_prediction_text(text):
    """Remove common chat-template artifacts from decoded predictions."""
    if not text:
        return ""
    cleaned = text.strip()
    lines = cleaned.split("\n")
    kept = []
    for line in lines:
        s = line.strip()
        if s.lower() in {"user", "assistant", "<|im_start|>", "<|im_end|>"}:
            continue
        if s:
            kept.append(s)
    return " ".join(kept).strip()


def save_uncertainty_results_to_csv(
    ground_truths, predictions, confidences, uncertainties, sample_ids, iteration=None
):
    """Save uncertainty computation results to CSV file."""
    results_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "ground_truth": ground_truths,
            "prediction": predictions,
            "confidence": confidences,
            "uncertainty": uncertainties,
        }
    )
    iteration_suffix = f"_iteration_{iteration}" if iteration is not None else ""
    csv_filename = f"uncertainty_results_{EXPERIMENT_NAME}{iteration_suffix}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Uncertainty results saved to: {csv_filename}")


def compute_uncertainty(model, processor, dataset, batch_size=2, iteration=None):
    """Compute uncertainty scores using memory-efficient logprob reconstruction."""
    model.eval()

    scores = []
    confidences = []
    ground_truths = []
    predictions = []
    sample_ids = []
    error_count = 0
    success_count = 0

    print(f"Computing uncertainty for {len(dataset)} samples using batch_size={batch_size}...")

    special_ids = build_special_token_ids(processor.tokenizer)

    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Computing uncertainty"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

        try:
            batch_inputs = []

            for sample in batch_samples:
                image = sample["image"]
                if hasattr(image, "copy") and hasattr(image, "thumbnail"):
                    image = image.copy()
                    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": "Transcribe the Jawi script in this image into Jawi text",
                            },
                        ],
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

            max_length = max(inp.input_ids.shape[1] for inp in batch_inputs)

            input_ids_list = []
            attention_mask_list = []
            pixel_values_list = []
            image_grid_thw_list = []

            for inp in batch_inputs:
                seq_len = inp.input_ids.shape[1]
                pad_length = max_length - seq_len

                if pad_length > 0:
                    input_ids = torch.nn.functional.pad(
                        inp.input_ids,
                        (pad_length, 0),
                        value=processor.tokenizer.pad_token_id,
                    )
                    attention_mask = torch.nn.functional.pad(
                        inp.attention_mask,
                        (pad_length, 0),
                        value=0,
                    )
                else:
                    input_ids = inp.input_ids
                    attention_mask = inp.attention_mask

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)

                if hasattr(inp, "pixel_values") and inp.pixel_values is not None:
                    pixel_values_list.append(inp.pixel_values)
                if hasattr(inp, "image_grid_thw") and inp.image_grid_thw is not None:
                    image_grid_thw_list.append(inp.image_grid_thw)

            batched_inputs = {
                "input_ids": torch.cat(input_ids_list, dim=0).to(device),
                "attention_mask": torch.cat(attention_mask_list, dim=0).to(device),
            }

            if pixel_values_list:
                batched_inputs["pixel_values"] = torch.cat(pixel_values_list, dim=0).to(device)
            if image_grid_thw_list:
                batched_inputs["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0).to(device)

            input_len = batched_inputs["input_ids"].shape[1]

            with torch.no_grad():
                gen_out = model.generate(
                    **batched_inputs,
                    max_new_tokens=64,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            gen_sequences = gen_out.sequences
            for i in range(len(batch_samples)):
                gen_tokens = gen_sequences[i, input_len:]
                generated_tokens_list = gen_tokens.tolist()

                scores_list = []
                for step_idx in range(len(gen_out.scores)):
                    if step_idx < len(generated_tokens_list):
                        scores_list.append(gen_out.scores[step_idx][i : i + 1])

                confidence = calculate_confidence_from_scores(
                    scores=scores_list,
                    generated_tokens=generated_tokens_list,
                    ignore_token_ids=special_ids,
                )

                uncertainty = 1.0 - confidence

                confidences.append(confidence)
                scores.append(uncertainty)

                ground_truths.append(batch_samples[i]["text"])
                sample_ids.append(batch_samples[i]["id"])

                decoded = processor.tokenizer.decode(
                    gen_tokens,
                    skip_special_tokens=True,
                )
                predictions.append(clean_prediction_text(decoded))

            success_count += len(batch_samples)

            del batched_inputs
            del gen_out
            clear_gpu()

        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")
            error_count += len(batch_samples)

            for sample in batch_samples:
                scores.append(1.0)
                confidences.append(0.0)
                ground_truths.append(sample["text"])
                predictions.append("")
                sample_ids.append(sample["id"])

            clear_gpu()

    print(f"DEBUG: Processed {success_count} samples successfully, {error_count} samples failed")
    print(f"DEBUG: Final uncertainty range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"DEBUG: Final confidence range: {min(confidences):.4f} - {max(confidences):.4f}")

    save_uncertainty_results_to_csv(
        ground_truths, predictions, confidences, scores, sample_ids, iteration
    )

    model.train()
    return np.array(scores), np.array(confidences)


# ====================================================
# ACQUISITION STRATEGY
# ====================================================

def select_acquisition_indices(unlabeled_pool, n_acquire, strategy, iteration, uncertainties=None):
    """Select indices from the unlabeled pool according to the acquisition strategy."""
    n_pool = len(unlabeled_pool)
    n_acquire = min(n_acquire, n_pool)

    if n_acquire <= 0:
        return np.array([], dtype=int)

    if strategy == "active":
        if uncertainties is None:
            raise ValueError("uncertainties must be provided for active acquisition")
        return np.argsort(-uncertainties)[:n_acquire]

    if strategy == "random":
        rng_seed = SEED + 10000 * RUN_ID + iteration
        rng = np.random.default_rng(rng_seed)
        return rng.choice(n_pool, size=n_acquire, replace=False)

    raise ValueError(f"Unknown acquisition strategy: {strategy}")


# ====================================================
# ACTIVE LEARNING / RANDOM SAMPLING LOOP
# ====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# reset tracking file at the start of a run
if os.path.exists(SELECTION_TRACKING_CSV):
    os.remove(SELECTION_TRACKING_CSV)

# save initial seed set tracking
seed_tracking_records = build_seed_tracking_records(
    dataset=labeled_set,
    strategy=ACQUISITION_STRATEGY,
)
append_selected_samples_to_csv(seed_tracking_records, SELECTION_TRACKING_CSV)

current_model_path = MODEL_NAME

print(f"Running experiment: {EXPERIMENT_NAME}")
print(f"Acquisition strategy: {ACQUISITION_STRATEGY}")
print(f"Seed: {SEED}")
print(f"Run ID: {RUN_ID}")
print(f"Selection tracking CSV: {SELECTION_TRACKING_CSV}")

for iteration in range(NUM_ITERATIONS):
    print(f"\n===== ITERATION {iteration} =====")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        train_parquet_path = tmp_file.name

    save_dataset_to_parquet(labeled_set, train_parquet_path)

    try:
        logger.info("Deleting model and processor before training subprocess...")
        del model
        del processor
        model = None
        processor = None
        clear_gpu()

        if torch.cuda.is_available():
            log_cuda_memory("GPU before training subprocess")

        trained_model_path = call_training_script(
            train_parquet_path, iteration, current_model_path
        )

        model, processor = load_model_and_processor(trained_model_path)
        clear_gpu()
        current_model_path = trained_model_path

    finally:
        if os.path.exists(train_parquet_path):
            os.unlink(train_parquet_path)

    if len(unlabeled_pool) == 0:
        break

    n_acquire = int(len(unlabeled_pool) * ACQUISITION_FRACTION)
    n_acquire = max(1, n_acquire)

    if ACQUISITION_STRATEGY == "active":
        uncertainties, confidences = compute_uncertainty(
            model,
            processor,
            unlabeled_pool,
            batch_size=UNCERTAINTY_BATCH_SIZE,
            iteration=iteration,
        )

        print(
            f"DEBUG: Uncertainty stats - min: {uncertainties.min():.4f}, "
            f"max: {uncertainties.max():.4f}, mean: {uncertainties.mean():.4f}"
        )
        print(
            f"DEBUG: Confidence stats - min: {confidences.min():.4f}, "
            f"max: {confidences.max():.4f}, mean: {confidences.mean():.4f}"
        )
        print(
            f"DEBUG: Number of samples with uncertainty = 1.0: "
            f"{(uncertainties == 1.0).sum()}/{len(uncertainties)}"
        )

        acquire_indices = select_acquisition_indices(
            unlabeled_pool=unlabeled_pool,
            n_acquire=n_acquire,
            strategy=ACQUISITION_STRATEGY,
            iteration=iteration,
            uncertainties=uncertainties,
        )

        selected_stat = f"{uncertainties[acquire_indices].mean():.4f}"
        selected_stat_name = "Mean uncertainty selected"

        selected_records = build_acquisition_tracking_records(
            unlabeled_pool=unlabeled_pool,
            acquire_indices=acquire_indices,
            iteration=iteration,
            strategy=ACQUISITION_STRATEGY,
            uncertainties=uncertainties,
            confidences=confidences,
        )

    elif ACQUISITION_STRATEGY == "random":
        acquire_indices = select_acquisition_indices(
            unlabeled_pool=unlabeled_pool,
            n_acquire=n_acquire,
            strategy=ACQUISITION_STRATEGY,
            iteration=iteration,
        )

        selected_stat = "N/A (random selection)"
        selected_stat_name = "Selection metric"

        selected_records = build_acquisition_tracking_records(
            unlabeled_pool=unlabeled_pool,
            acquire_indices=acquire_indices,
            iteration=iteration,
            strategy=ACQUISITION_STRATEGY,
        )

    else:
        raise ValueError(f"Unsupported acquisition strategy: {ACQUISITION_STRATEGY}")

    append_selected_samples_to_csv(selected_records, SELECTION_TRACKING_CSV)

    new_samples = unlabeled_pool.select(acquire_indices.tolist())
    labeled_set = concatenate_datasets([labeled_set, new_samples])

    remaining_indices = sorted(set(range(len(unlabeled_pool))) - set(acquire_indices.tolist()))
    unlabeled_pool = unlabeled_pool.select(remaining_indices)

    print(f"Added {len(new_samples)} samples.")
    print(f"{selected_stat_name}: {selected_stat}")
    print(f"Total labeled samples: {len(labeled_set)}")
    print(f"Remaining unlabeled samples: {len(unlabeled_pool)}")

print(f"{ACQUISITION_STRATEGY.capitalize()} learning run complete.")