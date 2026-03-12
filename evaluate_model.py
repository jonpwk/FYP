#!/usr/bin/env python3
"""
Model evaluation script for Qwen2.5 OCR model.
Evaluates the accuracy of culturalheritagenus/qwen-for-jawi-v1 on test data.

This script integrates with performance_metrics.py for improved Jawi script evaluation
including proper Unicode normalization, diacritics handling, and Arabic text processing.
Supports Qwen2.5-VL architecture models.
"""

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO
import time
from typing import List, Tuple, Dict
from performance_metrics import (
    levenshtein_distance,
    normalize_text,
    compute_mean_CER,
    compute_mean_WER,
    evaluate_ocr
)

def calculate_confidence_from_scores(scores, generated_tokens, ignore_token_ids=None):
    """
    Calculate confidence score from model output scores.
    
    Args:
        scores: List of logit tensors for each generation step
        generated_tokens: Generated token IDs
    
    Returns:
        float: Average confidence score (0-1, higher is more confident)
    """
    import torch
    import torch.nn.functional as F
    import math
    
    if not scores or len(generated_tokens) == 0:
        return 0.0
    
    if ignore_token_ids is None:
        ignore_token_ids = set()

    token_confidences = []
    
    for i, score_tensor in enumerate(scores):
        if i < len(generated_tokens):
            try:
                tok_id = int(generated_tokens[i])
                if tok_id in ignore_token_ids:
                    continue
                # Convert logits to probabilities (cast to float32 for stability)
                probs = F.softmax(score_tensor[0].to(torch.float32), dim=-1)
                # Get probability of the actually generated token
                token_prob = probs[tok_id].item()
                token_confidences.append(token_prob)
            except Exception as e:
                print(f"Error processing token {i}: {e}")
                continue
    
    # Return geometric mean of token probabilities (more conservative than arithmetic mean)
    if token_confidences:
        log_mean = sum(math.log(max(conf, 1e-10)) for conf in token_confidences) / len(token_confidences)
        return math.exp(log_mean)
    
    return 0.0

class ModelEvaluator:
    def __init__(self, model_name: str = "culturalheritagenus/Jawi-OCR-Qwen-v2"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self.config = None
        
    def load_model(self):
        """Load the model and processor."""
        print(f"Loading model: {self.model_name}")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load config and fix missing rope_scaling
        self.config = AutoConfig.from_pretrained("culturalheritagenus/Jawi-OCR-Qwen-v2")
        self.config.rope_scaling = {'type': 'default', 'mrope_section': [16, 24, 24], 'rope_type': 'default'}
        
        # Load model using AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, config=self.config)
        self.model = self.model.to(self.device)
        
        # Load processor from the same model instead of base Qwen model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        print("✓ Model and processor loaded successfully")
        
    def load_test_data(self, test_path: str) -> pd.DataFrame:
        """Load test data from parquet file."""
        print(f"Loading test data from: {test_path}")
        df = pd.read_parquet(test_path)
        print(f"✓ Loaded {len(df)} test samples")
        return df
        
    def predict_single_image(self, image: Image.Image, max_retries: int = 3) -> Tuple[str, float]:
        """Generate OCR prediction for a single image and return (text, confidence).

        Confidence is computed as exp(mean log-probability) over the generated tokens,
        where per-token log-probabilities are obtained from the model's output scores.
        """
        try:
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Transcribe the Jawi script in this image into Jawi text"}
                    ]
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            


            # Generate prediction
            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    return_dict_in_generate=True,
                    #temperature=1.0,
                    output_scores=True,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            # Full sequences include the prompt; trim to generated continuation (match test_new_model.py logic)
            sequences = gen_out.sequences  # [batch, in_len + gen_len]
            input_len = inputs.input_ids.shape[1]
            # Trim generated_ids by input length, as in test_new_model.py
            gen_token_ids = [out_ids[input_len:] for out_ids in sequences]
            # Decode generated text
            output_text = self.processor.tokenizer.batch_decode(gen_token_ids, skip_special_tokens=True)[0]

            # Compute confidence using geometric mean of per-token probabilities (retain original logic)
            generated_tokens_list = gen_token_ids[0].tolist() if hasattr(gen_token_ids[0], 'tolist') else list(gen_token_ids[0])

            # Exclude special/template tokens from scoring
            special_ids = set(getattr(self.processor.tokenizer, "all_special_ids", []) or [])
            eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if eos_id is not None:
                special_ids.add(eos_id)
            if pad_id is not None:
                special_ids.add(pad_id)

            confidence = calculate_confidence_from_scores(
                gen_out.scores, generated_tokens_list, ignore_token_ids=special_ids
            )

            return output_text, confidence

        except Exception as e:
            print(f"Error predicting image: {e}")
            if max_retries > 0:
                print(f"Retrying... ({max_retries} attempts left)")
                return self.predict_single_image(image, max_retries - 1)
            return "", 0.0
            
    
    def evaluate_dataset(self, df: pd.DataFrame, max_samples: int = None) -> Dict:
        """Evaluate the model on the entire dataset."""
        if max_samples:
            df = df.head(max_samples)
            print(f"Evaluating on first {max_samples} samples")
        
        predictions = []
        confidences = []
        ground_truths = []
        
        print(f"Starting evaluation on {len(df)} samples...")
        start_time = time.time()
        
        for i, row in df.iterrows():
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1) if i > 0 else 0
                remaining = (len(df) - i - 1) * avg_time
                print(f"Progress: {i+1}/{len(df)} ({(i+1)/len(df)*100:.1f}%) "
                      f"- Avg: {avg_time:.2f}s/sample - ETA: {remaining/60:.1f}min")
            
            try:
                # Get image bytes
                image_data = row['Image']
                image_bytes = image_data['bytes']
                # Convert bytes to PIL Image
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
                # Get ground truth
                ground_truth = row['Text']
                # Get prediction and confidence
                prediction, confidence = self.predict_single_image(image)
                predictions.append(prediction)
                confidences.append(confidence)
                ground_truths.append(ground_truth)
                # Print first few examples
                if i < 5:
                    print(f"\nSample {i+1}:")
                    print(f"  Ground Truth: {repr(ground_truth)}")
                    print(f"  Prediction:   {repr(prediction)}")
                    print(f"  Confidence:   {confidence:.4f}")
                    print(f"  Image Bytes Length: {len(image_bytes)}")

                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                predictions.append("")
                confidences.append(0.0)
                ground_truths.append(row['Text'])
        
        total_time = time.time() - start_time
        print(f"\n✓ Evaluation completed in {total_time/60:.2f} minutes")
        print(f"Average time per sample: {total_time/len(df):.2f} seconds")
        
        # Save predictions to CSV (with confidences)
        self.save_predictions_to_csv(ground_truths, predictions, confidences)
        
        # Calculate metrics
        return self.calculate_metrics(ground_truths, predictions)
    
    def calculate_metrics(self, ground_truths: List[str], predictions: List[str]) -> Dict:
        """Calculate evaluation metrics using improved functions from performance_metrics.py."""
        print("\nCalculating metrics...")
        
        # Calculate CER and WER using the improved functions
        cer = compute_mean_CER(ground_truths, predictions, normalize=True)
        wer = compute_mean_WER(ground_truths, predictions, normalize=True)
        
        # Calculate exact match accuracy (with normalization)
        exact_matches = []
        for gt, pred in zip(ground_truths, predictions):
            gt_norm = normalize_text(gt)
            pred_norm = normalize_text(pred)
            exact_matches.append(1.0 if gt_norm == pred_norm else 0.0)
        
        # Calculate BLEU-1 score (with normalization)
        bleu_scores = []
        for gt, pred in zip(ground_truths, predictions):
            gt_norm = normalize_text(gt)
            pred_norm = normalize_text(pred)
            gt_words = set(gt_norm.split())
            pred_words = set(pred_norm.split())
            
            if len(pred_words) == 0:
                bleu_scores.append(0.0)
            else:
                intersection = gt_words.intersection(pred_words)
                bleu_scores.append(len(intersection) / len(pred_words))
        
        metrics = {
            'exact_match_accuracy': sum(exact_matches) / len(exact_matches),
            'character_error_rate': cer,
            'word_error_rate': wer,
            'bleu_score': sum(bleu_scores) / len(bleu_scores),
            'total_samples': len(ground_truths),
            'perfect_predictions': sum(exact_matches)
        }
        
        return metrics
    
    def save_predictions_to_csv(self, ground_truths: List[str], predictions: List[str], confidences: List[float]):
        """Save predictions, confidences, and ground truths to CSV file."""
        results_df = pd.DataFrame({
            'ground_truth': ground_truths,
            'prediction': predictions,
            'confidence': confidences,
        })
        
        csv_filename = f"evaluation_results_{self.model_name.replace('/', '_')}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"✓ Predictions saved to: {csv_filename}")
    
    def print_results(self, metrics: Dict):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Total samples evaluated: {metrics['total_samples']}")
        print(f"Perfect predictions: {int(metrics['perfect_predictions'])}")
        print("-"*60)
        print("Metrics (with Jawi text normalization):")
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
        print(f"Character Error Rate:  {metrics['character_error_rate']:.4f}")
        print(f"Word Error Rate:       {metrics['word_error_rate']:.4f}")
        print(f"BLEU Score:           {metrics['bleu_score']:.4f}")
        print("="*60)
        
        # Interpretation
        print("\nInterpretation:")
        if metrics['exact_match_accuracy'] > 0.9:
            print("🟢 Excellent accuracy! The model performs very well.")
        elif metrics['exact_match_accuracy'] > 0.7:
            print("🟡 Good accuracy. Some room for improvement.")
        elif metrics['exact_match_accuracy'] > 0.5:
            print("🟠 Moderate accuracy. Significant improvement needed.")
        else:
            print("🔴 Low accuracy. Major improvements required.")
            
        if metrics['character_error_rate'] < 0.1:
            print("🟢 Very low character error rate.")
        elif metrics['character_error_rate'] < 0.3:
            print("🟡 Moderate character error rate.")
        else:
            print("🔴 High character error rate.")

def main():
    """Main evaluation function."""
    # Configuration
    #model_name = "culturalheritagenus/Jawi-OCR-Qwen-v2"  # Qwen2.5-based model
    model_name = "finetuning/models/Jawi-OCR-Qwen-v2-finetuned"  # Finetuned model
    test_data_path = "./data_v4/test"
    max_samples = None  # Set to None to evaluate all samples, or a number for quick testing
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_name)
    
    # Load model
    evaluator.load_model()
    
    # Load test data
    test_df = evaluator.load_test_data(test_data_path)
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(test_df, max_samples=max_samples)
    
    # Print results
    evaluator.print_results(metrics)
    
    return metrics

if __name__ == "__main__":
    metrics = main()