#!/usr/bin/env python3
"""
Model evaluation script for Qwen2.5 OCR model.
Evaluates the accuracy of culturalheritagenus/qwen-for-jawi-v1 on test data.

This script integrates with performance_metrics.py for improved Jawi script evaluation
including proper Unicode normalization, diacritics handling, and Arabic text processing.
Supports Qwen2.5-VL architecture models.
"""

import pandas as pd
from PIL import Image
import time
from typing import List, Tuple, Dict
from OCR_model_functions import OCRModelFunctions
from data_loading_functions import load_parquet_dataframe, load_image_as_rgb
from performance_metrics import (
    normalize_text,
    compute_mean_CER,
    compute_mean_WER,
)

class ModelEvaluator:
    def __init__(self, model_name: str = "culturalheritagenus/Jawi-OCR-Qwen-v2"):
        self.model_name = model_name
        self.ocr = OCRModelFunctions(model_name)
        self.model = None
        self.processor = None
        self.device = None
        self.config = None
        
    def load_model(self):
        """Load the model and processor."""
        print(f"Loading model: {self.model_name}")
        self.ocr.load()
        self.model = self.ocr.model
        self.processor = self.ocr.processor
        self.device = self.ocr.device
        self.config = self.ocr.config
        print(f"Using device: {self.device}")
        
        print("✓ Model and processor loaded successfully")
        
    def load_test_data(self, test_path: str) -> pd.DataFrame:
        """Load test data from parquet file."""
        print(f"Loading test data from: {test_path}")
        df = load_parquet_dataframe(test_path)
        print(f"✓ Loaded {len(df)} test samples")
        return df
        
    def predict_single_image(self, image: Image.Image, max_retries: int = 3) -> Tuple[str, float]:
        """Generate OCR prediction for a single image and return (text, confidence).

        Confidence is computed as exp(mean log-probability) over the generated tokens,
        where per-token log-probabilities are obtained from the model's output scores.
        """
        try:
            return self.ocr.predict_pil_with_confidence(image)

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
                image = load_image_as_rgb(image_data)
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
                    print(f"  Image Type:   {type(image_data)}")

                    
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