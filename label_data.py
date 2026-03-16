#!/usr/bin/env python3
"""
Auto-labeling script using a Qwen OCR model with confidence gating.

Inputs
- model: Hugging Face model id (e.g., "culturalheritagenus/qwen-for-jawi-v1")
- input_parquet: Parquet file with columns [Identifier, Image, Text]
- threshold: float in [0,1]; if prediction confidence >= threshold, use prediction, else use ground truth

Output
- Parquet with columns [Identifier, Image, Text], where Text is chosen by the rule above.

Note: Confidence computation mirrors evaluate_model.py (geometric mean of per-token
probabilities over the generated continuation, ignoring special/template tokens).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Any

import pandas as pd
from OCR_model_functions import OCRModelFunctions
from data_loading_functions import load_parquet_dataframe, extract_rows_for_ocr


class QwenOCR(OCRModelFunctions):
	"""Backward-compatible wrapper around centralized OCR model functions."""


def auto_label(
	model_name: str,
	input_parquet: str | Path,
	threshold: float,
	output_parquet: str | Path | None = None,
	save_csv: bool = False,
	csv_path: str | Path | None = None,
	batch_size: int = 8,  # Increased default batch size
	) -> Path:
	"""Run labeling and write the resulting parquet.

	Returns the path to the written parquet file.
	"""
	input_parquet = Path(input_parquet)
	if output_parquet is None:
		stem = input_parquet.stem
		suffix = "-" + ("{:.3f}".format(threshold).rstrip("0").rstrip("."))
		output_parquet = input_parquet.with_name(f"{stem}{suffix}.parquet")
	else:
		output_parquet = Path(output_parquet)


	# Load data
	df = load_parquet_dataframe(input_parquet, required_columns=["Identifier", "Image", "Text"])

	# Only process the first 10 rows for speed
	df = df.head(10)
	print(f"[DEBUG] Limiting labeling to first 10 data points.")

	# Load model
	print("[DEBUG] Loading model and processor...")
	ocr = QwenOCR(model_name)
	ocr.load()
	print("[DEBUG] Model and processor loaded.")

	# Extract all data first
	all_image_bytes, all_gt, all_ids, all_images = extract_rows_for_ocr(df)
	
	print(f"[DEBUG] Processing {len(all_image_bytes)} images with batch_size={batch_size}")
	
	# Process all images in batches
	all_results = ocr.predict_batch_efficient(all_image_bytes, batch_size=batch_size)
	
	# Process results
	chosen_texts: List[str] = []
	preds: List[str] = []
	confs: List[float] = []
	chosen_src: List[str] = []
	chosen_ids: List[Any] = []
	chosen_images: List[Any] = []
	chosen_gt_list: List[str] = []
	
	for i, ((pred, conf), gt, identifier, image) in enumerate(zip(all_results, all_gt, all_ids, all_images)):
		chosen = pred if conf >= threshold and pred.strip() else gt
		chosen_texts.append(chosen)
		preds.append(pred)
		confs.append(float(conf))
		chosen_src.append("PRED" if chosen == pred and pred.strip() else "GT")
		chosen_ids.append(identifier)
		chosen_images.append(image)
		chosen_gt_list.append(gt)
		
		# Only show details for first few results
		if i < 3:
			print(f"[DEBUG] Result {i+1}: pred='{pred}', conf={conf:.3f}, chosen='{chosen}' ({'PRED' if chosen==pred else 'GT'})")

	out_df = pd.DataFrame(
		{
			"Identifier": chosen_ids,
			"Image": chosen_images,
			"Text": chosen_texts,
		}
	)

	out_df.to_parquet(output_parquet, index=False)
	print(f"\n✓ Wrote labeled parquet to: {output_parquet}")
	print(f"Number of data points in output parquet: {len(out_df)}")

	# Optional audit CSV with decision details
	if save_csv:
		if csv_path is None:
			csv_path = Path(output_parquet).with_suffix(".csv")
		else:
			csv_path = Path(csv_path)
		# Write all predictions, not just one per batch
		audit = pd.DataFrame(
			{
				"Identifier": chosen_ids,
				"ground_truth": chosen_gt_list,
				"prediction": preds,
				"confidence": confs,
				"chosen_text": chosen_texts,
				"chosen_source": chosen_src,  # 'GT' or 'PRED'
			}
		)
		audit.to_csv(csv_path, index=False)
		print(f"✓ Wrote audit CSV to: {csv_path}")
	return output_parquet


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Auto-label OCR data with a Qwen model and confidence gating")
	p.add_argument("--model", required=True, help="Hugging Face model id, e.g., culturalheritagenus/qwen-for-jawi-v1")
	p.add_argument("--input_parquet", required=True, help="Path to input parquet with Identifier/Image/Text")
	p.add_argument("--threshold", type=float, required=True, help="Confidence threshold in [0,1]")
	p.add_argument("--output_parquet", default=None, help="Optional output parquet path; defaults to <input>_labeled_thr-<t>.parquet")
	p.add_argument("--save_csv", action="store_true", help="If set, also save an audit CSV with confidences and chosen_source (GT/PRED)")
	p.add_argument("--csv_path", default=None, help="Optional explicit path for the audit CSV (defaults to output_parquet with .csv suffix)")
	p.add_argument("--batch_size", type=int, default=8, help="Batch size for processing images (default: 8)")
	return p.parse_args()


def main():
	args = parse_args()
	if not (0.0 <= args.threshold <= 1.0):
		raise ValueError("--threshold must be between 0 and 1")
	auto_label(
		args.model,
		args.input_parquet,
		args.threshold,
		args.output_parquet,
		save_csv=args.save_csv,
		csv_path=args.csv_path,
		batch_size=args.batch_size,
	)


if __name__ == "__main__":
	main()

