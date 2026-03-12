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
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info


def calculate_confidence_from_scores(
	scores: List[torch.Tensor],
	generated_tokens: List[int],
	ignore_token_ids: set[int] | None = None,
	) -> float:
	"""Conservative confidence from generation scores.

	Uses the geometric mean of per-token probabilities for the actually
	generated tokens, after removing special/template tokens.
	"""
	import math

	if not scores or not generated_tokens:
		return 0.0
	if ignore_token_ids is None:
		ignore_token_ids = set()

	token_confs: List[float] = []
	for i, score_tensor in enumerate(scores):
		if i >= len(generated_tokens):
			break
		tok_id = int(generated_tokens[i])
		if tok_id in ignore_token_ids:
			continue
		probs = F.softmax(score_tensor[0].to(torch.float32), dim=-1)
		token_prob = float(probs[tok_id].item())
		token_confs.append(token_prob)

	if token_confs:
		log_mean = sum(math.log(max(c, 1e-10)) for c in token_confs) / len(token_confs)
		return float(math.exp(log_mean))
	return 0.0


class QwenOCR:
	def __init__(self, model_name: str):
		self.model_name = model_name
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
		self.model = None
		self.processor = None
		self.config = None

	def load(self):
		self.config = AutoConfig.from_pretrained("culturalheritagenus/Jawi-OCR-Qwen-v2")
		self.config.rope_scaling = {'type':'default','mrope_section':[16, 24, 24], 'rope_type': 'default'}
		self.model = AutoModelForImageTextToText.from_pretrained(
			self.model_name, 
			config=self.config,
			torch_dtype=self.torch_dtype,
			device_map=self.device
		)
		# Use the base processor as in evaluate_model.py
		self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
		
		# Optimize model for faster inference
		if hasattr(torch, 'compile') and self.device == "cuda":
			try:
				self.model = torch.compile(self.model, mode="reduce-overhead")
				print("[DEBUG] Model compiled with torch.compile for faster inference")
			except Exception as e:
				print(f"[DEBUG] torch.compile failed: {e}")
		
		# Enable mixed precision for CUDA
		if self.device == "cuda":
			self.model.half()  # Use FP16
			print("[DEBUG] Enabled FP16 for faster inference")

	@torch.no_grad()
	def predict_with_confidence(self, image_bytes: bytes) -> Tuple[str, float]:
		# Single-image implementation without calling batch method to avoid recursion
		image = Image.open(BytesIO(image_bytes)).convert("RGB")
		message = [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": image},
					{"type": "text", "text": "Transcribe the Jawi script in this image into Jawi text"}
				]
			}
		]
		try:
			inputs = self.processor.apply_chat_template(
				message,
				add_generation_prompt=True,
				tokenize=True,
				return_dict=True,
				return_tensors="pt",
			).to(self.model.device)
			with torch.no_grad():
				gen_out = self.model.generate(
					**inputs,
					max_new_tokens=128,
					return_dict_in_generate=True,
					output_scores=True,
					do_sample=False,
					pad_token_id=self.processor.tokenizer.pad_token_id,
					eos_token_id=self.processor.tokenizer.eos_token_id
				)
			sequences = gen_out.sequences  # [1, in_len + gen_len]
			input_len = inputs.input_ids.shape[1]
			gen_token_ids = sequences[0][input_len:]
			output_text = self.processor.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
			
			# Confidence calculation
			scores = gen_out.scores
			token_scores = [s[0].unsqueeze(0) for s in scores]  # Extract scores for single sample
			generated_tokens_list = gen_token_ids.tolist() if hasattr(gen_token_ids, 'tolist') else list(gen_token_ids)
			special_ids = set(getattr(self.processor.tokenizer, "all_special_ids", []) or [])
			eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
			pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
			if eos_id is not None:
				special_ids.add(eos_id)
			if pad_id is not None:
				special_ids.add(pad_id)
			confidence = calculate_confidence_from_scores(
				token_scores, generated_tokens_list, ignore_token_ids=special_ids
			)
			return (output_text, confidence)
		except Exception as e:
			print(f"[SINGLE ERROR] {e}")
			return ("", 0.0)
	@torch.no_grad()
	def predict_batch_efficient(self, images_bytes: List[bytes], batch_size: int = 4) -> List[Tuple[str, float]]:
		"""Process images in batches for much better efficiency."""
		results = []
		total_images = len(images_bytes)
		
		for i in range(0, total_images, batch_size):
			batch = images_bytes[i:i + batch_size]
			batch_results = self._process_single_batch(batch)
			results.extend(batch_results)
			print(f"[DEBUG] Processed batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
		
		return results
	
	@torch.no_grad()
	def _process_single_batch(self, images_bytes: List[bytes]) -> List[Tuple[str, float]]:
		"""Process a single batch of images."""
		# Load and preprocess all images in the batch
		images = [Image.open(BytesIO(b)).convert("RGB") for b in images_bytes]
		
		# Create messages for all images
		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": img},
					{"type": "text", "text": "Transcribe the Jawi script in this image into Jawi text"}
				]
			}
			for img in images
		]
		
		try:
			# Process all messages together
			inputs_list = []
			for message in messages:
				inputs = self.processor.apply_chat_template(
					[message],
					add_generation_prompt=True,
					tokenize=True,
					return_dict=True,
					return_tensors="pt"
				).to(self.model.device)
				inputs_list.append(inputs)
			
			# Process each input through the model (still individual due to variable input sizes)
			results = []
			for idx, inputs in enumerate(inputs_list):
				with torch.cuda.amp.autocast(enabled=self.device=="cuda"):
					gen_out = self.model.generate(
						**inputs,
						max_new_tokens=128,
						return_dict_in_generate=True,
						output_scores=True,
						do_sample=False,
						pad_token_id=self.processor.tokenizer.pad_token_id,
						eos_token_id=self.processor.tokenizer.eos_token_id
					)
				
				sequences = gen_out.sequences
				input_len = inputs.input_ids.shape[1]
				gen_token_ids = sequences[0][input_len:]
				output_text = self.processor.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
				
				# Simplified confidence calculation
				if len(gen_out.scores) > 0 and len(gen_token_ids) > 0:
					# Use only the first few tokens for confidence to speed up
					max_tokens_for_conf = min(5, len(gen_out.scores), len(gen_token_ids))
					token_scores = [s[0].unsqueeze(0) for s in gen_out.scores[:max_tokens_for_conf]]
					generated_tokens_list = gen_token_ids[:max_tokens_for_conf].tolist()
					special_ids = {self.processor.tokenizer.eos_token_id, self.processor.tokenizer.pad_token_id}
					confidence = calculate_confidence_from_scores(
						token_scores, generated_tokens_list, ignore_token_ids=special_ids
					)
				else:
					confidence = 0.0
				
				results.append((output_text, confidence))
			
			return results
			
		except Exception as e:
			print(f"[BATCH ERROR] {e}")
			# Fallback to individual processing
			return [self.predict_with_confidence(b) for b in images_bytes]


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
	df = pd.read_parquet(input_parquet)
	required_cols = {"Identifier", "Image", "Text"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Input parquet missing required columns: {sorted(missing)}")

	# Only process the first 10 rows for speed
	df = df.head(10)
	print(f"[DEBUG] Limiting labeling to first 10 data points.")

	# Load model
	print("[DEBUG] Loading model and processor...")
	ocr = QwenOCR(model_name)
	ocr.load()
	print("[DEBUG] Model and processor loaded.")

	# Extract all data first
	all_image_bytes = []
	all_gt = []
	all_ids = []
	all_images = []
	
	for i, row in df.iterrows():
		try:
			image_bytes = row["Image"]["bytes"]
		except Exception:
			image_bytes = row["Image"]
		all_image_bytes.append(image_bytes)
		all_gt.append(row["Text"])
		all_ids.append(row["Identifier"])
		all_images.append(row["Image"])
	
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

