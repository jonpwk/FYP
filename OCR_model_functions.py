#!/usr/bin/env python3
"""Shared OCR model loading and prediction helpers."""

from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig

from helper_functions import calculate_confidence_from_scores, build_special_token_ids


DEFAULT_PROMPT = "Transcribe the Jawi script in this image into Jawi text"
DEFAULT_CONFIG_MODEL = "culturalheritagenus/Jawi-OCR-Qwen-v2"
DEFAULT_PROCESSOR_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class OCRModelFunctions:
    """Shared OCR model wrapper used across labeling and evaluation scripts."""

    def __init__(
        self,
        model_name: str,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 128,
        config_model_name: str = DEFAULT_CONFIG_MODEL,
        processor_model_name: str = DEFAULT_PROCESSOR_MODEL,
        enable_compile: bool = True,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.config_model_name = config_model_name
        self.processor_model_name = processor_model_name
        self.enable_compile = enable_compile

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = None
        self.processor = None
        self.config = None

    def _load_model(self):
        """Load only model/config for current model_name."""
        # Load config from the checkpoint itself so weights and config always match.
        # Fall back to the default config model if the checkpoint has no config.
        try:
            self.config = AutoConfig.from_pretrained(self.model_name)
        except Exception:
            self.config = AutoConfig.from_pretrained(self.config_model_name)
        self.config.rope_scaling = {
            "type": "default",
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
        }

        # Load directly in the target dtype to avoid a 2x VRAM spike from conversion.
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            config=self.config,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )

        if self.device == "cuda" and self.enable_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass

    def load(self):
        """Load model and processor with common settings used in this codebase."""
        self._load_model()

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.processor_model_name,
                trust_remote_code=True,
                padding_side="left",
            )

            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.padding_side = "left"

    def unload_model(self):
        """Unload only model/config while keeping processor/tokenizer cached."""
        try:
            if self.model is not None:
                del self.model
        finally:
            self.model = None
            self.config = None

    def reload_model(self, model_name: str):
        """Reload model weights for a new checkpoint while reusing processor."""
        self.unload_model()
        self.model_name = model_name
        self._load_model()

    def _build_inputs(self, image: Image.Image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

    @torch.no_grad()
    def predict_pil_with_confidence(self, image: Image.Image) -> Tuple[str, float]:
        """Predict OCR text and confidence for one PIL image."""
        inputs = self._build_inputs(image)

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        sequences = gen_out.sequences
        input_len = inputs.input_ids.shape[1]
        gen_token_ids = sequences[0][input_len:]

        output_text = self.processor.tokenizer.decode(gen_token_ids, skip_special_tokens=True)

        token_scores = [s[0].unsqueeze(0) for s in gen_out.scores]
        generated_tokens_list = gen_token_ids.tolist() if hasattr(gen_token_ids, "tolist") else list(gen_token_ids)
        special_ids = build_special_token_ids(self.processor.tokenizer)

        confidence = calculate_confidence_from_scores(
            token_scores,
            generated_tokens_list,
            ignore_token_ids=special_ids,
        )

        return output_text, float(confidence)

    @torch.no_grad()
    def predict_with_confidence(self, image_bytes: bytes) -> Tuple[str, float]:
        """Predict OCR text and confidence for one image in bytes format."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.predict_pil_with_confidence(image)

    @torch.no_grad()
    def _process_single_batch(self, images_bytes: List[bytes]) -> List[Tuple[str, float]]:
        """Batch helper used by label/inverse-label workflows."""
        images = [Image.open(BytesIO(b)).convert("RGB") for b in images_bytes]
        results = []

        for image in images:
            try:
                result = self.predict_pil_with_confidence(image)
            except Exception:
                result = ("", 0.0)
            results.append(result)

        return results

    @torch.no_grad()
    def predict_batch_efficient(self, images_bytes: List[bytes], batch_size: int = 4) -> List[Tuple[str, float]]:
        """Process images in small batches for stable memory usage."""
        results = []
        total_images = len(images_bytes)

        for i in range(0, total_images, batch_size):
            batch = images_bytes[i:i + batch_size]
            results.extend(self._process_single_batch(batch))

        return results
