#!/usr/bin/env python3
"""Shared data loading helpers for parquet-based OCR workflows."""

from __future__ import annotations

from typing import Iterable, Any
import io
import os

import pandas as pd
from datasets import load_dataset, Dataset
from PIL import Image


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], context: str = "Input parquet") -> None:
    """Validate that a dataframe contains required columns."""
    required = set(required_columns)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{context} missing required columns: {sorted(missing)}")


def load_parquet_dataframe(parquet_path: str, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load parquet into DataFrame and optionally validate required columns."""
    df = pd.read_parquet(parquet_path)
    if required_columns is not None:
        validate_required_columns(df, required_columns)
    return df


def extract_image_bytes(image_data: Any) -> bytes:
    """Extract image bytes from supported parquet image formats."""
    if isinstance(image_data, dict) and "bytes" in image_data:
        return image_data["bytes"]
    return image_data


def load_image_as_rgb(image_data: Any) -> Image.Image:
    """Load parquet image payload into a PIL RGB image."""
    if hasattr(image_data, "convert"):
        return image_data.convert("RGB")
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    return Image.open(image_data).convert("RGB")


def extract_rows_for_ocr(df: pd.DataFrame):
    """Extract parallel arrays used by auto-labeling/inverse-labeling scripts."""
    all_image_bytes = []
    all_gt = []
    all_ids = []
    all_images = []

    for _, row in df.iterrows():
        image_bytes = extract_image_bytes(row["Image"])
        all_image_bytes.append(image_bytes)
        all_gt.append(row["Text"])
        all_ids.append(row["Identifier"])
        all_images.append(row["Image"])

    return all_image_bytes, all_gt, all_ids, all_images


def load_parquet_data_with_fallback(
    parquet_path: str,
    image_column: str = "Image",
    text_column: str = "Text",
    id_column: str = "Identifier",
):
    """Load parquet as HF dataset, falling back to pandas when needed."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    try:
        dataset = load_dataset("parquet", data_files=parquet_path)["train"]
        print(f"Loaded dataset with {len(dataset)} samples from {parquet_path}")
        return dataset
    except Exception as e:
        print(f"Error loading with datasets library: {e}")
        df = pd.read_parquet(parquet_path)
        validate_required_columns(df, [id_column, image_column, text_column], context="Parquet")
        dataset = Dataset.from_pandas(df)
        print(f"Loaded dataset with {len(dataset)} samples from {parquet_path} (pandas fallback)")
        return dataset


def decode_parquet_image_example(
    example,
    image_column: str = "Image",
    text_column: str = "Text",
    id_column: str = "Identifier",
):
    """Decode one parquet row to normalized OCR-friendly fields."""
    try:
        image_data = example[image_column]

        if hasattr(image_data, "convert"):
            image = image_data.convert("RGB")
        elif isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            image = Image.open(image_data).convert("RGB")

        example["image"] = image
        example["text"] = str(example[text_column]) if text_column in example else ""
        example["id"] = str(example[id_column]) if id_column in example else ""
        return example
    except Exception as e:
        print(f"Error decoding image: {e}")
        example["image"] = Image.new("RGB", (224, 224), color=(0, 0, 0))
        example["text"] = ""
        example["id"] = ""
        return example
