"""Shared helper utilities for token-confidence computation."""

import math
from typing import Iterable, Optional, Set

import torch
import torch.nn.functional as F


def build_special_token_ids(tokenizer) -> Set[int]:
    """Build a set of special token ids to ignore during confidence scoring."""
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if eos_id is not None:
        special_ids.add(eos_id)
    if pad_id is not None:
        special_ids.add(pad_id)
    return special_ids


def calculate_confidence_from_scores(
    scores,
    generated_tokens: Iterable[int],
    ignore_token_ids: Optional[Set[int]] = None,
) -> float:
    """Calculate confidence as geometric mean of generated-token probabilities."""
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
                probs = F.softmax(score_tensor[0].to(torch.float32), dim=-1)
                token_prob = probs[tok_id].item()
                token_confidences.append(token_prob)
            except Exception:
                continue

    if token_confidences:
        log_mean = sum(math.log(max(conf, 1e-10)) for conf in token_confidences) / len(token_confidences)
        return math.exp(log_mean)

    return 0.0