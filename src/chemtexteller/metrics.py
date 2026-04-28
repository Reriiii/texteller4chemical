from __future__ import annotations

from statistics import mean
from typing import Any

from rapidfuzz.distance import Levenshtein

from .tokenizer_utils import normalize_whitespace, whitespace_tokenize


def _token_distance(prediction: str, reference: str) -> int:
    return int(
        Levenshtein.distance(
            whitespace_tokenize(prediction),
            whitespace_tokenize(reference),
        )
    )


def _char_distance(prediction: str, reference: str) -> int:
    return int(Levenshtein.distance(prediction, reference))


def sequence_metrics(predictions: list[str], references: list[str]) -> dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {
            "num_samples": 0,
            "exact_match": 0.0,
            "normalized_exact_match": 0.0,
            "mean_token_edit_distance": 0.0,
            "mean_normalized_token_edit_distance": 0.0,
            "mean_char_edit_distance": 0.0,
            "average_target_length": 0.0,
            "average_prediction_length": 0.0,
        }

    exact = [pred == ref for pred, ref in zip(predictions, references)]
    normalized_exact = [
        normalize_whitespace(pred) == normalize_whitespace(ref)
        for pred, ref in zip(predictions, references)
    ]
    token_distances = [_token_distance(pred, ref) for pred, ref in zip(predictions, references)]
    char_distances = [_char_distance(pred, ref) for pred, ref in zip(predictions, references)]
    ref_lengths = [len(whitespace_tokenize(ref)) for ref in references]
    pred_lengths = [len(whitespace_tokenize(pred)) for pred in predictions]
    norm_token_distances = [
        dist / max(1, ref_len)
        for dist, ref_len in zip(token_distances, ref_lengths)
    ]

    return {
        "num_samples": len(predictions),
        "exact_match": sum(exact) / len(exact),
        "normalized_exact_match": sum(normalized_exact) / len(normalized_exact),
        "mean_token_edit_distance": mean(token_distances),
        "mean_normalized_token_edit_distance": mean(norm_token_distances),
        "mean_char_edit_distance": mean(char_distances),
        "average_target_length": mean(ref_lengths),
        "average_prediction_length": mean(pred_lengths),
    }


def per_sample_metrics(prediction: str, reference: str) -> dict[str, Any]:
    token_distance = _token_distance(prediction, reference)
    char_distance = _char_distance(prediction, reference)
    ref_len = len(whitespace_tokenize(reference))
    return {
        "exact_match": prediction == reference,
        "normalized_exact_match": normalize_whitespace(prediction)
        == normalize_whitespace(reference),
        "token_edit_distance": token_distance,
        "normalized_token_edit_distance": token_distance / max(1, ref_len),
        "char_edit_distance": char_distance,
    }
