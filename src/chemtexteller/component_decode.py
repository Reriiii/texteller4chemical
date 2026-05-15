from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from PIL import Image

from .inference import generate_from_pixel_values, move_pixel_values


TOKEN_RE = re.compile(
    r"branch\(|branch\)|\\[A-Za-z]+|\?\[[^\]]+\]"
    r"|[-=~<>|_:]*\[:\s*-?\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\]"
    r"|[A-Za-z]+|\d+|[^\s]"
)
REACTION_MARKERS = (
    r"\rightarrow",
    r"\xrightarrow",
    r"\xrightleftharpoons",
    r"\rightleftharpoons",
    " + ",
)


@dataclass(frozen=True)
class TwoPassDecodeConfig:
    min_aspect_ratio: float = 2.4
    window_aspect_ratio: float = 2.2
    overlap_ratio: float = 0.20
    max_crops: int = 5
    min_crop_width: int = 128
    split_strategy: str = "ink_or_windows"
    min_gap_width: int = 10
    gap_ink_fraction: float = 0.01
    component_padding: int = 6
    min_component_width: int = 48
    crop_max_new_tokens: int | None = 512
    selection: str = "syntax_strict"
    min_length_gain_tokens: int = 20
    min_length_ratio: float = 0.70
    max_length_ratio: float = 1.25
    min_first_pass_tokens: int = 96
    trigger_reaction_markers: bool = True
    trigger_multi_chemfig: bool = True
    min_chemfig_count: int = 2
    trim_threshold: int = 15


@dataclass(frozen=True)
class SyntaxStats:
    token_count: int
    brace_delta: int
    paren_delta: int
    chemfig_count: int
    branch_count: int
    unk_count: int
    incomplete_tail: bool
    score: float

    @property
    def validish(self) -> bool:
        return (
            self.brace_delta == 0
            and self.paren_delta == 0
            and not self.incomplete_tail
            and self.chemfig_count > 0
        )


@dataclass(frozen=True)
class TwoPassDecodeResult:
    prediction: str
    first_pass_prediction: str
    stitched_prediction: str
    used: bool
    reasons: tuple[str, ...]
    crop_count: int
    component_predictions: tuple[str, ...]


def markup_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or ""))


def syntax_stats(text: str) -> SyntaxStats:
    text = str(text or "").strip()
    tokens = markup_tokens(text)
    tail = tokens[-1] if tokens else ""
    incomplete_tail = tail in {
        "\\",
        "{",
        "(",
        "[",
        "_",
        "^",
        "-",
        "=",
        "~",
        "branch",
        "\\chemfig",
        "\\overset",
        "\\underset",
    }
    brace_delta = text.count("{") - text.count("}")
    paren_delta = text.count("(") - text.count(")")
    chemfig_count = text.count(r"\chemfig")
    branch_count = tokens.count("branch") + tokens.count("branch(")
    unk_count = text.count(r"\unk")
    score = 0.0
    score -= 20.0 * abs(brace_delta)
    score -= 10.0 * abs(paren_delta)
    score -= 12.0 if incomplete_tail else 0.0
    score -= 3.0 * unk_count
    score += 4.0 if chemfig_count > 0 else -10.0
    score += min(len(tokens), 256) / 64.0
    return SyntaxStats(
        token_count=len(tokens),
        brace_delta=brace_delta,
        paren_delta=paren_delta,
        chemfig_count=chemfig_count,
        branch_count=branch_count,
        unk_count=unk_count,
        incomplete_tail=incomplete_tail,
        score=score,
    )


def _trim_white_border(image: Image.Image, threshold: int) -> Image.Image:
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    if arr.size == 0:
        return image
    ink = arr < max(0, 255 - threshold)
    if not ink.any():
        return image
    ys, xs = np.where(ink)
    left = max(0, int(xs.min()) - 2)
    top = max(0, int(ys.min()) - 2)
    right = min(image.width, int(xs.max()) + 3)
    bottom = min(image.height, int(ys.max()) + 3)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def _trigger_reasons(
    image: Image.Image,
    first_pass_prediction: str,
    cfg: TwoPassDecodeConfig,
) -> tuple[str, ...]:
    stats = syntax_stats(first_pass_prediction)
    width, height = image.size
    aspect_ratio = width / max(1, height)
    reasons: list[str] = []
    if aspect_ratio >= cfg.min_aspect_ratio:
        reasons.append(f"wide_image_aspect={aspect_ratio:.2f}")
    if cfg.min_first_pass_tokens > 0 and stats.token_count >= cfg.min_first_pass_tokens:
        reasons.append(f"long_first_pass_tokens={stats.token_count}")
    if cfg.trigger_multi_chemfig and stats.chemfig_count >= cfg.min_chemfig_count:
        reasons.append(f"multi_chemfig={stats.chemfig_count}")
    if cfg.trigger_reaction_markers and any(marker in first_pass_prediction for marker in REACTION_MARKERS):
        reasons.append("reaction_marker")
    if stats.brace_delta != 0:
        reasons.append(f"brace_delta={stats.brace_delta}")
    if stats.paren_delta != 0:
        reasons.append(f"paren_delta={stats.paren_delta}")
    if stats.incomplete_tail:
        reasons.append("incomplete_tail")
    return tuple(reasons)


def split_horizontal_windows(
    image: Image.Image,
    cfg: TwoPassDecodeConfig,
) -> list[Image.Image]:
    image = _trim_white_border(image, cfg.trim_threshold)
    width, height = image.size
    if width <= 0 or height <= 0:
        return []
    desired_width = max(cfg.min_crop_width, int(round(height * cfg.window_aspect_ratio)))
    if width <= desired_width * 1.15:
        return []
    crop_count = int(np.ceil(width / max(1, desired_width * (1.0 - cfg.overlap_ratio))))
    crop_count = max(2, min(cfg.max_crops, crop_count))
    if crop_count <= 1:
        return []
    if crop_count == 2:
        starts = [0, max(0, width - desired_width)]
    else:
        starts = [
            int(round(i * (width - desired_width) / (crop_count - 1)))
            for i in range(crop_count)
        ]

    crops: list[Image.Image] = []
    for left in starts:
        left = max(0, min(width - 1, left))
        right = min(width, left + desired_width)
        if right - left < cfg.min_crop_width:
            continue
        crops.append(image.crop((left, 0, right, height)))
    return crops


def _fill_short_false_runs(mask: np.ndarray, max_gap_width: int) -> np.ndarray:
    if max_gap_width <= 0 or mask.size == 0:
        return mask
    filled = mask.copy()
    idx = 0
    width = len(filled)
    while idx < width:
        if filled[idx]:
            idx += 1
            continue
        start = idx
        while idx < width and not filled[idx]:
            idx += 1
        end = idx
        if start > 0 and end < width and end - start <= max_gap_width:
            filled[start:end] = True
    return filled


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    idx = 0
    width = len(mask)
    while idx < width:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < width and mask[idx]:
            idx += 1
        runs.append((start, idx))
    return runs


def _merge_segments_to_limit(
    segments: list[tuple[int, int]],
    max_segments: int,
) -> list[tuple[int, int]]:
    segments = list(segments)
    while len(segments) > max_segments and len(segments) > 1:
        gaps = [
            (segments[idx + 1][0] - segments[idx][1], idx)
            for idx in range(len(segments) - 1)
        ]
        _, merge_idx = min(gaps, key=lambda item: item[0])
        left = segments[merge_idx]
        right = segments[merge_idx + 1]
        segments[merge_idx : merge_idx + 2] = [(left[0], right[1])]
    return segments


def _merge_narrow_segments(
    segments: list[tuple[int, int]],
    min_width: int,
) -> list[tuple[int, int]]:
    if len(segments) <= 1 or min_width <= 0:
        return segments
    merged: list[tuple[int, int]] = []
    for left, right in segments:
        width = right - left
        if merged and width < min_width:
            prev_left, _ = merged[-1]
            merged[-1] = (prev_left, right)
        else:
            merged.append((left, right))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_width:
        first = merged.pop(0)
        next_left, next_right = merged[0]
        merged[0] = (first[0], next_right)
    return merged


def split_ink_components(
    image: Image.Image,
    cfg: TwoPassDecodeConfig,
) -> list[Image.Image]:
    image = _trim_white_border(image, cfg.trim_threshold)
    width, height = image.size
    if width <= 0 or height <= 0:
        return []
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = arr < max(0, 255 - cfg.trim_threshold)
    if not ink.any():
        return []
    max_gap_ink = max(1, int(round(height * cfg.gap_ink_fraction)))
    column_has_ink = ink.sum(axis=0) > max_gap_ink
    column_has_ink = _fill_short_false_runs(column_has_ink, cfg.min_gap_width)
    runs = _mask_runs(column_has_ink)
    if len(runs) < 2:
        return []
    segments = [
        (
            max(0, left - cfg.component_padding),
            min(width, right + cfg.component_padding),
        )
        for left, right in runs
    ]
    segments = _merge_narrow_segments(segments, cfg.min_component_width)
    segments = _merge_segments_to_limit(segments, cfg.max_crops)
    if len(segments) < 2:
        return []
    return [image.crop((left, 0, right, height)) for left, right in segments]


def split_component_crops(
    image: Image.Image,
    cfg: TwoPassDecodeConfig,
) -> list[Image.Image]:
    strategy = cfg.split_strategy.lower()
    if strategy in {"ink", "ink_components"}:
        return split_ink_components(image, cfg)
    if strategy in {"windows", "horizontal_windows"}:
        return split_horizontal_windows(image, cfg)
    if strategy not in {"auto", "ink_or_windows"}:
        raise ValueError(f"Unsupported two-pass split_strategy: {cfg.split_strategy}")
    crops = split_ink_components(image, cfg)
    if len(crops) >= 2:
        return crops
    return split_horizontal_windows(image, cfg)


def _longest_token_overlap(
    left: list[str],
    right: list[str],
    min_overlap: int = 4,
    max_overlap: int = 80,
) -> int:
    limit = min(max_overlap, len(left), len(right))
    for size in range(limit, min_overlap - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def stitch_component_predictions(predictions: list[str]) -> str:
    token_sequences = [markup_tokens(prediction) for prediction in predictions if prediction.strip()]
    token_sequences = [tokens for tokens in token_sequences if tokens]
    if not token_sequences:
        return ""
    stitched = list(token_sequences[0])
    for tokens in token_sequences[1:]:
        overlap = _longest_token_overlap(stitched, tokens)
        stitched.extend(tokens[overlap:])
    return " ".join(stitched).strip()


def should_use_stitched_prediction(
    first_pass_prediction: str,
    stitched_prediction: str,
    cfg: TwoPassDecodeConfig,
) -> bool:
    selection = cfg.selection.lower()
    if selection == "always":
        return bool(stitched_prediction.strip())
    if selection in {"never", "off", "false"}:
        return False

    first_stats = syntax_stats(first_pass_prediction)
    stitched_stats = syntax_stats(stitched_prediction)
    if not _length_ratio_ok(first_stats, stitched_stats, cfg):
        return False

    if selection == "syntax_strict":
        return (
            not first_stats.validish
            and stitched_stats.validish
            and stitched_stats.score >= first_stats.score + 3
        )

    if selection == "syntax":
        return stitched_stats.score > first_stats.score

    if stitched_stats.score >= first_stats.score + 5:
        return True
    if not first_stats.validish and stitched_stats.score >= first_stats.score:
        return True
    length_gain = stitched_stats.token_count - first_stats.token_count
    if (
        length_gain >= cfg.min_length_gain_tokens
        and stitched_stats.score >= first_stats.score - 2
    ):
        return True
    return False


def _length_ratio_ok(
    first_stats: SyntaxStats,
    stitched_stats: SyntaxStats,
    cfg: TwoPassDecodeConfig,
) -> bool:
    if first_stats.token_count <= 0 or stitched_stats.token_count <= 0:
        return stitched_stats.token_count > 0
    ratio = stitched_stats.token_count / first_stats.token_count
    return cfg.min_length_ratio <= ratio <= cfg.max_length_ratio


def _crop_generation_kwargs(
    gen_kwargs: dict[str, object],
    cfg: TwoPassDecodeConfig,
) -> dict[str, object]:
    crop_kwargs = dict(gen_kwargs)
    if cfg.crop_max_new_tokens is not None and cfg.crop_max_new_tokens > 0:
        original = int(crop_kwargs.get("max_new_tokens", cfg.crop_max_new_tokens))
        crop_kwargs["max_new_tokens"] = min(original, int(cfg.crop_max_new_tokens))
    return crop_kwargs


def decode_components_for_image(
    *,
    image_path: str | Path,
    first_pass_prediction: str,
    model: torch.nn.Module,
    tokenizer: Any,
    transform: Any,
    gen_kwargs: dict[str, object],
    device: torch.device,
    dtype: torch.dtype | None,
    cfg: TwoPassDecodeConfig,
) -> TwoPassDecodeResult:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    trimmed = _trim_white_border(image, cfg.trim_threshold)
    reasons = _trigger_reasons(trimmed, first_pass_prediction, cfg)
    if not reasons:
        return TwoPassDecodeResult(
            prediction=first_pass_prediction,
            first_pass_prediction=first_pass_prediction,
            stitched_prediction="",
            used=False,
            reasons=(),
            crop_count=0,
            component_predictions=(),
        )

    crops = split_component_crops(trimmed, cfg)
    if len(crops) < 2:
        return TwoPassDecodeResult(
            prediction=first_pass_prediction,
            first_pass_prediction=first_pass_prediction,
            stitched_prediction="",
            used=False,
            reasons=reasons + ("no_component_split",),
            crop_count=len(crops),
            component_predictions=(),
        )

    tensors = [transform(crop) for crop in crops]
    pixel_values = torch.stack(tensors)
    pixel_values = move_pixel_values(pixel_values, device, dtype)
    generated = generate_from_pixel_values(
        model,
        pixel_values,
        _crop_generation_kwargs(gen_kwargs, cfg),
    )
    component_predictions = tuple(
        tokenizer.batch_decode(generated, skip_special_tokens=True)
    )
    stitched = stitch_component_predictions(list(component_predictions))
    use_stitched = should_use_stitched_prediction(first_pass_prediction, stitched, cfg)
    prediction = stitched if use_stitched else first_pass_prediction
    return TwoPassDecodeResult(
        prediction=prediction,
        first_pass_prediction=first_pass_prediction,
        stitched_prediction=stitched,
        used=use_stitched,
        reasons=reasons if use_stitched else reasons + ("kept_first_pass",),
        crop_count=len(crops),
        component_predictions=component_predictions,
    )
