from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.tokenizer_utils import whitespace_tokenize
from chemtexteller.utils import ensure_dir, save_json, setup_logging, write_jsonl


logger = setup_logging()

DEFAULT_DATASET_ID = "ConstantHao/EDU-CHEMC_MM23"
DEFAULT_TARGET_FIELDS = ("chemfig", "chemfg", "ssml_sd", "ssml_normed", "ssml_rcgd")
SPLIT_MAP = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the Hugging Face EDU-CHEMC dataset into this repo's imagefolder format."
    )
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed/edu_chemc_normed"))
    parser.add_argument("--target_field", type=str, default="ssml_normed")
    parser.add_argument(
        "--allow_rcgd",
        action="store_true",
        help="Allow ssml_rcgd despite it being unsuitable for a normal sequence decoder.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="HF split names to materialize. HF 'val' is written as repo split 'validation'.",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional smoke-test limit per split; omit for the full dataset.",
    )
    parser.add_argument(
        "--image_format",
        choices=["source", "png", "jpg", "jpeg"],
        default="source",
        help="Use source image suffix from image_path, or force a specific saved format.",
    )
    parser.add_argument("--image_quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output split directories before writing them.",
    )
    return parser.parse_args()


def load_dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    if args.revision:
        kwargs["revision"] = args.revision
    if args.cache_dir:
        kwargs["cache_dir"] = str(args.cache_dir)
    return kwargs


def ensure_hf_cache_dirs(args: argparse.Namespace) -> None:
    if args.cache_dir:
        ensure_directory(args.cache_dir, "HF datasets cache")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    ensure_directory(hf_home, "HF_HOME")
    ensure_directory(hf_home / "hub", "HF hub cache")
    ensure_directory(hf_home / "datasets", "HF datasets cache")


def ensure_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:
        raise SystemExit(
            f"{label} path exists but is not a directory: {path}. "
            "Set HF_HOME and --cache_dir to a writable directory."
        ) from exc
    if not path.is_dir():
        raise SystemExit(
            f"{label} path exists but is not a directory: {path}. "
            "Set HF_HOME and --cache_dir to a writable directory."
        )


def hf_download_error_message(args: argparse.Namespace, exc: FileNotFoundError) -> str:
    offline_vars = [
        name
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
        if os.environ.get(name)
    ]
    hint = ""
    if offline_vars:
        hint = f"\nOffline env vars are set: {', '.join(offline_vars)}. Unset them to download."
    cache_hint = "\nIf this server uses a custom disk, pass: --cache_dir /path/to/hf_cache"
    return (
        f"Could not download Hugging Face dataset {args.dataset_id!r}.\n"
        "Check internet/proxy access to https://huggingface.co and Hugging Face offline env vars."
        f"{hint}{cache_hint}\nOriginal error: {exc}"
    )


def normalize_target(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        if all(isinstance(item, str) for item in raw):
            return " ".join(item.strip() for item in raw if item.strip()).strip()
        return json.dumps(raw, ensure_ascii=False)
    return str(raw).strip()


def collect_targets(row: dict[str, Any], target_field: str) -> dict[str, str]:
    fields = set(DEFAULT_TARGET_FIELDS)
    fields.add(target_field)
    targets: dict[str, str] = {}
    for field in sorted(fields):
        raw = row.get(field)
        if raw is None and field == "chemfig":
            raw = row.get("chemfg")
        if raw is None and field == "chemfg":
            raw = row.get("chemfig")
        value = normalize_target(raw)
        if value:
            targets[field] = value
    if "chemfg" in targets and "chemfig" not in targets:
        targets["chemfig"] = targets["chemfg"]
    if "chemfig" in targets and "chemfg" not in targets:
        targets["chemfg"] = targets["chemfig"]
    return targets


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1 - frac) + sorted_values[high] * frac)


def repo_split_name(hf_split: str) -> str:
    if hf_split not in SPLIT_MAP:
        raise ValueError(
            f"Unsupported split {hf_split!r}; expected one of {sorted(SPLIT_MAP)}."
        )
    return SPLIT_MAP[hf_split]


def clean_leaf_name(value: str, fallback_stem: str) -> str:
    leaf = value.replace("\\", "/").rsplit("/", 1)[-1].strip()
    leaf = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf)
    if not leaf or leaf in {".", ".."}:
        return fallback_stem
    return leaf


def resolve_image_name(row: dict[str, Any], split: str, idx: int, image_format: str) -> str:
    fallback_stem = f"{split}_{idx:06d}"
    raw_path = row.get("image_path")
    source_name = clean_leaf_name(str(raw_path), fallback_stem) if raw_path else fallback_stem
    stem = Path(source_name).stem or fallback_stem
    source_suffix = Path(source_name).suffix.lower()
    if image_format == "source":
        suffix = source_suffix if source_suffix in IMAGE_SUFFIXES else ".jpg"
    else:
        suffix = f".{image_format}"
    return f"{stem}{suffix}"


def unique_name(name: str, used: set[str], split: str, idx: int) -> str:
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    deduped = f"{stem}_{split}_{idx:06d}{suffix}"
    used.add(deduped)
    return deduped


def save_image(raw_image: Any, dst_path: Path, image_quality: int) -> None:
    ensure_dir(dst_path.parent)
    if isinstance(raw_image, dict):
        raw_bytes = raw_image.get("bytes")
        if isinstance(raw_bytes, bytes):
            dst_path.write_bytes(raw_bytes)
            return
        raw_path = raw_image.get("path")
        if raw_path:
            shutil.copy2(raw_path, dst_path)
            return
    if not hasattr(raw_image, "save"):
        raise TypeError(f"Unsupported image value: {type(raw_image).__name__}")

    image = raw_image
    save_kwargs: dict[str, Any] = {}
    if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
        if getattr(image, "mode", None) not in {"RGB", "L"}:
            image = image.convert("RGB")
        save_kwargs["quality"] = image_quality
    image.save(dst_path, **save_kwargs)


def reset_split_dir(split_dir: Path, overwrite: bool) -> None:
    metadata_path = split_dir / "metadata.jsonl"
    if split_dir.exists() and overwrite:
        shutil.rmtree(split_dir)
    elif metadata_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing prepared split: {metadata_path}. Pass --overwrite."
        )
    ensure_dir(split_dir)


def materialize_split(
    split_dataset: Any,
    hf_split: str,
    repo_split: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_dir = args.out_dir / repo_split
    reset_split_dir(split_dir, overwrite=args.overwrite)

    metadata_rows: list[dict[str, Any]] = []
    token_counts: Counter[str] = Counter()
    target_lengths: list[int] = []
    used_names: set[str] = set()
    skipped_missing_target = 0
    skipped_empty_target = 0
    skipped_missing_image = 0

    total = len(split_dataset)
    if args.max_samples_per_split is not None:
        total = min(total, args.max_samples_per_split)

    for idx in tqdm(range(total), desc=f"Materializing {hf_split}->{repo_split}"):
        row = split_dataset[idx]
        if not isinstance(row, dict):
            row = dict(row)
        raw_target = row.get(args.target_field)
        if raw_target is None:
            skipped_missing_target += 1
            continue
        target = normalize_target(raw_target)
        if not target:
            skipped_empty_target += 1
            continue
        raw_image = row.get("image")
        if raw_image is None:
            skipped_missing_image += 1
            continue

        image_name = unique_name(
            resolve_image_name(row, repo_split, idx, args.image_format),
            used=used_names,
            split=repo_split,
            idx=idx,
        )
        save_image(raw_image, split_dir / image_name, image_quality=args.image_quality)

        units = whitespace_tokenize(target)
        token_counts.update(units)
        target_lengths.append(len(units))
        targets = collect_targets(row, args.target_field)
        targets[args.target_field] = target
        metadata_rows.append(
            {
                "file_name": image_name,
                "image_name": image_name,
                "target": target,
                "target_field": args.target_field,
                "targets": targets,
            }
        )

    write_jsonl(metadata_rows, split_dir / "metadata.jsonl")
    return {
        "hf_split": hf_split,
        "repo_split": repo_split,
        "input_rows": total,
        "valid_samples": len(metadata_rows),
        "skipped_missing_target": skipped_missing_target,
        "skipped_empty_target": skipped_empty_target,
        "skipped_missing_image": skipped_missing_image,
        "target_length_mean": mean(target_lengths) if target_lengths else 0.0,
        "target_length_p50": median(target_lengths) if target_lengths else 0.0,
        "target_length_p95": percentile(target_lengths, 0.95),
        "target_length_max": max(target_lengths) if target_lengths else 0,
        "unique_token_count": len(token_counts),
        "top_100_tokens": token_counts.most_common(100),
    }


def main() -> None:
    args = parse_args()
    if args.target_field == "ssml_rcgd" and not args.allow_rcgd:
        raise SystemExit(
            "ssml_rcgd is a conditional/reconnection target and is not suitable for the "
            "baseline TexTeller sequence decoder. Pass --allow_rcgd only if you know what "
            "you are doing."
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("The 'datasets' package is required. Run `uv sync` or install repo deps.") from exc

    logger.info("Loading Hugging Face dataset %s", args.dataset_id)
    ensure_hf_cache_dirs(args)
    try:
        dataset = load_dataset(args.dataset_id, **load_dataset_kwargs(args))
    except FileNotFoundError as exc:
        raise SystemExit(hf_download_error_message(args, exc)) from exc
    if not hasattr(dataset, "keys"):
        raise SystemExit(f"Expected a DatasetDict with splits, got {type(dataset).__name__}.")

    split_stats: list[dict[str, Any]] = []
    for hf_split in args.splits:
        if hf_split not in dataset:
            available = ", ".join(str(name) for name in dataset.keys())
            raise SystemExit(f"Split {hf_split!r} not found in dataset. Available: {available}")
        repo_split = repo_split_name(hf_split)
        split_stats.append(materialize_split(dataset[hf_split], hf_split, repo_split, args))

    stats = {
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "revision": args.revision,
        "target_field": args.target_field,
        "official_splits_preserved": True,
        "split_mapping": {item["hf_split"]: item["repo_split"] for item in split_stats},
        "splits": split_stats,
        "total_valid_samples": sum(int(item["valid_samples"]) for item in split_stats),
    }
    save_json(stats, args.out_dir / "dataset_stats.json")
    logger.info("Materialized %s samples into %s", stats["total_valid_samples"], args.out_dir)


if __name__ == "__main__":
    main()
