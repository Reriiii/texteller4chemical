from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.data import IMAGE_EXTENSIONS
from chemtexteller.tokenizer_utils import whitespace_tokenize
from chemtexteller.utils import copy_or_symlink, ensure_dir, save_json, setup_logging, write_jsonl


logger = setup_logging()

DEFAULT_TARGET_FIELDS = ("ssml_sd", "ssml_normed", "chemfig", "chemfg", "ssml_rcgd")


def preview_directory(path: Path, max_items: int = 20) -> list[str]:
    if not path.exists():
        return ["<path does not exist>"]
    if not path.is_dir():
        return [f"<not a directory: {path}>"]
    items: list[str] = []
    for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        suffix = "/" if child.is_dir() else ""
        items.append(f"{child.name}{suffix}")
        if len(items) >= max_items:
            break
    if not items:
        return ["<empty directory>"]
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EDU-CHEMC imagefolder dataset.")
    parser.add_argument("--src_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed/edu_chemc"))
    parser.add_argument("--target_field", type=str, default="ssml_sd")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "symlink", "reference"],
        default="copy",
        help=(
            "copy duplicates images, symlink creates lightweight links, reference writes "
            "absolute source image paths into metadata without creating image files."
        ),
    )
    parser.add_argument("--max_target_units", type=int, default=None)
    parser.add_argument("--min_width", type=int, default=1)
    parser.add_argument("--min_height", type=int, default=1)
    parser.add_argument(
        "--allow_rcgd",
        action="store_true",
        help="Allow ssml_rcgd despite it being unsuitable for a normal sequence decoder.",
    )
    return parser.parse_args()


def normalize_target(raw: Any, target_field: str) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        if all(isinstance(item, str) for item in raw):
            return " ".join(item.strip() for item in raw if item.strip()).strip()
        return json.dumps(raw, ensure_ascii=False)
    return str(raw).strip()


def resolve_target(annotation: dict[str, Any], target_field: str) -> Any | None:
    if target_field in annotation:
        return annotation[target_field]
    if target_field == "chemfig" and "chemfg" in annotation:
        return annotation["chemfg"]
    if target_field == "chemfg" and "chemfig" in annotation:
        return annotation["chemfig"]
    return None


def collect_targets(annotation: dict[str, Any], target_field: str) -> dict[str, str]:
    fields = set(DEFAULT_TARGET_FIELDS)
    fields.add(target_field)
    targets: dict[str, str] = {}
    for field in sorted(fields):
        raw_target = resolve_target(annotation, field)
        if raw_target is None:
            continue
        target = normalize_target(raw_target, field)
        if target:
            targets[field] = target

    if "chemfg" in targets and "chemfig" not in targets:
        targets["chemfig"] = targets["chemfg"]
    if "chemfig" in targets and "chemfg" not in targets:
        targets["chemfg"] = targets["chemfig"]
    return targets


def validate_image(path: Path, min_width: int, min_height: int) -> tuple[int, int]:
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        width, height = img.size
    if width < min_width or height < min_height:
        raise ValueError(f"image too small: {width}x{height}")
    return width, height


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


def split_samples(samples: list[dict[str, Any]], val_ratio: float, test_ratio: float) -> dict[str, list[dict[str, Any]]]:
    n_total = len(samples)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = max(0, n_total - n_val - n_test)
    return {
        "train": samples[:n_train],
        "validation": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }


def main() -> None:
    args = parse_args()
    if args.target_field == "ssml_rcgd" and not args.allow_rcgd:
        raise SystemExit(
            "ssml_rcgd is a conditional/reconnection target and is not suitable for the "
            "baseline TexTeller sequence decoder. Pass --allow_rcgd only if you know what "
            "you are doing."
        )
    if args.target_field == "ssml_rcgd":
        logger.warning("Using ssml_rcgd. This is not recommended for a standard sequence decoder.")

    if not args.src_dir.exists():
        raise SystemExit(f"Source directory does not exist: {args.src_dir}")
    if not args.src_dir.is_dir():
        raise SystemExit(f"Source path is not a directory: {args.src_dir}")
    image_paths = sorted(
        path for path in args.src_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        preview = "\n  - ".join(preview_directory(args.src_dir))
        supported = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise SystemExit(
            f"No image files found under: {args.src_dir}\n"
            f"Supported extensions: {supported}\n"
            "This usually means the Kaggle input path is one level too high/low, "
            "or the dataset is still compressed as .zip/.rar/.7z.\n"
            f"Top-level entries in src_dir:\n  - {preview}"
        )
    stats: dict[str, Any] = {
        "total_image_files": len(image_paths),
        "valid_samples": 0,
        "skipped_missing_json": 0,
        "skipped_bad_json": 0,
        "skipped_missing_target": 0,
        "skipped_empty_target": 0,
        "skipped_bad_image": 0,
        "skipped_too_long": 0,
    }

    samples: list[dict[str, Any]] = []
    token_counts: Counter[str] = Counter()
    target_lengths: list[int] = []

    for image_path in tqdm(image_paths, desc="Scanning EDU-CHEMC"):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            stats["skipped_missing_json"] += 1
            continue
        try:
            annotation = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            stats["skipped_bad_json"] += 1
            continue
        raw_target = resolve_target(annotation, args.target_field)
        if raw_target is None:
            stats["skipped_missing_target"] += 1
            continue
        target = normalize_target(raw_target, args.target_field)
        if not target:
            stats["skipped_empty_target"] += 1
            continue
        units = whitespace_tokenize(target)
        if args.max_target_units is not None and len(units) > args.max_target_units:
            stats["skipped_too_long"] += 1
            continue
        try:
            validate_image(image_path, args.min_width, args.min_height)
        except Exception:
            stats["skipped_bad_image"] += 1
            continue

        targets = collect_targets(annotation, args.target_field)
        targets[args.target_field] = target
        sample = {
            "src_image": image_path,
            "target": target,
            "targets": targets,
            "length": len(units),
        }
        samples.append(sample)
        token_counts.update(units)
        target_lengths.append(len(units))

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    splits = split_samples(samples, args.val_ratio, args.test_ratio)

    global_idx = 1
    for split, split_samples_ in splits.items():
        split_dir = ensure_dir(args.out_dir / split)
        metadata_rows: list[dict[str, Any]] = []
        for sample in tqdm(split_samples_, desc=f"Writing {split}"):
            src_image = sample["src_image"]
            image_name = f"sample_{global_idx:06d}{src_image.suffix.lower()}"
            if args.copy_mode == "reference":
                file_name = str(src_image.resolve())
            else:
                file_name = image_name
                dst_path = split_dir / file_name
                copy_or_symlink(src_image, dst_path, args.copy_mode)
            metadata_rows.append(
                {
                    "file_name": file_name,
                    "image_name": image_name,
                    "target": sample["target"],
                    "target_field": args.target_field,
                    "targets": sample["targets"],
                }
            )
            global_idx += 1
        write_jsonl(metadata_rows, split_dir / "metadata.jsonl")

    stats.update(
        {
            "valid_samples": len(samples),
            "train_size": len(splits["train"]),
            "validation_size": len(splits["validation"]),
            "test_size": len(splits["test"]),
            "target_length_mean": mean(target_lengths) if target_lengths else 0.0,
            "target_length_p50": median(target_lengths) if target_lengths else 0.0,
            "target_length_p95": percentile(target_lengths, 0.95),
            "target_length_max": max(target_lengths) if target_lengths else 0,
            "unique_token_count": len(token_counts),
            "top_100_tokens": token_counts.most_common(100),
            "target_field": args.target_field,
            "seed": args.seed,
        }
    )
    save_json(stats, args.out_dir / "dataset_stats.json")
    logger.info("Prepared %s valid samples in %s", len(samples), args.out_dir)


if __name__ == "__main__":
    main()
