from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.tokenizer_utils import whitespace_tokenize
from chemtexteller.utils import ensure_dir, save_json, setup_logging, write_jsonl


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a tiny prepared dataset from one source split. This is intended for "
            "local architecture smoke tests without touching official validation/test."
        )
    )
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/processed/edu_chemc_graph_norm"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed/edu_chemc_graph_norm_tiny500"))
    parser.add_argument("--source_split", type=str, default="train")
    parser.add_argument("--total_samples", type=int, default=500)
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=50,
        help=(
            "Hold out this many rows from the selected source rows as the tiny validation split. "
            "The total source rows used is still --total_samples."
        ),
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=["head", "random"],
        default="head",
        help="head reads the first N source rows quickly; random uses reservoir sampling over the full split.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=["reference", "copy", "symlink"],
        default="reference",
        help=(
            "reference writes absolute image paths back to the source prepared dataset; "
            "copy/symlink materialize images under the tiny dataset."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reset_out_dir(out_dir: Path, source_dir: Path, overwrite: bool) -> None:
    resolved_out = out_dir.resolve()
    resolved_source = source_dir.resolve()
    if (
        resolved_out == resolved_source
        or resolved_out in resolved_source.parents
        or resolved_source in resolved_out.parents
    ):
        raise SystemExit(f"Refusing to write tiny dataset inside the source split: {out_dir}")
    if out_dir.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {out_dir}. Pass --overwrite.")
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return float(ordered[low])
    frac = pos - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def resolve_source_image(row: dict[str, Any], source_split_dir: Path) -> Path:
    file_name = row.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("metadata row is missing a non-empty file_name")
    image_path = Path(file_name)
    if not image_path.is_absolute():
        image_path = source_split_dir / image_path
    return image_path


def image_leaf(row: dict[str, Any], source_image: Path, fallback_index: int) -> str:
    image_name = row.get("image_name")
    if isinstance(image_name, str) and image_name.strip():
        return Path(image_name).name
    leaf = source_image.name
    if leaf:
        return leaf
    return f"sample_{fallback_index:06d}.jpg"


def unique_leaf(name: str, used: set[str], idx: int) -> str:
    if name not in used:
        used.add(name)
        return name
    path = Path(name)
    stem = path.stem or "sample"
    suffix = path.suffix or ".jpg"
    deduped = f"{stem}_{idx:06d}{suffix}"
    used.add(deduped)
    return deduped


def materialize_image(
    source_image: Path,
    split_dir: Path,
    leaf: str,
    copy_mode: str,
) -> str:
    if copy_mode == "reference":
        return str(source_image.resolve())
    dst = split_dir / leaf
    ensure_dir(dst.parent)
    if copy_mode == "copy":
        shutil.copy2(source_image, dst)
    elif copy_mode == "symlink":
        try:
            dst.symlink_to(source_image.resolve())
        except OSError:
            logger.warning("Symlink failed for %s; falling back to copy.", source_image)
            shutil.copy2(source_image, dst)
    else:
        raise ValueError(f"Unsupported copy_mode: {copy_mode}")
    return leaf


def target_text(row: dict[str, Any]) -> str:
    target = row.get("target")
    if isinstance(target, str) and target.strip():
        return target.strip()
    targets = row.get("targets")
    target_field = row.get("target_field")
    if isinstance(targets, dict) and isinstance(target_field, str):
        value = targets.get(target_field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def write_split(
    *,
    name: str,
    rows: list[tuple[int, dict[str, Any]]],
    source_split_dir: Path,
    out_dir: Path,
    copy_mode: str,
) -> dict[str, Any]:
    split_dir = ensure_dir(out_dir / name)
    used_names: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    missing_images: list[dict[str, Any]] = []

    for local_idx, (source_idx, row) in enumerate(rows):
        source_image = resolve_source_image(row, source_split_dir)
        if not source_image.is_file():
            missing_images.append(
                {
                    "source_index": source_idx,
                    "file_name": row.get("file_name"),
                    "resolved_path": str(source_image),
                }
            )
            continue
        leaf = unique_leaf(image_leaf(row, source_image, local_idx), used_names, local_idx)
        file_name = materialize_image(source_image, split_dir, leaf, copy_mode)
        out_row = dict(row)
        out_row["file_name"] = file_name
        out_row["image_name"] = leaf
        out_row["source_index"] = source_idx
        output_rows.append(out_row)
        text = target_text(out_row)
        if text:
            lengths.append(len(whitespace_tokenize(text)))

    write_jsonl(output_rows, split_dir / "metadata.jsonl")
    return {
        "split": name,
        "source_rows": len(rows),
        "written_rows": len(output_rows),
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:5],
        "target_length_mean": mean(lengths) if lengths else 0.0,
        "target_length_p50": median(lengths) if lengths else 0.0,
        "target_length_p95": percentile(lengths, 0.95),
        "target_length_max": max(lengths) if lengths else 0,
    }


def sample_metadata_rows(
    metadata_path: Path,
    total_samples: int,
    seed: int,
    selection: str,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    rng = random.Random(seed)
    selected: list[tuple[int, dict[str, Any]]] = []
    seen = 0
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {metadata_path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {metadata_path}:{line_no}")
            if selection == "head":
                selected.append((seen, row))
                seen += 1
                if len(selected) >= total_samples:
                    break
                continue
            if len(selected) < total_samples:
                selected.append((seen, row))
            else:
                replacement_idx = rng.randint(0, seen)
                if replacement_idx < total_samples:
                    selected[replacement_idx] = (seen, row)
            seen += 1
    if total_samples > seen:
        raise SystemExit(
            f"Requested {total_samples} samples from {metadata_path}, but only {seen} rows exist."
        )
    if selection == "random":
        rng.shuffle(selected)
    return selected, seen


def main() -> None:
    args = parse_args()
    if args.total_samples <= 1:
        raise SystemExit("--total_samples must be greater than 1.")
    if args.validation_samples <= 0:
        raise SystemExit("--validation_samples must be positive so the tiny train run can evaluate.")
    if args.validation_samples >= args.total_samples:
        raise SystemExit("--validation_samples must be smaller than --total_samples.")

    source_split_dir = args.dataset_dir / args.source_split
    metadata_path = source_split_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing source metadata: {metadata_path}")

    reset_out_dir(args.out_dir, source_split_dir, overwrite=args.overwrite)
    selected, source_rows_scanned = sample_metadata_rows(
        metadata_path,
        total_samples=args.total_samples,
        seed=args.sample_seed,
        selection=args.selection,
    )
    train_count = args.total_samples - args.validation_samples
    train_rows = selected[:train_count]
    validation_rows = selected[train_count:]

    split_stats = [
        write_split(
            name="train",
            rows=train_rows,
            source_split_dir=source_split_dir,
            out_dir=args.out_dir,
            copy_mode=args.copy_mode,
        ),
        write_split(
            name="validation",
            rows=validation_rows,
            source_split_dir=source_split_dir,
            out_dir=args.out_dir,
            copy_mode=args.copy_mode,
        ),
    ]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "source_split": args.source_split,
        "source_rows_scanned": source_rows_scanned,
        "out_dir": str(args.out_dir),
        "total_source_rows_used": args.total_samples,
        "sample_seed": args.sample_seed,
        "selection": args.selection,
        "copy_mode": args.copy_mode,
        "splits": split_stats,
    }
    save_json(summary, args.out_dir / "tiny_dataset_summary.json")
    logger.info(
        "Wrote tiny dataset to %s using %s source %s rows: %s train / %s validation.",
        args.out_dir,
        args.total_samples,
        args.source_split,
        split_stats[0]["written_rows"],
        split_stats[1]["written_rows"],
    )


if __name__ == "__main__":
    main()
