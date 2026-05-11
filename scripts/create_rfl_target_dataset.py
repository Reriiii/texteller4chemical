from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.rfl_adapter import convert_ssml_to_rfl  # noqa: E402
from chemtexteller.utils import ensure_dir, read_jsonl, save_json, setup_logging, write_jsonl  # noqa: E402


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a prepared EDU-CHEMC dataset variant with RFL targets."
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--source_key", type=str, default="ssml_sd")
    parser.add_argument("--target_field", type=str, default="ssml_rfl")
    parser.add_argument("--rfl_tool_dir", type=Path, default=Path("external/RFL-MSD/RFL"))
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--on_error",
        choices=["fallback", "skip", "raise"],
        default="fallback",
        help="How to handle RFL conversion failures. fallback keeps source_key as the target.",
    )
    return parser.parse_args()


def lookup_value(row: dict[str, Any], key: str) -> str:
    direct = row.get(key)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    targets = row.get("targets")
    if isinstance(targets, dict):
        nested_key = key.split(".", 1)[1] if key.startswith("targets.") else key
        nested = targets.get(nested_key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    raise KeyError(key)


def resolve_image_path(split_dir: Path, row: dict[str, Any]) -> str:
    file_name = row.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("metadata row is missing file_name")
    image_path = Path(file_name)
    if not image_path.is_absolute():
        image_path = split_dir / image_path
    return str(image_path.resolve())


def convert_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    in_split_dir = args.dataset_dir / split
    rows = read_jsonl(in_split_dir / "metadata.jsonl")
    if args.max_samples_per_split is not None:
        rows = rows[: args.max_samples_per_split]

    out_split_dir = ensure_dir(args.out_dir / split)
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fallback_count = 0
    skipped_count = 0

    for idx, row in enumerate(tqdm(rows, desc=f"RFL {split}")):
        image_name = row.get("image_name") or Path(str(row.get("file_name", f"{idx:06d}"))).name
        source = lookup_value(row, args.source_key)
        result = convert_ssml_to_rfl(source, args.rfl_tool_dir)

        if result.success:
            target = result.target
            status = "converted"
        else:
            failures.append(
                {
                    "split": split,
                    "row": idx,
                    "image_name": image_name,
                    "error": result.error,
                }
            )
            if args.on_error == "raise":
                raise RuntimeError(f"RFL conversion failed for {split} row {idx}: {result.error}")
            if args.on_error == "skip":
                skipped_count += 1
                continue
            target = source
            status = "fallback"
            fallback_count += 1

        targets = dict(row.get("targets") or {})
        targets[args.source_key] = source
        targets[args.target_field] = target
        out_row = dict(row)
        out_row["file_name"] = resolve_image_path(in_split_dir, row)
        out_row["image_name"] = image_name
        out_row["target"] = target
        out_row["target_field"] = args.target_field
        out_row["targets"] = targets
        out_row["target_status"] = status
        output_rows.append(out_row)

    write_jsonl(output_rows, out_split_dir / "metadata.jsonl")
    return {
        "split": split,
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "converted": len(output_rows) - fallback_count,
        "fallback": fallback_count,
        "skipped": skipped_count,
        "failures": failures[:50],
        "failure_count": len(failures),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == dataset_dir or dataset_dir in out_dir.parents:
        raise SystemExit(
            f"Refusing to write inside the source dataset directory: {args.out_dir}"
        )
    if args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory exists. Pass --overwrite to replace: {args.out_dir}")
        shutil.rmtree(args.out_dir)
    ensure_dir(args.out_dir)

    summaries = [convert_split(args, split) for split in args.splits]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "source_key": args.source_key,
        "target_field": args.target_field,
        "rfl_tool_dir": str(args.rfl_tool_dir),
        "on_error": args.on_error,
        "splits": summaries,
    }
    save_json(summary, args.out_dir / "target_conversion_summary.json")
    logger.info("Wrote RFL target dataset to %s", args.out_dir)


if __name__ == "__main__":
    main()
