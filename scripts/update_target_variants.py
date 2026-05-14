from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.target_normalization import (  # noqa: E402
    SSML_GRAPH_COMPACT_FIELD,
    SSML_GRAPH_NORM_FIELD,
    SSML_GRAPH_NORM_SOURCE_FIELD,
    SSML_GRAPH_SD_FIELD,
    normalize_target_for_field_with_stats,
)
from chemtexteller.tokenizer_utils import whitespace_tokenize  # noqa: E402
from chemtexteller.utils import read_jsonl, save_json, setup_logging, write_jsonl  # noqa: E402


logger = setup_logging()

DEFAULT_FIELDS = (
    SSML_GRAPH_NORM_FIELD,
    SSML_GRAPH_SD_FIELD,
    SSML_GRAPH_COMPACT_FIELD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add graph-preserving target variants to an existing prepared EDU-CHEMC dataset. "
            "Dry-run is the default; pass --write to modify metadata.jsonl."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/processed/edu_chemc_graph_norm"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument(
        "--source_key",
        type=str,
        default=SSML_GRAPH_NORM_SOURCE_FIELD,
        help="Source label used to derive graph-safe variants, usually ssml_normed.",
    )
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    parser.add_argument(
        "--set_target_field",
        type=str,
        default=None,
        help=(
            "Optional: also set top-level target/target_field to this computed field. "
            "Use only when creating a dedicated target-variant dataset."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing variant values if they differ from the current normalizer.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write metadata changes. Without this flag the script only reports what would change.",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional smoke-test limit. Cannot be combined with --write.",
    )
    parser.add_argument(
        "--backup_suffix",
        type=str,
        default=".bak",
        help="Backup suffix used next to metadata.jsonl when --write is set.",
    )
    parser.add_argument(
        "--out_report",
        type=Path,
        default=Path("outputs/reports/target_variants_update.json"),
    )
    return parser.parse_args()


def lookup_value(row: dict[str, Any], key: str) -> str:
    value: Any = row
    if key.startswith("targets."):
        key = key.split(".", 1)[1]
        value = row.get("targets", {})

    if "." not in key:
        direct = row.get(key)
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        targets = row.get("targets")
        if isinstance(targets, dict):
            nested = targets.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(key)
        value = value[part]
    if not isinstance(value, str) or not value.strip():
        raise KeyError(key)
    return value.strip()


def ensure_targets(row: dict[str, Any]) -> dict[str, str]:
    raw_targets = row.get("targets")
    if not isinstance(raw_targets, dict):
        raw_targets = {}
        row["targets"] = raw_targets
    targets: dict[str, str] = {}
    for key, value in raw_targets.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            targets[key] = value
    row["targets"] = targets
    return targets


def split_summary(args: argparse.Namespace, split: str) -> dict[str, Any]:
    metadata_path = args.dataset_dir / split / "metadata.jsonl"
    rows = read_jsonl(metadata_path)
    total_rows = len(rows)
    if args.max_samples_per_split is not None:
        rows = rows[: args.max_samples_per_split]
    changed_rows = 0
    refreshed_values = 0
    added_values = 0
    missing_source_rows = 0
    field_stats: dict[str, dict[str, Any]] = {
        field: {
            "added": 0,
            "refreshed": 0,
            "already_current": 0,
            "mismatched_existing": 0,
            "changed_rows": 0,
            "bond_specs_seen": 0,
            "bond_specs_changed": 0,
            "lengths_seen": 0,
            "lengths_dropped": 0,
            "lengths_preserved_nonzero": 0,
            "token_lengths": [],
        }
        for field in args.fields
    }

    for row_idx, row in enumerate(rows):
        try:
            source = lookup_value(row, args.source_key)
        except KeyError:
            missing_source_rows += 1
            continue

        targets = ensure_targets(row)
        row_changed = False
        for field in args.fields:
            field_row_changed = False
            normalized, stats = normalize_target_for_field_with_stats(source, field)
            field_summary = field_stats[field]
            field_summary["bond_specs_seen"] += stats.bond_specs_seen
            field_summary["bond_specs_changed"] += stats.bond_specs_changed
            field_summary["lengths_seen"] += stats.lengths_seen
            field_summary["lengths_dropped"] += stats.lengths_dropped
            field_summary["lengths_preserved_nonzero"] += stats.lengths_preserved_nonzero
            field_summary["token_lengths"].append(len(whitespace_tokenize(normalized)))

            existing = targets.get(field)
            if existing is None:
                targets[field] = normalized
                added_values += 1
                field_summary["added"] += 1
                row_changed = True
                field_row_changed = True
            elif existing == normalized:
                field_summary["already_current"] += 1
            else:
                field_summary["mismatched_existing"] += 1
                if args.refresh:
                    targets[field] = normalized
                    refreshed_values += 1
                    field_summary["refreshed"] += 1
                    row_changed = True
                    field_row_changed = True
                else:
                    raise ValueError(
                        f"{metadata_path}:{row_idx + 1} has stale targets.{field}. "
                        "Re-run with --refresh to update it."
                    )

            if args.set_target_field == field:
                if row.get("target") != normalized or row.get("target_field") != field:
                    row["target"] = normalized
                    row["target_field"] = field
                    row_changed = True
                    field_row_changed = True

            if field_row_changed:
                field_summary["changed_rows"] += 1

        if row_changed:
            changed_rows += 1

    for field, stats in field_stats.items():
        lengths = stats.pop("token_lengths")
        lengths_sorted = sorted(lengths)
        stats["min_tokens"] = lengths_sorted[0] if lengths_sorted else 0
        stats["max_tokens"] = lengths_sorted[-1] if lengths_sorted else 0
        stats["mean_tokens"] = sum(lengths) / len(lengths) if lengths else 0.0
        if lengths_sorted:
            stats["p50_tokens"] = lengths_sorted[len(lengths_sorted) // 2]
            stats["p95_tokens"] = lengths_sorted[int((len(lengths_sorted) - 1) * 0.95)]
        else:
            stats["p50_tokens"] = 0
            stats["p95_tokens"] = 0

    if args.write and changed_rows:
        backup_path = metadata_path.with_name(metadata_path.name + args.backup_suffix)
        if not backup_path.exists():
            shutil.copy2(metadata_path, backup_path)
        write_jsonl(rows, metadata_path)

    return {
        "split": split,
        "metadata_path": str(metadata_path),
        "samples": len(rows),
        "total_samples": total_rows,
        "changed_rows": changed_rows,
        "added_values": added_values,
        "refreshed_values": refreshed_values,
        "missing_source_rows": missing_source_rows,
        "wrote": bool(args.write and changed_rows),
        "fields": field_stats,
    }


def main() -> None:
    args = parse_args()
    if args.set_target_field is not None and args.set_target_field not in args.fields:
        raise SystemExit("--set_target_field must be one of --fields.")
    if args.write and args.max_samples_per_split is not None:
        raise SystemExit("--max_samples_per_split is for dry-run smoke tests and cannot be used with --write.")

    summaries = [split_summary(args, split) for split in args.splits]
    report = {
        "dataset_dir": str(args.dataset_dir),
        "source_key": args.source_key,
        "fields": args.fields,
        "set_target_field": args.set_target_field,
        "refresh": args.refresh,
        "write": args.write,
        "max_samples_per_split": args.max_samples_per_split,
        "splits": summaries,
    }
    save_json(report, args.out_report)
    for item in summaries:
        logger.info(
            "%s | samples=%s changed_rows=%s added=%s refreshed=%s wrote=%s",
            item["split"],
            item["samples"],
            item["changed_rows"],
            item["added_values"],
            item["refreshed_values"],
            item["wrote"],
        )
    logger.info("Wrote report to %s", args.out_report)
    if not args.write:
        logger.info("Dry-run only. Re-run with --write to update metadata.")


if __name__ == "__main__":
    main()
