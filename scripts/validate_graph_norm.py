from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.graph_matching_eval import (  # noqa: E402
    run_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.target_normalization import (  # noqa: E402
    SSML_GRAPH_NORM_FIELD,
    SSML_GRAPH_NORM_SOURCE_FIELD,
    normalize_ssml_graph_with_stats,
)
from chemtexteller.utils import ensure_dir, read_jsonl, save_json, setup_logging  # noqa: E402


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that ssml_graph_norm preserves the graph semantics of ssml_normed "
            "with GraphMatchingTool."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/processed/edu_chemc_graph_norm"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--source_key", type=str, default=SSML_GRAPH_NORM_SOURCE_FIELD)
    parser.add_argument("--normalized_key", type=str, default=SSML_GRAPH_NORM_FIELD)
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=Path("external/GraphMatchingTool"))
    parser.add_argument("--graph_num_workers", type=int, default=8)
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/reports/graph_norm_validation"))
    parser.add_argument("--keep_temp", action="store_true")
    parser.add_argument("--min_graph_em", type=float, default=1.0)
    parser.add_argument("--min_structure_em", type=float, default=1.0)
    parser.add_argument("--angle_step", type=float, default=1.0)
    parser.add_argument("--angle_decimals", type=int, default=1)
    parser.add_argument("--length_decimals", type=int, default=1)
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


def image_name_for_graph(row: dict[str, Any], idx: int) -> str:
    value = row.get("image_name") or row.get("file_name") or f"sample_{idx:06d}"
    return Path(str(value)).stem


def split_rows(args: argparse.Namespace, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(args.dataset_dir / split / "metadata.jsonl")
    if args.max_samples_per_split is not None:
        rows = rows[: args.max_samples_per_split]

    graph_rows: list[dict[str, Any]] = []
    changed_rows = 0
    bond_specs_seen = 0
    bond_specs_changed = 0
    fallback_computed_rows = 0

    for idx, row in enumerate(rows):
        source = lookup_value(row, args.source_key)
        try:
            normalized = lookup_value(row, args.normalized_key)
        except KeyError:
            normalized, stats = normalize_ssml_graph_with_stats(
                source,
                angle_step=args.angle_step,
                angle_decimals=args.angle_decimals,
                length_decimals=args.length_decimals,
            )
            fallback_computed_rows += 1
        else:
            computed, stats = normalize_ssml_graph_with_stats(
                source,
                angle_step=args.angle_step,
                angle_decimals=args.angle_decimals,
                length_decimals=args.length_decimals,
            )
            if normalized != computed:
                raise ValueError(
                    f"{split} row {idx} has {args.normalized_key!r} but it does not "
                    "match the current ssml_graph_norm normalizer."
                )

        changed_rows += int(stats.changed)
        bond_specs_seen += stats.bond_specs_seen
        bond_specs_changed += stats.bond_specs_changed
        graph_rows.append(
            {
                "image_name": image_name_for_graph(row, idx),
                "prediction": normalized,
                "graph_label": source,
            }
        )

    stats_dict = {
        "split": split,
        "samples": len(graph_rows),
        "changed_rows": changed_rows,
        "changed_rows_pct": (100.0 * changed_rows / len(graph_rows)) if graph_rows else 0.0,
        "bond_specs_seen": bond_specs_seen,
        "bond_specs_changed": bond_specs_changed,
        "fallback_computed_rows": fallback_computed_rows,
    }
    return graph_rows, stats_dict


def validate_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    graph_rows, stats = split_rows(args, split)
    out_dir = ensure_dir(args.out_dir)
    rec_path = out_dir / f"{split}.ssml_graph_norm.rec.txt"
    lab_path = out_dir / f"{split}.ssml_normed.lab.txt"
    result_path = out_dir / f"{split}.graph_result.txt"

    write_graph_matching_files(graph_rows, rec_path, lab_path)
    result = run_graph_matching_tool(
        args.graph_matching_tool_dir,
        rec_path,
        lab_path,
        result_path,
        args.graph_num_workers,
    )
    if not args.keep_temp:
        rec_path.unlink(missing_ok=True)
        lab_path.unlink(missing_ok=True)

    metrics = result.metrics
    stats.update(metrics)
    stats["graph_result_path"] = str(result_path)
    return stats


def main() -> None:
    args = parse_args()
    summaries = [validate_split(args, split) for split in args.splits]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "source_key": args.source_key,
        "normalized_key": args.normalized_key,
        "splits": summaries,
        "min_graph_em": args.min_graph_em,
        "min_structure_em": args.min_structure_em,
    }
    save_json(summary, args.out_dir / "summary.json")

    failed = []
    for item in summaries:
        graph_em = float(item.get("graph_em", 0.0))
        structure_em = float(item.get("graph_structure_em", 0.0))
        logger.info(
            "%s | samples=%s changed_rows=%s graph_em=%.6f structure_em=%.6f",
            item["split"],
            item["samples"],
            item["changed_rows"],
            graph_em,
            structure_em,
        )
        if graph_em < args.min_graph_em or structure_em < args.min_structure_em:
            failed.append(item["split"])

    if failed:
        raise SystemExit(
            "Graph-preserving validation failed for splits: "
            f"{', '.join(failed)}. See {args.out_dir / 'summary.json'}"
        )
    logger.info("Graph-preserving validation passed. Wrote %s", args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
