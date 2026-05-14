from __future__ import annotations

import argparse
import sys
from itertools import zip_longest
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
    BOND_SPEC_RE,
    SSML_GRAPH_NORM_FIELD,
    SSML_GRAPH_NORM_SOURCE_FIELD,
    normalize_ssml_graph_with_stats,
    normalize_target_for_field_with_stats,
)
from chemtexteller.utils import ensure_dir, read_jsonl, save_json, setup_logging, write_jsonl  # noqa: E402


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
    parser.add_argument(
        "--max_failure_examples",
        type=int,
        default=20,
        help="Maximum failed graph-normalization cases to log and save per split.",
    )
    parser.add_argument(
        "--angle_step",
        type=float,
        default=1.0,
        help="Only used when --normalized_key is ssml_graph_norm.",
    )
    parser.add_argument(
        "--angle_decimals",
        type=int,
        default=1,
        help="Only used when --normalized_key is ssml_graph_norm.",
    )
    parser.add_argument(
        "--length_decimals",
        type=int,
        default=1,
        help="Only used when --normalized_key is ssml_graph_norm.",
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


def image_name_for_graph(row: dict[str, Any], idx: int) -> str:
    value = row.get("image_name") or row.get("file_name") or f"sample_{idx:06d}"
    return Path(str(value)).stem


def safe_key_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace("\\", "_")


def normalize_for_validation(
    source: str,
    args: argparse.Namespace,
) -> tuple[str, Any]:
    if args.normalized_key == SSML_GRAPH_NORM_FIELD:
        return normalize_ssml_graph_with_stats(
            source,
            angle_step=args.angle_step,
            angle_decimals=args.angle_decimals,
            length_decimals=args.length_decimals,
        )
    return normalize_target_for_field_with_stats(source, args.normalized_key)


def bond_specs(text: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for match in BOND_SPEC_RE.finditer(text):
        specs.append(
            {
                "raw": match.group(0),
                "prefix": match.group("prefix"),
                "angle": float(match.group("angle")),
                "length": float(match.group("length")),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return specs


def first_token_difference(left: str, right: str) -> dict[str, Any] | None:
    left_tokens = left.split()
    right_tokens = right.split()
    for idx, (left_token, right_token) in enumerate(
        zip_longest(left_tokens, right_tokens, fillvalue=None)
    ):
        if left_token != right_token:
            start = max(0, idx - 5)
            end = idx + 6
            return {
                "token_index": idx,
                "source_token": left_token,
                "normalized_token": right_token,
                "source_context": left_tokens[start:end],
                "normalized_context": right_tokens[start:end],
            }
    return None


def analyze_normalization_difference(source: str, normalized: str) -> dict[str, Any]:
    source_bonds = bond_specs(source)
    normalized_bonds = bond_specs(normalized)
    changed_bonds: list[dict[str, Any]] = []
    for idx, (source_bond, normalized_bond) in enumerate(
        zip_longest(source_bonds, normalized_bonds)
    ):
        if source_bond == normalized_bond:
            continue
        if source_bond is None or normalized_bond is None:
            changed_bonds.append(
                {
                    "bond_index": idx,
                    "source": source_bond,
                    "normalized": normalized_bond,
                }
            )
            continue
        if source_bond["raw"] != normalized_bond["raw"]:
            changed_bonds.append(
                {
                    "bond_index": idx,
                    "source": source_bond["raw"],
                    "normalized": normalized_bond["raw"],
                    "angle_delta": normalized_bond["angle"] - source_bond["angle"],
                    "length_delta": normalized_bond["length"] - source_bond["length"],
                    "prefix_changed": source_bond["prefix"] != normalized_bond["prefix"],
                }
            )
    return {
        "source_token_len": len(source.split()),
        "normalized_token_len": len(normalized.split()),
        "source_char_len": len(source),
        "normalized_char_len": len(normalized),
        "first_token_difference": first_token_difference(source, normalized),
        "source_bond_count": len(source_bonds),
        "normalized_bond_count": len(normalized_bonds),
        "changed_bond_count": len(changed_bonds),
        "changed_bonds_preview": changed_bonds[:25],
    }


def parse_graph_failures(result_path: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    with result_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) != 5:
                continue
            image_name, structure_flag, em_flag, label, prediction = parts
            try:
                structure_flag_int = int(structure_flag)
                em_flag_int = int(em_flag)
            except ValueError:
                continue
            if structure_flag_int == 0 and em_flag_int == 0:
                continue
            failures.append(
                {
                    "line": line_no,
                    "image_name": image_name,
                    "structure_flag": structure_flag_int,
                    "em_flag": em_flag_int,
                    "tool_label": label,
                    "tool_prediction": prediction,
                }
            )
    return failures


def log_failure_examples(split: str, failure_cases: list[dict[str, Any]], max_examples: int) -> None:
    for case in failure_cases[:max_examples]:
        analysis = case["analysis"]
        first_diff = analysis.get("first_token_difference") or {}
        logger.error(
            (
                "%s graph-normalization mismatch | row=%s image=%s "
                "structure_flag=%s em_flag=%s changed_bonds=%s first_diff=%s"
            ),
            split,
            case.get("row_index"),
            case.get("image_name"),
            case.get("structure_flag"),
            case.get("em_flag"),
            analysis.get("changed_bond_count"),
            first_diff,
        )
        for changed_bond in analysis.get("changed_bonds_preview", [])[:10]:
            logger.error("%s mismatch changed bond: %s", split, changed_bond)
        logger.error("%s source: %s", split, case.get("source"))
        logger.error("%s normalized: %s", split, case.get("normalized"))


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
            normalized, stats = normalize_for_validation(source, args)
            fallback_computed_rows += 1
        else:
            computed, stats = normalize_for_validation(source, args)
            if normalized != computed:
                raise ValueError(
                    f"{split} row {idx} has {args.normalized_key!r} but it does not "
                    f"match the current {args.normalized_key} normalizer."
                )

        changed_rows += int(stats.changed)
        bond_specs_seen += stats.bond_specs_seen
        bond_specs_changed += stats.bond_specs_changed
        graph_rows.append(
            {
                "row_index": idx,
                "image_name": image_name_for_graph(row, idx),
                "prediction": normalized,
                "graph_label": source,
                "source": source,
                "normalized": normalized,
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
    normalized_name = safe_key_name(args.normalized_key)
    source_name = safe_key_name(args.source_key)
    rec_path = out_dir / f"{split}.{normalized_name}.rec.txt"
    lab_path = out_dir / f"{split}.{source_name}.lab.txt"
    result_path = out_dir / f"{split}.{normalized_name}_vs_{source_name}.graph_result.txt"

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
    failures = parse_graph_failures(result_path)
    stats["graph_failure_count"] = len(failures)
    stats["graph_failure_examples_path"] = None
    if failures:
        rows_by_image = {str(row["image_name"]): row for row in graph_rows}
        failure_cases: list[dict[str, Any]] = []
        for failure in failures:
            graph_row = rows_by_image.get(str(failure["image_name"]))
            if graph_row is None:
                failure_cases.append({**failure, "error": "No matching graph row"})
                continue
            source = str(graph_row["source"])
            normalized = str(graph_row["normalized"])
            failure_cases.append(
                {
                    **failure,
                    "row_index": graph_row["row_index"],
                    "source": source,
                    "normalized": normalized,
                    "analysis": analyze_normalization_difference(source, normalized),
                }
            )
        examples_path = out_dir / f"{split}.{safe_key_name(args.normalized_key)}_failures.jsonl"
        write_jsonl(failure_cases, examples_path)
        stats["graph_failure_examples_path"] = str(examples_path)
        stats["graph_failure_examples"] = failure_cases[: args.max_failure_examples]
        log_failure_examples(split, failure_cases, args.max_failure_examples)
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
