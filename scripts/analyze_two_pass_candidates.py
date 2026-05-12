from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.graph_matching_eval import (  # noqa: E402
    run_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.metrics import per_sample_metrics, sequence_metrics  # noqa: E402
from chemtexteller.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze two-pass component decoding candidates from evaluate.py CSV output."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Evaluation CSV path.")
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("outputs/two_pass_candidate_analysis"),
        help="Prefix for JSON and optional GraphMatchingTool files.",
    )
    parser.add_argument(
        "--graph_matching_tool_dir",
        type=Path,
        default=None,
        help="Optional GraphMatchingTool directory for graph EM candidate analysis.",
    )
    parser.add_argument("--graph_num_workers", type=int, default=8)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def nonempty(row: dict[str, str], key: str) -> bool:
    return bool(str(row.get(key, "")).strip())


def is_truthy(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def int_value(row: dict[str, str], key: str) -> int:
    value = str(row.get(key, "")).strip()
    if not value:
        return 0
    return int(float(value))


def prediction_for(row: dict[str, str], field: str) -> str:
    if field == "first_pass_prediction":
        return row.get("first_pass_prediction") or row.get("raw_prediction", "")
    return row.get(field, "")


def sequence_block(
    rows: list[dict[str, str]],
    prediction_field: str,
) -> dict[str, Any]:
    usable = [row for row in rows if nonempty(row, prediction_field)]
    predictions = [prediction_for(row, prediction_field) for row in usable]
    references = [row.get("ground_truth", "") for row in usable]
    metrics = sequence_metrics(predictions, references)
    metrics["usable_samples"] = len(usable)
    return metrics


def token_oracle_block(rows: list[dict[str, str]]) -> dict[str, Any]:
    usable = [row for row in rows if nonempty(row, "two_pass_stitched_prediction")]
    better = 0
    worse = 0
    tied = 0
    first_distances: list[float] = []
    stitched_distances: list[float] = []
    oracle_predictions: list[str] = []
    references: list[str] = []
    for row in usable:
        reference = row.get("ground_truth", "")
        first = prediction_for(row, "first_pass_prediction")
        stitched = prediction_for(row, "two_pass_stitched_prediction")
        first_metrics = per_sample_metrics(first, reference)
        stitched_metrics = per_sample_metrics(stitched, reference)
        first_dist = float(first_metrics["normalized_token_edit_distance"])
        stitched_dist = float(stitched_metrics["normalized_token_edit_distance"])
        first_distances.append(first_dist)
        stitched_distances.append(stitched_dist)
        references.append(reference)
        if stitched_dist < first_dist:
            better += 1
            oracle_predictions.append(stitched)
        elif stitched_dist > first_dist:
            worse += 1
            oracle_predictions.append(first)
        else:
            tied += 1
            oracle_predictions.append(first)
    oracle_metrics = sequence_metrics(oracle_predictions, references) if usable else {}
    return {
        "usable_samples": len(usable),
        "stitched_better_token_count": better,
        "stitched_worse_token_count": worse,
        "stitched_tied_token_count": tied,
        "stitched_better_token_rate": better / len(usable) if usable else 0.0,
        "stitched_worse_token_rate": worse / len(usable) if usable else 0.0,
        "first_mean_norm_token_edit": (
            sum(first_distances) / len(first_distances) if first_distances else 0.0
        ),
        "stitched_mean_norm_token_edit": (
            sum(stitched_distances) / len(stitched_distances)
            if stitched_distances
            else 0.0
        ),
        "sequence_oracle": oracle_metrics,
    }


def graph_rows(rows: list[dict[str, str]], prediction_field: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        prediction = prediction_for(row, prediction_field)
        if not prediction.strip() or not row.get("graph_label", "").strip():
            continue
        output.append(
            {
                "image_name": row.get("image_name", ""),
                "prediction": prediction,
                "graph_label": row.get("graph_label", ""),
            }
        )
    return output


def graph_block(
    rows: list[dict[str, str]],
    prediction_field: str,
    *,
    tool_dir: Path,
    output_prefix: Path,
    subset_name: str,
    num_workers: int,
) -> dict[str, Any]:
    usable = graph_rows(rows, prediction_field)
    if not usable:
        return {"usable_samples": 0}
    safe_field = prediction_field.replace(".", "_")
    rec_path = output_prefix.with_name(
        f"{output_prefix.name}.{subset_name}.{safe_field}.rec.txt"
    )
    lab_path = output_prefix.with_name(
        f"{output_prefix.name}.{subset_name}.{safe_field}.lab.txt"
    )
    graph_output = output_prefix.with_name(
        f"{output_prefix.name}.{subset_name}.{safe_field}.graph_result.txt"
    )
    write_graph_matching_files(usable, rec_path, lab_path)
    result = run_graph_matching_tool(
        tool_dir=tool_dir,
        rec_path=rec_path,
        lab_path=lab_path,
        output_path=graph_output,
        num_workers=num_workers,
    )
    return {
        "usable_samples": len(usable),
        **result.metrics,
        "graph_output_txt": str(graph_output),
    }


def summarize_subset(
    name: str,
    rows: list[dict[str, str]],
    *,
    graph_matching_tool_dir: Path | None,
    output_prefix: Path,
    graph_num_workers: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": len(rows),
        "sequence": {
            "final_prediction": sequence_block(rows, "prediction"),
            "first_pass_prediction": sequence_block(rows, "first_pass_prediction"),
            "stitched_prediction": sequence_block(rows, "two_pass_stitched_prediction"),
        },
        "token_oracle": token_oracle_block(rows),
    }
    if graph_matching_tool_dir is not None:
        summary["graph"] = {
            "final_prediction": graph_block(
                rows,
                "prediction",
                tool_dir=graph_matching_tool_dir,
                output_prefix=output_prefix,
                subset_name=name,
                num_workers=graph_num_workers,
            ),
            "first_pass_prediction": graph_block(
                rows,
                "first_pass_prediction",
                tool_dir=graph_matching_tool_dir,
                output_prefix=output_prefix,
                subset_name=name,
                num_workers=graph_num_workers,
            ),
            "stitched_prediction": graph_block(
                rows,
                "two_pass_stitched_prediction",
                tool_dir=graph_matching_tool_dir,
                output_prefix=output_prefix,
                subset_name=name,
                num_workers=graph_num_workers,
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv)
    subsets: dict[str, Callable[[dict[str, str]], bool]] = {
        "all": lambda row: True,
        "triggered": lambda row: nonempty(row, "two_pass_reason"),
        "cropped": lambda row: int_value(row, "two_pass_crops") >= 2,
        "used": lambda row: is_truthy(row, "two_pass_used"),
        "has_stitched": lambda row: nonempty(row, "two_pass_stitched_prediction"),
    }
    output: dict[str, Any] = {
        "csv": str(args.csv),
        "subsets": {},
    }
    for name, predicate in subsets.items():
        subset_rows = [row for row in rows if predicate(row)]
        output["subsets"][name] = summarize_subset(
            name,
            subset_rows,
            graph_matching_tool_dir=args.graph_matching_tool_dir,
            output_prefix=args.output_prefix,
            graph_num_workers=args.graph_num_workers,
        )

    ensure_dir(args.output_prefix.parent)
    output_json = args.output_prefix.with_suffix(".json")
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Wrote analysis to {output_json}")


if __name__ == "__main__":
    main()
