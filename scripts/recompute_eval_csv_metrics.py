from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.graph_matching_eval import (  # noqa: E402
    run_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.metrics import per_sample_metrics, sequence_metrics  # noqa: E402
from chemtexteller.tokenizer_utils import normalize_whitespace, whitespace_tokenize  # noqa: E402
from chemtexteller.utils import ensure_dir, save_json  # noqa: E402


CJK_RE = re.compile(r"[\u3400-\u9fff]")
ANCHOR_RE = re.compile(r"\?\[[^\]]+\]")
ANGLE_BOND_RE = re.compile(r"(?<![A-Za-z])[-=~<>|_:]*\[:\s*-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)?\]")
TOKEN_RE = re.compile(
    r"\\[A-Za-z]+|\?\[[^\]]+\]|[-=~<>|_:]*\[:\s*-?\d+(?:\.\d+)?(?:,\s*-?\d+(?:\.\d+)?)?\]|[A-Za-z]+|\d+|[^\s]"
)
ATOMISH_RE = re.compile(
    r"^(?:C|H|O|N|S|P|F|I|B|r|l|a|K|M|n|d|u|Z|A|g|e|L|i)$"
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def clean_key(value: str) -> str:
    return Path(str(value)).stem


def token_list(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text or ""))


def count_features(text: Any) -> dict[str, int]:
    text = str(text or "")
    toks = token_list(text)
    return {
        "tokens": len(toks),
        "chars": len(text),
        "chemfig": text.count(r"\chemfig"),
        "arrow": text.count(r"\rightarrow")
        + text.count(r"\leftarrow")
        + text.count(r"\leftrightarrow"),
        "overset": text.count(r"\overset"),
        "underset": text.count(r"\underset"),
        "branch": len(re.findall(r"\bbranch\b", text)),
        "ring_anchor": len(ANCHOR_RE.findall(text)),
        "angle_bond": len(ANGLE_BOND_RE.findall(text)),
        "double_bond": text.count("=[:") + text.count("{=}"),
        "triple_bond": text.count("~[:") + text.count(r"\equiv"),
        "circle": text.count(r"\circle"),
        "chemabove": text.count(r"\Chemabove") + text.count(r"\chemabove"),
        "cjk": len(CJK_RE.findall(text)),
        "digit": len(re.findall(r"\d+", text)),
        "plus": toks.count("+"),
        "brace_delta": text.count("{") - text.count("}"),
        "paren_delta": text.count("(") - text.count(")"),
        "unk": text.count(r"\unk"),
    }


def family_counts(text: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for tok in token_list(text):
        if tok.startswith("\\"):
            if tok in {
                r"\chemfig",
                r"\rightarrow",
                r"\overset",
                r"\underset",
                r"\Chemabove",
                r"\circle",
                r"\triangle",
            }:
                counts[f"macro:{tok}"] += 1
            else:
                counts["macro:other"] += 1
        elif ANGLE_BOND_RE.fullmatch(tok):
            counts["bond_angle"] += 1
            if tok.startswith("="):
                counts["bond_type:="] += 1
            elif tok.startswith("~"):
                counts["bond_type:~"] += 1
            elif tok.startswith("-"):
                counts["bond_type:-"] += 1
            else:
                counts["bond_type:other"] += 1
        elif ANCHOR_RE.fullmatch(tok):
            counts["ring_anchor"] += 1
        elif tok == "branch":
            counts["branch"] += 1
        elif tok in {"{", "}", "(", ")", "[", "]"}:
            counts[f"syntax:{tok}"] += 1
        elif CJK_RE.search(tok):
            counts["cjk"] += 1
        elif tok.isdigit():
            counts["number"] += 1
        elif ATOMISH_RE.match(tok):
            counts[f"atom:{tok}"] += 1
        elif tok in {"+", "/", ",", ".", "^", "_"}:
            counts[f"symbol:{tok}"] += 1
        else:
            counts["other_token"] += 1
    return counts


def counter_l1(left: Counter[str], right: Counter[str], prefix: str | None = None) -> int:
    keys = set(left) | set(right)
    if prefix is not None:
        keys = {key for key in keys if key.startswith(prefix)}
    return sum(abs(left.get(key, 0) - right.get(key, 0)) for key in keys)


def preview(text: Any, limit: int = 260) -> str:
    value = normalize_whitespace(str(text or ""))
    return value if len(value) <= limit else value[: limit - 3] + "..."


def rate(correct: int | float, total: int) -> float:
    return 0.0 if total <= 0 else float(correct) / total


def parse_graph_result(path: Path) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) != 5:
                continue
            key, structure_flag, em_flag, label, prediction = parts
            parsed[key] = {
                "graph_structure_ok": int(structure_flag) == 0,
                "graph_em_ok": int(em_flag) == 0,
                "tool_label": label,
                "tool_prediction": prediction,
            }
    return parsed


def choose_bucket(
    *,
    graph_em_ok: bool,
    graph_structure_ok: bool,
    deltas: dict[str, int],
    label_counts: Counter[str],
    prediction_counts: Counter[str],
    length_delta: int,
) -> str:
    if graph_em_ok:
        return "graph_em_ok"
    if graph_structure_ok:
        return "structure_ok_em_fail_text_or_serialization"
    if abs(deltas["chemfig"]) > 0 or abs(deltas["arrow"]) > 0:
        return "structure_fail_component_or_reaction_count"
    if length_delta <= -20:
        return "structure_fail_truncated_or_too_short"
    if length_delta >= 20:
        return "structure_fail_too_long_or_repeated"
    if deltas["brace_delta"] != 0 or deltas["paren_delta"] != 0:
        return "structure_fail_syntax_balance"
    if counter_l1(label_counts, prediction_counts, "atom:") >= 4:
        return "structure_fail_atom_label_or_formula"
    if abs(deltas["branch"]) >= 4 or abs(deltas["ring_anchor"]) >= 2:
        return "structure_fail_topology_branch_ring"
    if counter_l1(label_counts, prediction_counts, "bond_type:") >= 2 or abs(
        deltas["angle_bond"]
    ) >= 4:
        return "structure_fail_bond_type_geometry"
    return "structure_fail_other_graph_mismatch"


def length_bin(length: int) -> str:
    if length <= 32:
        return "<=32"
    if length <= 64:
        return "33-64"
    if length <= 128:
        return "65-128"
    if length <= 256:
        return "129-256"
    return ">256"


def aggregate_binary(rows: Iterable[dict[str, Any]], key: str) -> tuple[int, int, float]:
    items = list(rows)
    correct = sum(1 for item in items if bool(item.get(key)))
    return correct, len(items), rate(correct, len(items))


def compare_metric_values(csv_rows: list[dict[str, str]], case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    numeric_pairs = [
        ("exact_match", "exact_match"),
        ("normalized_exact_match", "normalized_exact_match"),
        ("token_edit_distance", "token_edit_distance"),
        ("normalized_token_edit_distance", "normalized_token_edit_distance"),
        ("char_edit_distance", "char_edit_distance"),
    ]
    for csv_key, recomputed_key in numeric_pairs:
        if not csv_rows or csv_key not in csv_rows[0]:
            continue
        diffs: list[float] = []
        mismatches = 0
        for raw, computed in zip(csv_rows, case_rows, strict=True):
            raw_value = str(raw.get(csv_key, "")).strip()
            if raw_value.lower() in {"true", "false"}:
                csv_value = 1.0 if raw_value.lower() == "true" else 0.0
            else:
                try:
                    csv_value = float(raw_value)
                except ValueError:
                    continue
            computed_value = computed[recomputed_key]
            if isinstance(computed_value, bool):
                recomputed_value = 1.0 if computed_value else 0.0
            else:
                recomputed_value = float(computed_value)
            diff = abs(csv_value - recomputed_value)
            diffs.append(diff)
            mismatches += int(diff > 1e-9)
        comparisons[csv_key] = {
            "checked": len(diffs),
            "mismatches": mismatches,
            "max_abs_diff": max(diffs) if diffs else 0.0,
        }
    return comparisons


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    graph_em_correct, graph_total, graph_em = aggregate_binary(rows, "graph_em_ok")
    structure_correct, structure_total, structure_em = aggregate_binary(rows, "graph_structure_ok")
    exact_correct, _, exact_match = aggregate_binary(rows, "exact_match")
    return {
        "samples": len(rows),
        "exact_match": exact_match,
        "exact_match_correct": exact_correct,
        "graph_em": graph_em,
        "graph_em_correct": graph_em_correct,
        "graph_em_total": graph_total,
        "graph_structure_em": structure_em,
        "graph_structure_em_correct": structure_correct,
        "graph_structure_em_total": structure_total,
        "mean_token_edit_distance": sum(float(row["token_edit_distance"]) for row in rows)
        / len(rows),
        "mean_normalized_token_edit_distance": sum(
            float(row["normalized_token_edit_distance"]) for row in rows
        )
        / len(rows),
        "average_target_length": sum(int(row["target_tokens"]) for row in rows) / len(rows),
        "average_prediction_length": sum(int(row["prediction_tokens"]) for row in rows)
        / len(rows),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv_rows(args.csv)
    if not rows:
        raise ValueError(f"No rows in {args.csv}")
    missing = [
        col
        for col in [args.image_name_col, args.reference_col, args.prediction_col, args.graph_label_col]
        if col not in rows[0]
    ]
    if missing:
        raise KeyError(f"Missing columns in {args.csv}: {missing}")

    out_dir = ensure_dir(args.out_dir)
    predictions = [normalize_whitespace(row[args.prediction_col]) for row in rows]
    references = [normalize_whitespace(row[args.reference_col]) for row in rows]
    sequence = sequence_metrics(predictions, references)

    graph_rows = [
        {
            "image_name": clean_key(row[args.image_name_col]),
            "prediction": row[args.prediction_col],
            "graph_label": row[args.graph_label_col],
        }
        for row in rows
    ]
    rec_path = out_dir / f"{args.csv.stem}.recomputed.rec.txt"
    lab_path = out_dir / f"{args.csv.stem}.recomputed.lab.txt"
    graph_result_path = out_dir / f"{args.csv.stem}.recomputed.graph_result.txt"
    write_graph_matching_files(graph_rows, rec_path, lab_path)
    graph_result = run_graph_matching_tool(
        tool_dir=args.graph_matching_tool_dir,
        rec_path=rec_path,
        lab_path=lab_path,
        output_path=graph_result_path,
        num_workers=args.graph_num_workers,
    )
    graph_by_key = parse_graph_result(graph_result_path)

    downloaded_graph_comparison: dict[str, Any] = {}
    if args.existing_graph_result is not None and args.existing_graph_result.exists():
        old_graph = parse_graph_result(args.existing_graph_result)
        common_keys = set(old_graph) & set(graph_by_key)
        flag_mismatches = [
            key
            for key in common_keys
            if old_graph[key]["graph_em_ok"] != graph_by_key[key]["graph_em_ok"]
            or old_graph[key]["graph_structure_ok"] != graph_by_key[key]["graph_structure_ok"]
        ]
        downloaded_graph_comparison = {
            "existing_graph_result": str(args.existing_graph_result),
            "existing_rows": len(old_graph),
            "recomputed_rows": len(graph_by_key),
            "common_rows": len(common_keys),
            "flag_mismatches": len(flag_mismatches),
            "flag_mismatch_examples": sorted(flag_mismatches)[:20],
        }

    case_rows: list[dict[str, Any]] = []
    missing_tokens: Counter[str] = Counter()
    extra_tokens: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()

    for row, prediction, reference in zip(rows, predictions, references, strict=True):
        image_name = row[args.image_name_col]
        key = clean_key(image_name)
        graph = graph_by_key.get(key, {})
        graph_em_ok = bool(graph.get("graph_em_ok", False))
        graph_structure_ok = bool(graph.get("graph_structure_ok", False))
        sample_metrics = per_sample_metrics(prediction, reference)
        target_features = count_features(row[args.graph_label_col])
        pred_features = count_features(prediction)
        deltas = {
            name: pred_features[name] - target_features[name] for name in target_features
        }
        target_counts = family_counts(row[args.graph_label_col])
        pred_counts = family_counts(prediction)
        bucket = choose_bucket(
            graph_em_ok=graph_em_ok,
            graph_structure_ok=graph_structure_ok,
            deltas=deltas,
            label_counts=target_counts,
            prediction_counts=pred_counts,
            length_delta=deltas["tokens"],
        )
        bucket_counts[bucket] += 1

        if not graph_em_ok:
            label_tokens = Counter(token_list(row[args.graph_label_col]))
            pred_tokens = Counter(token_list(prediction))
            missing_tokens.update(label_tokens - pred_tokens)
            extra_tokens.update(pred_tokens - label_tokens)

        case = {
            "image_name": image_name,
            "bucket": bucket,
            "graph_em_ok": graph_em_ok,
            "graph_structure_ok": graph_structure_ok,
            "exact_match": bool(sample_metrics["exact_match"]),
            "normalized_exact_match": bool(sample_metrics["normalized_exact_match"]),
            "token_edit_distance": int(sample_metrics["token_edit_distance"]),
            "normalized_token_edit_distance": float(
                sample_metrics["normalized_token_edit_distance"]
            ),
            "char_edit_distance": int(sample_metrics["char_edit_distance"]),
            "target_tokens": len(whitespace_tokenize(reference)),
            "prediction_tokens": len(whitespace_tokenize(prediction)),
            "graph_label_tokens": target_features["tokens"],
            "graph_prediction_tokens": pred_features["tokens"],
            "length_delta": deltas["tokens"],
            "length_bin": length_bin(len(whitespace_tokenize(reference))),
            "target_chemfig": target_features["chemfig"],
            "target_arrow": target_features["arrow"],
            "target_branch": target_features["branch"],
            "target_ring_anchor": target_features["ring_anchor"],
            "target_cjk": target_features["cjk"],
            "target_digit": target_features["digit"],
            "delta_chemfig": deltas["chemfig"],
            "delta_arrow": deltas["arrow"],
            "delta_branch": deltas["branch"],
            "delta_ring_anchor": deltas["ring_anchor"],
            "delta_angle_bond": deltas["angle_bond"],
            "delta_cjk": deltas["cjk"],
            "delta_digit": deltas["digit"],
            "atom_l1_delta": counter_l1(target_counts, pred_counts, "atom:"),
            "bond_type_l1_delta": counter_l1(target_counts, pred_counts, "bond_type:"),
            "macro_l1_delta": counter_l1(target_counts, pred_counts, "macro:"),
            "target_preview": preview(row[args.graph_label_col]),
            "prediction_preview": preview(prediction),
        }
        case_rows.append(case)

    metric_comparison = compare_metric_values(rows, case_rows)

    case_rows.sort(
        key=lambda item: (
            bool(item["graph_em_ok"]),
            -int(item["token_edit_distance"]),
            str(item["image_name"]),
        )
    )
    cases_path = out_dir / f"{args.csv.stem}.error_cases.csv"
    with cases_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    by_length_bin = {
        key: summarize_group([row for row in case_rows if row["length_bin"] == key])
        for key in ["<=32", "33-64", "65-128", "129-256", ">256"]
    }
    by_reaction_complexity = {
        "single_or_no_chemfig": summarize_group(
            [row for row in case_rows if int(row["target_chemfig"]) <= 1]
        ),
        "multi_chemfig": summarize_group(
            [row for row in case_rows if int(row["target_chemfig"]) > 1]
        ),
        "has_arrow": summarize_group([row for row in case_rows if int(row["target_arrow"]) > 0]),
        "has_cjk": summarize_group([row for row in case_rows if int(row["target_cjk"]) > 0]),
        "branch_heavy_gt20": summarize_group(
            [row for row in case_rows if int(row["target_branch"]) > 20]
        ),
        "ring_anchor_heavy_gt6": summarize_group(
            [row for row in case_rows if int(row["target_ring_anchor"]) > 6]
        ),
    }

    downloaded_metrics_path = args.csv.with_suffix(".metrics.json")
    downloaded_metrics: dict[str, Any] = {}
    if downloaded_metrics_path.exists():
        downloaded_metrics = json.loads(downloaded_metrics_path.read_text(encoding="utf-8"))

    recomputed_graph = {
        **graph_result.metrics,
        "graph_em_all_sample": rate(
            int(graph_result.metrics["graph_em_correct"]),
            len(rows),
        ),
        "graph_structure_em_all_sample": rate(
            int(graph_result.metrics["graph_structure_em_correct"]),
            len(rows),
        ),
        "graph_result_txt": str(graph_result_path),
    }
    summary = {
        "csv": str(args.csv),
        "num_samples": len(rows),
        "sequence_metrics_recomputed": sequence,
        "graph_metrics_recomputed": recomputed_graph,
        "downloaded_metrics_json": downloaded_metrics,
        "csv_metric_value_check": metric_comparison,
        "downloaded_graph_result_check": downloaded_graph_comparison,
        "bucket_counts": dict(bucket_counts.most_common()),
        "bucket_rates": {
            key: count / len(rows) for key, count in bucket_counts.most_common()
        },
        "by_length_bin": by_length_bin,
        "by_reaction_complexity": by_reaction_complexity,
        "top_missing_tokens_on_graph_fail": missing_tokens.most_common(40),
        "top_extra_tokens_on_graph_fail": extra_tokens.most_common(40),
        "hardest_cases": case_rows[:30],
        "cases_csv": str(cases_path),
        "rec_path": str(rec_path),
        "lab_path": str(lab_path),
    }

    summary_path = out_dir / f"{args.csv.stem}.recomputed_metrics.json"
    save_json(summary, summary_path)

    report_path = out_dir / f"{args.csv.stem}.error_report.md"
    report_lines = [
        "# Recomputed Eval Analysis",
        "",
        f"- CSV: `{args.csv}`",
        f"- Samples: {len(rows)}",
        f"- Exact match: {sequence['exact_match']:.6f}",
        f"- Mean token ED: {sequence['mean_token_edit_distance']:.4f}",
        f"- Mean normalized token ED: {sequence['mean_normalized_token_edit_distance']:.6f}",
        f"- Graph EM: {recomputed_graph['graph_em']:.6f} "
        f"({recomputed_graph['graph_em_correct']}/{recomputed_graph['graph_em_total']})",
        f"- Structure EM: {recomputed_graph['graph_structure_em']:.6f} "
        f"({recomputed_graph['graph_structure_em_correct']}/{recomputed_graph['graph_structure_em_total']})",
        "",
        "## Error Buckets",
        "",
    ]
    for bucket, count in bucket_counts.most_common():
        report_lines.append(f"- `{bucket}`: {count} ({count / len(rows):.2%})")
    report_lines.extend(["", "## Length Bins", ""])
    for key, item in by_length_bin.items():
        if item["samples"]:
            report_lines.append(
                f"- `{key}`: n={item['samples']}, graph_EM={item['graph_em']:.4f}, "
                f"structure_EM={item['graph_structure_em']:.4f}, "
                f"mean_norm_ED={item['mean_normalized_token_edit_distance']:.4f}"
            )
    report_lines.extend(["", "## Top Missing Tokens On Graph Fail", ""])
    report_lines.extend([f"- `{tok}`: {count}" for tok, count in missing_tokens.most_common(20)])
    report_lines.extend(["", "## Top Extra Tokens On Graph Fail", ""])
    report_lines.extend([f"- `{tok}`: {count}" for tok, count in extra_tokens.most_common(20)])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "cases": str(cases_path),
                "report": str(report_path),
                "exact_match": sequence["exact_match"],
                "graph_em": recomputed_graph["graph_em"],
                "graph_structure_em": recomputed_graph["graph_structure_em"],
                "bucket_counts": dict(bucket_counts.most_common(8)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute sequence and GraphMatchingTool metrics from an eval CSV."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=Path("external/GraphMatchingTool"))
    parser.add_argument("--graph_num_workers", type=int, default=1)
    parser.add_argument("--existing_graph_result", type=Path, default=None)
    parser.add_argument("--image_name_col", type=str, default="image_name")
    parser.add_argument("--reference_col", type=str, default="ground_truth")
    parser.add_argument("--prediction_col", type=str, default="prediction")
    parser.add_argument("--graph_label_col", type=str, default="graph_label")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
