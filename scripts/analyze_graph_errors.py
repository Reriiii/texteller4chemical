from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


CJK_RE = re.compile(r"[\u3400-\u9fff]")
ANCHOR_RE = re.compile(r"\?\[[^\]]+\]")
ANGLE_BOND_RE = re.compile(r"(?<![A-Za-z])[-=~<>|_:]*\[:\s*-?\d+\]")
TOKEN_RE = re.compile(
    r"\\[A-Za-z]+|\?\[[^\]]+\]|[-=~<>|_:]*\[:\s*-?\d+\]|[A-Za-z]+|\d+|[^\s]"
)
ATOMISH_RE = re.compile(
    r"^(?:C|H|O|N|S|P|F|I|B|r|l|a|K|M|n|d|u|Z|A|g|e|L|i)$"
)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or ""))


def count_features(text: str) -> dict[str, int]:
    text = str(text or "")
    token_list = tokens(text)
    return {
        "tokens": len(token_list),
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
        "plus": token_list.count("+"),
        "brace_delta": text.count("{") - text.count("}"),
        "paren_delta": text.count("(") - text.count(")"),
        "unk": text.count(r"\unk"),
    }


def family_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for tok in tokens(text):
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


def l1_delta(left: Counter[str], right: Counter[str], prefix: str | None = None) -> int:
    keys = set(left) | set(right)
    if prefix:
        keys = {key for key in keys if key.startswith(prefix)}
    return sum(abs(left.get(key, 0) - right.get(key, 0)) for key in keys)


def compact(text: Any, limit: int = 260) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def parse_graph_result(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) != 5:
                continue
            key, structure_flag, em_flag, label, prediction = parts
            rows.append(
                {
                    "key": key,
                    "graph_structure_ok": int(structure_flag) == 0,
                    "graph_em_ok": int(em_flag) == 0,
                    "tool_label": label,
                    "tool_prediction": prediction,
                }
            )
    return rows


def parse_exception_file(path: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows_by_index = {idx: row for idx, row in enumerate(df.to_dict("records"))}
    exceptions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        result_id_match = re.search(r"result_id=(\d+)", line)
        error_match = re.search(r"error=(.*?)\tline=", line)
        result_id = int(result_id_match.group(1)) if result_id_match else None
        error = error_match.group(1) if error_match else "unknown"
        source_row = rows_by_index.get(result_id, {}) if result_id is not None else {}
        exceptions.append(
            {
                "result_id": result_id,
                "key": str(source_row.get("key", "")),
                "error": error,
                "raw": line,
            }
        )
    return exceptions


def choose_bucket(
    *,
    graph_em_ok: bool,
    graph_structure_ok: bool,
    parse_error: str,
    deltas: dict[str, int],
    label_counts: Counter[str],
    prediction_counts: Counter[str],
    length_delta: int,
) -> str:
    if parse_error:
        return "parse_exception"
    if graph_em_ok:
        return "graph_em_ok"
    if graph_structure_ok:
        return "structure_ok_em_fail_global_text"
    if abs(deltas["chemfig"]) > 0 or abs(deltas["arrow"]) > 0:
        return "structure_fail_component_or_reaction_count"
    if length_delta <= -20:
        return "structure_fail_truncated_or_too_short"
    if deltas["brace_delta"] != 0 or deltas["paren_delta"] != 0:
        return "structure_fail_syntax_balance"
    if l1_delta(label_counts, prediction_counts, "atom:") >= 4:
        return "structure_fail_atom_label_or_formula"
    if abs(deltas["branch"]) >= 4 or abs(deltas["ring_anchor"]) >= 2:
        return "structure_fail_topology_branch_ring"
    if l1_delta(label_counts, prediction_counts, "bond_type:") >= 2 or abs(
        deltas["angle_bond"]
    ) >= 4:
        return "structure_fail_bond_type_geometry"
    return "structure_fail_other_graph_mismatch"


def analyze(csv_path: Path, graph_result_path: Path, exception_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "graph_error_summary.json"
    cases_path = out_dir / "graph_error_cases.csv"
    report_path = out_dir / "graph_error_report.md"

    df = pd.read_csv(csv_path).fillna("")
    df["key"] = df["image_name"].astype(str).str.replace(r"\.jpg$", "", regex=True)

    graph_rows = parse_graph_result(graph_result_path)
    graph_by_key = {row["key"]: row for row in graph_rows}
    exceptions = parse_exception_file(exception_path, df)
    exception_by_key = {
        exception["key"]: exception for exception in exceptions if exception.get("key")
    }

    case_rows: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    bucket_features: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_lengths: dict[str, list[int]] = defaultdict(list)
    bucket_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in df.to_dict("records"):
        key = row["key"]
        label = str(row["graph_label"])
        prediction = str(row["prediction"])
        label_features = count_features(label)
        prediction_features = count_features(prediction)
        label_counts = family_counts(label)
        prediction_counts = family_counts(prediction)
        deltas = {
            name: prediction_features[name] - label_features[name]
            for name in label_features
        }
        length_delta = deltas["tokens"]
        graph = graph_by_key.get(key, {})
        exception = exception_by_key.get(key)
        parse_error = exception["error"] if exception else ""
        graph_structure_ok = bool(graph.get("graph_structure_ok", False))
        graph_em_ok = bool(graph.get("graph_em_ok", False))
        bucket = choose_bucket(
            graph_em_ok=graph_em_ok,
            graph_structure_ok=graph_structure_ok,
            parse_error=parse_error,
            deltas=deltas,
            label_counts=label_counts,
            prediction_counts=prediction_counts,
            length_delta=length_delta,
        )

        bucket_counts[bucket] += 1
        bucket_lengths[bucket].append(length_delta)
        for name in [
            "chemfig",
            "arrow",
            "branch",
            "ring_anchor",
            "angle_bond",
            "cjk",
            "digit",
            "brace_delta",
            "tokens",
        ]:
            bucket_features[bucket][f"abs_delta_{name}"] += abs(deltas[name])
        bucket_features[bucket]["atom_l1_delta"] += l1_delta(
            label_counts, prediction_counts, "atom:"
        )
        bucket_features[bucket]["bond_type_l1_delta"] += l1_delta(
            label_counts, prediction_counts, "bond_type:"
        )
        bucket_features[bucket]["macro_l1_delta"] += l1_delta(
            label_counts, prediction_counts, "macro:"
        )

        case = {
            "image_name": row["image_name"],
            "bucket": bucket,
            "graph_em_ok": graph_em_ok,
            "graph_structure_ok": graph_structure_ok,
            "parse_error": parse_error,
            "token_edit_distance": row.get("token_edit_distance", ""),
            "normalized_token_edit_distance": row.get("normalized_token_edit_distance", ""),
            "target_tokens": label_features["tokens"],
            "prediction_tokens": prediction_features["tokens"],
            "length_delta": length_delta,
            **{f"delta_{name}": value for name, value in deltas.items()},
            "atom_l1_delta": l1_delta(label_counts, prediction_counts, "atom:"),
            "bond_type_l1_delta": l1_delta(label_counts, prediction_counts, "bond_type:"),
            "macro_l1_delta": l1_delta(label_counts, prediction_counts, "macro:"),
            "target_preview": compact(label),
            "prediction_preview": compact(prediction),
        }
        case_rows.append(case)
        if len(bucket_examples[bucket]) < 8:
            bucket_examples[bucket].append(
                {
                    "image_name": row["image_name"],
                    "length_delta": length_delta,
                    "target_tokens": label_features["tokens"],
                    "prediction_tokens": prediction_features["tokens"],
                    "token_edit_distance": row.get("token_edit_distance", ""),
                    "parse_error": parse_error,
                    "target_preview": compact(label, 180),
                    "prediction_preview": compact(prediction, 180),
                }
            )

    case_rows.sort(
        key=lambda case: (
            case["bucket"] == "graph_em_ok",
            -float(case["token_edit_distance"] or 0),
            case["image_name"],
        )
    )
    with cases_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    parsed_total = len(graph_rows)
    exception_total = len(exceptions)
    structure_correct = sum(row["graph_structure_ok"] for row in graph_rows)
    em_correct = sum(row["graph_em_ok"] for row in graph_rows)

    bucket_summary: dict[str, Any] = {}
    for bucket, count in bucket_counts.most_common():
        lengths = bucket_lengths[bucket]
        features = bucket_features[bucket]
        sorted_lengths = sorted(lengths)
        bucket_summary[bucket] = {
            "count": count,
            "rate_all_samples": rate(count, len(df)),
            "mean_length_delta": sum(lengths) / len(lengths) if lengths else 0.0,
            "median_length_delta": sorted_lengths[len(sorted_lengths) // 2]
            if sorted_lengths
            else 0,
            "mean_abs_delta_per_case": {
                key: value / count for key, value in features.most_common(12)
            },
            "examples": bucket_examples[bucket],
        }

    summary = {
        "csv_path": str(csv_path),
        "graph_result_path": str(graph_result_path),
        "num_samples": len(df),
        "parsed_total": parsed_total,
        "parse_exception_total": exception_total,
        "graph_em": rate(em_correct, parsed_total),
        "graph_em_correct": int(em_correct),
        "graph_em_total": parsed_total,
        "graph_em_all_sample": rate(em_correct, len(df)),
        "graph_structure_em": rate(structure_correct, parsed_total),
        "graph_structure_em_correct": int(structure_correct),
        "graph_structure_em_total": parsed_total,
        "graph_structure_em_all_sample": rate(structure_correct, len(df)),
        "bucket_counts": dict(bucket_counts.most_common()),
        "bucket_summary": bucket_summary,
        "parse_error_counts": dict(
            Counter(exception["error"] for exception in exceptions).most_common()
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Graph Error Analysis",
        "",
        f"- CSV: `{csv_path}`",
        f"- GraphMatchingTool result: `{graph_result_path}`",
        f"- Samples: {len(df)}",
        f"- Parsed by GraphMatchingTool: {parsed_total}/{len(df)}",
        f"- Parse exceptions: {exception_total}/{len(df)}",
        f"- EM on parsed subset: {em_correct}/{parsed_total} = {rate(em_correct, parsed_total):.4f}",
        f"- Structure EM on parsed subset: {structure_correct}/{parsed_total} = {rate(structure_correct, parsed_total):.4f}",
        f"- EM all-sample lower bound: {em_correct}/{len(df)} = {rate(em_correct, len(df)):.4f}",
        f"- Structure EM all-sample lower bound: {structure_correct}/{len(df)} = {rate(structure_correct, len(df)):.4f}",
        "",
        "## Buckets",
        "",
    ]
    for bucket, info in bucket_summary.items():
        report_lines.append(
            f"- `{bucket}`: {info['count']} ({info['rate_all_samples']:.2%}), "
            f"mean length delta {info['mean_length_delta']:.1f}"
        )
    report_lines.extend(
        [
            "",
            "## Suggested Fix Priority",
            "",
            "1. Select checkpoints by validation Graph EM/Structure EM instead of eval_loss only.",
            "2. Add graph-aware syntax/postprocess repair for parse exceptions and bracket/branch balance.",
            "3. Sweep generation parameters for short outputs: length_penalty, early_stopping=false, num_beams, and max_new_tokens=1024.",
            "4. Add a validation-time GraphMatchingTool callback on a fixed small subset to catch regressions early.",
            "5. Add curriculum/oversampling for long multi-step reactions and high chemfig/arrow-count samples.",
            "6. For structure-ok but EM-fail cases, normalize reagent/global text and target serialization consistently.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "cases": str(cases_path),
                "report": str(report_path),
                "graph_em": summary["graph_em"],
                "structure_em": summary["graph_structure_em"],
                "buckets": summary["bucket_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GraphMatchingTool errors from an eval CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--graph_result", type=Path, required=True)
    parser.add_argument("--exception_file", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.csv, args.graph_result, args.exception_file, args.out_dir)


if __name__ == "__main__":
    main()
