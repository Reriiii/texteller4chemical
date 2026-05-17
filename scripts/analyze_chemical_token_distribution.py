from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.tokenizer_utils import (  # noqa: E402
    chemical_markup_tokens,
    chemical_token_category,
    chemical_token_counter,
    load_hf_tokenizer,
    load_texts_from_csv,
    load_texts_from_metadata,
    whitespace_tokenize,
)
from chemtexteller.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze chemical markup token distribution and tokenizer fragmentation."
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="CSV, JSONL metadata, or text file source. Can be passed multiple times.",
    )
    parser.add_argument(
        "--csv_fields",
        nargs="+",
        default=["ground_truth", "graph_label", "prediction", "raw_prediction"],
    )
    parser.add_argument(
        "--metadata_target_keys",
        nargs="+",
        default=["targets.ssml_graph_norm", "targets.ssml_normed"],
    )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="OleehyO/TexTeller")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--min_frequency", type=int, default=20)
    parser.add_argument("--recommend_max_tokens", type=int, default=48)
    parser.add_argument("--recommend_min_pieces", type=int, default=1)
    parser.add_argument(
        "--candidate_mode",
        choices=["frequency", "error_aware"],
        default="frequency",
        help=(
            "frequency ranks by source occurrence. error_aware also uses eval CSV "
            "misses, graph failures, target length buckets, and tokenizer fragmentation."
        ),
    )
    parser.add_argument(
        "--eval_csv",
        type=Path,
        action="append",
        default=[],
        help="Optional evaluate.py CSV used to score tokens that are missed on failures.",
    )
    parser.add_argument(
        "--graph_result",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional GraphMatchingTool result txt aligned with --eval_csv. "
            "If omitted, rows with exact_match != 1 are treated as failures."
        ),
    )
    parser.add_argument("--eval_reference_field", type=str, default="ground_truth")
    parser.add_argument("--eval_prediction_field", type=str, default="prediction")
    parser.add_argument("--eval_image_name_field", type=str, default="image_name")
    parser.add_argument(
        "--length_buckets",
        nargs="+",
        default=[
            "short:0:32:1.0",
            "medium:33:64:1.2",
            "long:65:128:2.0",
            "very_long:129:256:3.5",
            "extreme:257:inf:4.0",
        ],
        help="Length buckets as name:min:max:weight, using whitespace target length.",
    )
    parser.add_argument(
        "--max_tokens_per_category",
        nargs="+",
        default=[],
        help="Optional category caps, e.g. branch=2 macro=48 ring_marker=50.",
    )
    parser.add_argument(
        "--recommend_categories",
        nargs="+",
        default=["branch", "macro", "ring_marker"],
    )
    parser.add_argument(
        "--exclude_recommend_categories",
        nargs="+",
        default=["bond_geometry"],
    )
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/token_analysis"))
    return parser.parse_args()


def parse_category_caps(items: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for item in items:
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid --max_tokens_per_category item {item!r}; expected category=value."
            )
        caps[key.strip()] = int(value)
    return caps


def parse_length_buckets(items: list[str]) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid --length_buckets item {item!r}; expected name:min:max:weight."
            )
        name, min_text, max_text, weight_text = parts
        max_len = math.inf if max_text.lower() in {"inf", "none", "null"} else int(max_text)
        buckets.append(
            {
                "name": name,
                "min": int(min_text),
                "max": max_len,
                "weight": float(weight_text),
            }
        )
    return buckets


def length_bucket_for(length: int, buckets: list[dict[str, Any]]) -> dict[str, Any]:
    for bucket in buckets:
        if int(bucket["min"]) <= length <= bucket["max"]:
            return bucket
    return buckets[-1]


def clean_image_key(value: Any) -> str:
    return Path(str(value or "")).stem


def chemical_counter_for_text(text: str) -> Counter[str]:
    return Counter(token.strip() for token in chemical_markup_tokens(text) if token.strip())


def parse_graph_result(path: Path) -> dict[str, dict[str, bool]]:
    graph: dict[str, dict[str, bool]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) < 5:
                continue
            image_name, structure_flag, em_flag, _label, _prediction = parts
            try:
                structure_ok = int(structure_flag) == 0
                graph_ok = int(em_flag) == 0
            except ValueError:
                continue
            graph[clean_image_key(image_name)] = {
                "graph_ok": graph_ok,
                "structure_ok": structure_ok,
            }
    return graph


def boolish_success(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "1.0", "true", "yes"}


def update_eval_error_stats(
    stats_by_token: defaultdict[str, dict[str, Any]],
    *,
    eval_csv: Path,
    graph_result: Path | None,
    reference_field: str,
    prediction_field: str,
    image_name_field: str,
    length_buckets: list[dict[str, Any]],
) -> dict[str, Any]:
    graph_by_image = parse_graph_result(graph_result) if graph_result is not None else {}
    bucket_summary: dict[str, dict[str, Any]] = {
        str(bucket["name"]): {
            "samples": 0,
            "graph_failures": 0,
            "target_token_occurrences": 0,
            "missing_token_occurrences": 0,
            "weighted_missing_token_occurrences": 0.0,
        }
        for bucket in length_buckets
    }
    rows_seen = 0
    failures = 0
    with eval_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_seen += 1
            reference = row.get(reference_field, "") or ""
            prediction = row.get(prediction_field, "") or ""
            target_counts = chemical_counter_for_text(reference)
            pred_counts = chemical_counter_for_text(prediction)
            target_len = len(whitespace_tokenize(reference))
            bucket = length_bucket_for(target_len, length_buckets)
            bucket_name = str(bucket["name"])
            bucket_weight = float(bucket["weight"])
            graph_state = graph_by_image.get(clean_image_key(row.get(image_name_field)))
            if graph_state is None:
                failed = not boolish_success(row.get("exact_match")) and not boolish_success(
                    row.get("normalized_exact_match")
                )
            else:
                failed = not graph_state["graph_ok"]
            if failed:
                failures += 1

            summary = bucket_summary[bucket_name]
            summary["samples"] += 1
            summary["graph_failures"] += int(failed)
            summary["target_token_occurrences"] += sum(target_counts.values())

            for token in set(target_counts) | set(pred_counts):
                stats = stats_by_token[token]
                target_count = target_counts.get(token, 0)
                pred_count = pred_counts.get(token, 0)
                missing = max(0, target_count - pred_count)
                extra = max(0, pred_count - target_count)
                stats["eval_target_count"] += target_count
                stats["eval_prediction_count"] += pred_count
                stats["eval_target_rows"] += int(target_count > 0)
                stats["eval_prediction_rows"] += int(pred_count > 0)
                stats[f"target_count_{bucket_name}"] += target_count
                if target_count > 0:
                    stats[f"target_rows_{bucket_name}"] += 1
                if failed:
                    stats["failed_target_count"] += target_count
                    stats["failed_prediction_count"] += pred_count
                    stats["missing_on_graph_fail"] += missing
                    stats["extra_on_graph_fail"] += extra
                    stats["weighted_missing_on_graph_fail"] += missing * bucket_weight
                    stats[f"missing_on_graph_fail_{bucket_name}"] += missing
                    stats[f"weighted_missing_on_graph_fail_{bucket_name}"] += (
                        missing * bucket_weight
                    )
                    stats[f"failed_target_count_{bucket_name}"] += target_count
                    summary["missing_token_occurrences"] += missing
                    summary["weighted_missing_token_occurrences"] += missing * bucket_weight
    return {
        "eval_csv": str(eval_csv),
        "graph_result": str(graph_result) if graph_result is not None else None,
        "rows": rows_seen,
        "graph_failures": failures,
        "length_bucket_summary": bucket_summary,
    }


def load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    for source in args.source:
        if not source.exists():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        if suffix == ".csv":
            texts.extend(load_texts_from_csv(source, args.csv_fields))
        elif suffix in {".jsonl", ".json"}:
            texts.extend(load_texts_from_metadata(source, args.metadata_target_keys))
        else:
            with source.open("r", encoding="utf-8") as f:
                texts.extend(line.strip() for line in f if line.strip())
    return texts


def tokenizer_piece_info(tokenizer: Any | None, token: str) -> dict[str, Any]:
    if tokenizer is None:
        return {"piece_count": None, "pieces": []}
    encoded = tokenizer(token, add_special_tokens=False)
    ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        return {"piece_count": None, "pieces": []}
    pieces = []
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        try:
            pieces = [str(piece) for piece in convert(ids)]
        except Exception:
            pieces = []
    return {"piece_count": len(ids), "pieces": pieces}


def distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        occurrences = sum(int(row["count"]) for row in subset)
        weighted_piece_sum = sum(
            int(row["count"]) * int(row.get("piece_count") or 0)
            for row in subset
            if row.get("piece_count") is not None
        )
        by_category[category] = {
            "unique_tokens": len(subset),
            "occurrences": occurrences,
            "weighted_avg_pieces": (
                weighted_piece_sum / occurrences if occurrences and weighted_piece_sum else None
            ),
            "top_tokens": subset[:10],
        }
    return by_category


def recommend_tokens(
    rows: list[dict[str, Any]],
    *,
    categories: set[str],
    exclude_categories: set[str],
    min_frequency: int,
    max_tokens: int,
    min_pieces: int,
    score_key: str = "count",
    max_tokens_per_category: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    recommended: list[dict[str, Any]] = []
    per_category_counts: Counter[str] = Counter()
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get(score_key) or 0.0),
            -int(row.get("count") or 0),
            str(row.get("token") or ""),
        ),
    )
    for row in ranked_rows:
        category = str(row["category"])
        piece_count = row.get("piece_count")
        if int(row["count"]) < min_frequency:
            continue
        if categories and category not in categories:
            continue
        if category in exclude_categories:
            continue
        if piece_count is not None and int(piece_count) < min_pieces:
            continue
        category_limit = (
            max_tokens_per_category.get(category)
            if isinstance(max_tokens_per_category, dict)
            else None
        )
        if category_limit is not None and per_category_counts[category] >= int(category_limit):
            continue
        recommended.append(row)
        per_category_counts[category] += 1
        if len(recommended) >= max_tokens:
            break
    return recommended


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "rank",
        "token",
        "category",
        "count",
        "frequency",
        "piece_count",
        "candidate_score",
        "eval_target_count",
        "eval_prediction_count",
        "eval_target_rows",
        "eval_prediction_rows",
        "missing_on_graph_fail",
        "extra_on_graph_fail",
        "weighted_missing_on_graph_fail",
        "failed_target_count",
        "failed_prediction_count",
        "pieces",
    ]
    dynamic_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith(
                (
                    "target_count_",
                    "target_rows_",
                    "missing_on_graph_fail_",
                    "weighted_missing_on_graph_fail_",
                    "failed_target_count_",
                )
            )
        }
    )
    fieldnames = fieldnames[:-1] + dynamic_fields + ["pieces"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "pieces": " ".join(str(piece) for piece in row.get("pieces", [])),
                }
            )


def write_markdown(
    path: Path,
    *,
    source_paths: list[Path],
    total_texts: int,
    raw_token_count: int,
    category_summary: dict[str, Any],
    top_rows: list[dict[str, Any]],
    recommended: list[dict[str, Any]],
    eval_summaries: list[dict[str, Any]] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Chemical Token Distribution")
    lines.append("")
    lines.append("## Sources")
    for source in source_paths:
        lines.append(f"- `{source}`")
    lines.append("")
    lines.append(f"- Text fields analyzed: `{total_texts}`")
    lines.append(f"- Raw whitespace-token occurrences: `{raw_token_count}`")
    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Unique | Occurrences | Weighted Avg Pieces |")
    lines.append("|---|---:|---:|---:|")
    for category, summary in sorted(
        category_summary.items(),
        key=lambda item: int(item[1]["occurrences"]),
        reverse=True,
    ):
        avg = summary["weighted_avg_pieces"]
        avg_text = "" if avg is None else f"{avg:.2f}"
        lines.append(
            f"| `{category}` | {summary['unique_tokens']} | {summary['occurrences']} | {avg_text} |"
        )
    lines.append("")
    lines.append("## Recommended Conservative Additions")
    lines.append("")
    lines.append(
        "These exclude bond-geometry tokens by default because the previous run added too many "
        "new high-frequency geometry IDs at once."
    )
    lines.append("")
    lines.append("| Token | Category | Count | Pieces | Score | Missing On Fail | Weighted Missing |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in recommended:
        token = str(row["token"]).replace("|", "\\|")
        lines.append(
            f"| `{token}` | `{row['category']}` | {row['count']} | "
            f"{row.get('piece_count') or ''} | {float(row.get('candidate_score') or 0.0):.2f} | "
            f"{int(row.get('missing_on_graph_fail') or 0)} | "
            f"{float(row.get('weighted_missing_on_graph_fail') or 0.0):.1f} |"
        )
    if eval_summaries:
        lines.append("")
        lines.append("## Eval Error Sources")
        lines.append("")
        for summary in eval_summaries:
            lines.append(
                f"- `{summary['eval_csv']}` rows={summary['rows']} "
                f"graph_failures={summary['graph_failures']}"
            )
            bucket_summary = summary.get("length_bucket_summary", {})
            for bucket_name, bucket in bucket_summary.items():
                lines.append(
                    f"  - `{bucket_name}` samples={bucket['samples']} "
                    f"failures={bucket['graph_failures']} "
                    f"missing={bucket['missing_token_occurrences']} "
                    f"weighted_missing={bucket['weighted_missing_token_occurrences']:.1f}"
                )
    lines.append("")
    lines.append("## Top Tokens Overall")
    lines.append("")
    lines.append("| Token | Category | Count | Pieces |")
    lines.append("|---|---|---:|---:|")
    for row in top_rows:
        token = str(row["token"]).replace("|", "\\|")
        lines.append(
            f"| `{token}` | `{row['category']}` | {row['count']} | {row.get('piece_count') or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def error_aware_candidate_score(row: dict[str, Any]) -> float:
    count = int(row.get("count") or 0)
    piece_count = row.get("piece_count")
    pieces = int(piece_count) if piece_count is not None else 1
    missing = float(row.get("missing_on_graph_fail") or 0.0)
    weighted_missing = float(row.get("weighted_missing_on_graph_fail") or 0.0)
    eval_target_count = float(row.get("eval_target_count") or 0.0)
    failed_target_count = float(row.get("failed_target_count") or 0.0)
    fragmentation = max(0, pieces - 1)
    return (
        math.log1p(count) * 1.0
        + math.log1p(eval_target_count) * 0.5
        + math.log1p(failed_target_count) * 0.8
        + math.log1p(missing) * 2.0
        + math.log1p(weighted_missing) * 2.5
        + fragmentation * 1.5
    )


def main() -> None:
    args = parse_args()
    texts = load_texts(args)
    length_buckets = parse_length_buckets(args.length_buckets)
    raw_counter: Counter[str] = Counter()
    for text in texts:
        raw_counter.update(whitespace_tokenize(text))
    chem_counter = chemical_token_counter(texts)
    tokenizer = None
    if args.tokenizer_path or args.pretrained_model_name_or_path:
        tokenizer = load_hf_tokenizer(
            args.pretrained_model_name_or_path,
            tokenizer_path=args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
        )

    total_chemical = sum(chem_counter.values())
    eval_stats: defaultdict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    eval_summaries: list[dict[str, Any]] = []
    if args.eval_csv:
        if args.graph_result and len(args.graph_result) not in {0, len(args.eval_csv)}:
            raise ValueError(
                "--graph_result must be omitted or passed the same number of times as --eval_csv."
            )
        for idx, eval_csv in enumerate(args.eval_csv):
            graph_result = args.graph_result[idx] if idx < len(args.graph_result) else None
            if not eval_csv.exists():
                raise FileNotFoundError(eval_csv)
            if graph_result is not None and not graph_result.exists():
                raise FileNotFoundError(graph_result)
            eval_summaries.append(
                update_eval_error_stats(
                    eval_stats,
                    eval_csv=eval_csv,
                    graph_result=graph_result,
                    reference_field=args.eval_reference_field,
                    prediction_field=args.eval_prediction_field,
                    image_name_field=args.eval_image_name_field,
                    length_buckets=length_buckets,
                )
            )

    rows: list[dict[str, Any]] = []
    all_tokens = set(chem_counter) | set(eval_stats)
    for rank, (token, count) in enumerate(
        sorted(
            ((token, chem_counter.get(token, 0)) for token in all_tokens),
            key=lambda item: (-item[1], item[0]),
        ),
        start=1,
    ):
        piece_info = tokenizer_piece_info(tokenizer, token)
        row = {
            "rank": rank,
            "token": token,
            "category": chemical_token_category(token),
            "count": count,
            "frequency": count / total_chemical if total_chemical else 0.0,
            **piece_info,
        }
        row.update(eval_stats.get(token, {}))
        row["candidate_score"] = error_aware_candidate_score(row)
        rows.append(row)

    category_summary = distribution_summary(rows)
    score_key = "candidate_score" if args.candidate_mode == "error_aware" else "count"
    recommended = recommend_tokens(
        rows,
        categories=set(args.recommend_categories),
        exclude_categories=set(args.exclude_recommend_categories),
        min_frequency=args.min_frequency,
        max_tokens=args.recommend_max_tokens,
        min_pieces=args.recommend_min_pieces,
        score_key=score_key,
        max_tokens_per_category=parse_category_caps(args.max_tokens_per_category),
    )

    out_dir = ensure_dir(args.out_dir)
    write_csv(out_dir / "chemical_token_candidates.csv", rows)
    (out_dir / "recommended_tokens.txt").write_text(
        "\n".join(str(row["token"]) for row in recommended) + "\n",
        encoding="utf-8",
    )
    summary = {
        "sources": [str(source) for source in args.source],
        "text_fields_analyzed": len(texts),
        "raw_whitespace_token_occurrences": sum(raw_counter.values()),
        "chemical_token_occurrences": total_chemical,
        "unique_chemical_tokens": len(rows),
        "category_summary": category_summary,
        "candidate_mode": args.candidate_mode,
        "eval_summaries": eval_summaries,
        "max_tokens_per_category": parse_category_caps(args.max_tokens_per_category),
        "recommended_tokens": recommended,
    }
    (out_dir / "chemical_token_distribution.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        out_dir / "chemical_token_distribution.md",
        source_paths=args.source,
        total_texts=len(texts),
        raw_token_count=sum(raw_counter.values()),
        category_summary=category_summary,
        top_rows=rows[: args.top_k],
        recommended=recommended,
        eval_summaries=eval_summaries,
    )
    print(f"Wrote analysis to {out_dir}")
    print(f"Recommended tokens: {len(recommended)}")


if __name__ == "__main__":
    main()
