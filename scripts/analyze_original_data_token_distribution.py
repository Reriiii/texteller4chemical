from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.tokenizer_utils import (  # noqa: E402
    chemical_markup_tokens,
    chemical_token_category,
    load_hf_tokenizer,
    target_from_metadata_row,
    whitespace_tokenize,
)
from chemtexteller.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream original EDU-CHEMC metadata and analyze chemical-token distribution, "
            "coverage, target-length buckets, and no-bond tokenizer candidates."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/processed/edu_chemc_normed"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--target_key", type=str, default="targets.ssml_normed")
    parser.add_argument(
        "--length_buckets",
        nargs="+",
        default=[
            "short:0:32:1.0",
            "medium:33:64:1.2",
            "long:65:128:2.0",
            "very_long:129:256:3.0",
            "extreme:257:inf:4.0",
        ],
        help="Buckets as name:min:max:weight over whitespace target length.",
    )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--candidate_max_tokens", type=int, default=100)
    parser.add_argument("--candidate_min_frequency", type=int, default=1)
    parser.add_argument(
        "--candidate_categories",
        nargs="+",
        default=["branch", "macro", "ring_marker"],
    )
    parser.add_argument(
        "--exclude_candidate_categories",
        nargs="+",
        default=["bond_geometry", "other"],
    )
    parser.add_argument(
        "--max_tokens_per_category",
        nargs="+",
        default=["branch=2", "macro=48", "ring_marker=50"],
    )
    parser.add_argument(
        "--ring_coverage_topk",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 25, 33],
    )
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/original_token_distribution"))
    return parser.parse_args()


def parse_length_buckets(items: list[str]) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid bucket {item!r}; expected name:min:max:weight.")
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


def parse_category_caps(items: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for item in items:
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


def tokenizer_piece_info(tokenizer: Any | None, token: str) -> dict[str, Any]:
    if tokenizer is None:
        return {"piece_count": None, "pieces": []}
    encoded = tokenizer(token, add_special_tokens=False)
    ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        return {"piece_count": None, "pieces": []}
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    pieces: list[str] = []
    if callable(convert):
        with suppress_tokenizer_errors():
            pieces = [str(piece) for piece in convert(ids)]
    return {"piece_count": len(ids), "pieces": pieces}


class suppress_tokenizer_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
        return True


def safe_target(row: dict[str, Any], target_key: str) -> str:
    try:
        return target_from_metadata_row(row, target_key)
    except KeyError:
        if target_key == "target":
            raise
        return target_from_metadata_row(row, "target")


def update_token_stats(
    stats: defaultdict[str, dict[str, Any]],
    *,
    token: str,
    count: int,
    split: str,
    bucket_name: str,
    bucket_weight: float,
) -> None:
    row = stats[token]
    category = chemical_token_category(token)
    row["token"] = token
    row["category"] = category
    row["count"] += count
    row["doc_count"] += 1
    row[f"count_{split}"] += count
    row[f"docs_{split}"] += 1
    row[f"count_{bucket_name}"] += count
    row[f"docs_{bucket_name}"] += 1
    row["weighted_length_count"] += count * bucket_weight


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    buckets = parse_length_buckets(args.length_buckets)
    stats: defaultdict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    split_summary: dict[str, dict[str, Any]] = {}
    bucket_summary: dict[str, dict[str, Any]] = {
        str(bucket["name"]): {
            "samples": 0,
            "token_occurrences": 0,
            "chemical_token_occurrences": 0,
        }
        for bucket in buckets
    }

    for split in args.splits:
        metadata_path = args.dataset_dir / split / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        summary = {
            "metadata": str(metadata_path),
            "samples": 0,
            "target_token_occurrences": 0,
            "chemical_token_occurrences": 0,
            "min_target_len": None,
            "max_target_len": 0,
            "target_len_sum": 0,
        }
        with metadata_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                target = safe_target(row, args.target_key)
                target_len = len(whitespace_tokenize(target))
                bucket = length_bucket_for(target_len, buckets)
                bucket_name = str(bucket["name"])
                bucket_weight = float(bucket["weight"])
                chemical_counts = Counter(chemical_markup_tokens(target))
                summary["samples"] += 1
                summary["target_token_occurrences"] += target_len
                summary["chemical_token_occurrences"] += sum(chemical_counts.values())
                summary["target_len_sum"] += target_len
                summary["max_target_len"] = max(int(summary["max_target_len"]), target_len)
                current_min = summary["min_target_len"]
                summary["min_target_len"] = (
                    target_len if current_min is None else min(int(current_min), target_len)
                )
                bucket_summary[bucket_name]["samples"] += 1
                bucket_summary[bucket_name]["token_occurrences"] += target_len
                bucket_summary[bucket_name]["chemical_token_occurrences"] += sum(
                    chemical_counts.values()
                )
                for token, count in chemical_counts.items():
                    update_token_stats(
                        stats,
                        token=token,
                        count=int(count),
                        split=split,
                        bucket_name=bucket_name,
                        bucket_weight=bucket_weight,
                    )
        samples = int(summary["samples"])
        summary["avg_target_len"] = (
            float(summary["target_len_sum"]) / samples if samples else 0.0
        )
        split_summary[split] = summary
    return {
        "stats": stats,
        "split_summary": split_summary,
        "bucket_summary": bucket_summary,
        "length_buckets": buckets,
    }


def rows_from_stats(
    stats: dict[str, dict[str, Any]],
    *,
    tokenizer: Any | None,
    total_occurrences: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token, stat in stats.items():
        row = dict(stat)
        row["frequency"] = row["count"] / total_occurrences if total_occurrences else 0.0
        row.update(tokenizer_piece_info(tokenizer, token))
        piece_count = row.get("piece_count")
        fragmentation = max(0, int(piece_count) - 1) if piece_count is not None else 0
        row["candidate_score"] = (
            math.log1p(int(row["count"])) * 1.0
            + math.log1p(float(row.get("weighted_length_count") or 0.0)) * 1.5
            + fragmentation * 1.5
        )
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["count"]), str(row["token"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    categories: set[str],
    exclude_categories: set[str],
    max_tokens: int,
    min_frequency: int,
    max_tokens_per_category: dict[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_category: Counter[str] = Counter()
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("candidate_score") or 0.0),
            -int(row.get("count") or 0),
            str(row.get("token") or ""),
        ),
    )
    for row in ranked:
        category = str(row["category"])
        if int(row["count"]) < min_frequency:
            continue
        if categories and category not in categories:
            continue
        if category in exclude_categories:
            continue
        cap = max_tokens_per_category.get(category)
        if cap is not None and per_category[category] >= cap:
            continue
        selected.append(row)
        per_category[category] += 1
        if len(selected) >= max_tokens:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    dynamic = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith(("count_", "docs_"))
        }
    )
    fieldnames = [
        "rank",
        "token",
        "category",
        "count",
        "doc_count",
        "frequency",
        "weighted_length_count",
        "candidate_score",
        "piece_count",
        *dynamic,
        "pieces",
    ]
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


def coverage_rows(rows: list[dict[str, Any]], topk_values: list[int]) -> list[dict[str, Any]]:
    total = sum(int(row["count"]) for row in rows)
    output: list[dict[str, Any]] = []
    for topk in topk_values:
        chosen = rows[:topk]
        covered = sum(int(row["count"]) for row in chosen)
        output.append(
            {
                "top_k": topk,
                "unique_available": len(rows),
                "covered_occurrences": covered,
                "total_occurrences": total,
                "coverage": covered / total if total else 0.0,
            }
        )
    return output


def write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    split_summary: dict[str, Any],
    bucket_summary: dict[str, Any],
    category_summary: dict[str, Any],
    ring_rows: list[dict[str, Any]],
    ring_coverage: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Original Data Token Distribution")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset_dir}`")
    lines.append(f"- Target key: `{args.target_key}`")
    lines.append("")
    lines.append("## Splits")
    lines.append("")
    lines.append("| Split | Samples | Avg Len | Max Len | Chemical Tokens |")
    lines.append("|---|---:|---:|---:|---:|")
    for split, summary in split_summary.items():
        lines.append(
            f"| `{split}` | {summary['samples']} | {summary['avg_target_len']:.2f} | "
            f"{summary['max_target_len']} | {summary['chemical_token_occurrences']} |"
        )
    lines.append("")
    lines.append("## Categories")
    lines.append("")
    lines.append("| Category | Unique | Occurrences | Coverage |")
    lines.append("|---|---:|---:|---:|")
    for category, summary in sorted(
        category_summary.items(),
        key=lambda item: int(item[1]["occurrences"]),
        reverse=True,
    ):
        lines.append(
            f"| `{category}` | {summary['unique']} | {summary['occurrences']} | "
            f"{summary['coverage']:.4f} |"
        )
    lines.append("")
    lines.append("## Ring Marker Coverage")
    lines.append("")
    lines.append("| Top K | Unique Available | Covered | Total | Coverage |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in ring_coverage:
        lines.append(
            f"| {row['top_k']} | {row['unique_available']} | {row['covered_occurrences']} | "
            f"{row['total_occurrences']} | {row['coverage']:.6f} |"
        )
    lines.append("")
    lines.append("Top ring markers:")
    lines.append("")
    lines.append("| Rank | Token | Count | Coverage | Cumulative |")
    lines.append("|---:|---|---:|---:|---:|")
    total_ring = sum(int(row["count"]) for row in ring_rows)
    cumulative = 0
    for row in ring_rows[:40]:
        cumulative += int(row["count"])
        token = str(row["token"]).replace("|", "\\|")
        lines.append(
            f"| {row['rank']} | `{token}` | {row['count']} | "
            f"{int(row['count']) / total_ring:.6f} | {cumulative / total_ring:.6f} |"
        )
    lines.append("")
    lines.append("## No-Bond Candidate Tokens")
    lines.append("")
    lines.append("| Token | Category | Count | Score | Pieces |")
    lines.append("|---|---|---:|---:|---:|")
    for row in candidates:
        token = str(row["token"]).replace("|", "\\|")
        pieces = "" if row.get("piece_count") is None else str(row["piece_count"])
        lines.append(
            f"| `{token}` | `{row['category']}` | {row['count']} | "
            f"{float(row['candidate_score']):.2f} | {pieces} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tokenizer = None
    if args.tokenizer_path or args.pretrained_model_name_or_path:
        tokenizer = load_hf_tokenizer(
            args.pretrained_model_name_or_path or "",
            tokenizer_path=args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
        )
    result = analyze(args)
    stats = result["stats"]
    total_occurrences = sum(int(row["count"]) for row in stats.values())
    rows = rows_from_stats(stats, tokenizer=tokenizer, total_occurrences=total_occurrences)

    category_summary: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        occurrences = sum(int(row["count"]) for row in subset)
        category_summary[category] = {
            "unique": len(subset),
            "occurrences": occurrences,
            "coverage": occurrences / total_occurrences if total_occurrences else 0.0,
        }

    ring_rows = [row for row in rows if row["category"] == "ring_marker"]
    ring_rows.sort(key=lambda row: (-int(row["count"]), str(row["token"])))
    for rank, row in enumerate(ring_rows, start=1):
        row["rank"] = rank
    ring_coverage = coverage_rows(ring_rows, args.ring_coverage_topk)

    candidates = select_candidates(
        rows,
        categories=set(args.candidate_categories),
        exclude_categories=set(args.exclude_candidate_categories),
        max_tokens=args.candidate_max_tokens,
        min_frequency=args.candidate_min_frequency,
        max_tokens_per_category=parse_category_caps(args.max_tokens_per_category),
    )

    out_dir = ensure_dir(args.out_dir)
    write_csv(out_dir / "chemical_token_distribution.csv", rows)
    write_csv(out_dir / "ring_marker_distribution.csv", ring_rows)
    write_csv(out_dir / "no_bond_candidate_tokens.csv", candidates)
    (out_dir / "no_bond_candidate_tokens.txt").write_text(
        "\n".join(str(row["token"]) for row in candidates) + "\n",
        encoding="utf-8",
    )
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "target_key": args.target_key,
        "splits": args.splits,
        "split_summary": result["split_summary"],
        "bucket_summary": result["bucket_summary"],
        "category_summary": category_summary,
        "ring_marker_coverage": ring_coverage,
        "candidate_count": len(candidates),
        "candidate_categories": dict(Counter(str(row["category"]) for row in candidates)),
        "candidate_tokens": candidates,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        out_dir / "summary.md",
        args=args,
        split_summary=result["split_summary"],
        bucket_summary=result["bucket_summary"],
        category_summary=category_summary,
        ring_rows=ring_rows,
        ring_coverage=ring_coverage,
        candidates=candidates,
    )
    print(f"Wrote analysis to {out_dir}")
    print(f"Unique chemical tokens: {len(rows)}")
    print(f"Ring marker unique: {len(ring_rows)}")
    print(f"No-bond candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
