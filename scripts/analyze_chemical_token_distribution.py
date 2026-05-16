from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.tokenizer_utils import (  # noqa: E402
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
) -> list[dict[str, Any]]:
    recommended: list[dict[str, Any]] = []
    for row in rows:
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
        recommended.append(row)
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
        "pieces",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    lines.append("| Token | Category | Count | Pieces |")
    lines.append("|---|---|---:|---:|")
    for row in recommended:
        token = str(row["token"]).replace("|", "\\|")
        lines.append(
            f"| `{token}` | `{row['category']}` | {row['count']} | {row.get('piece_count') or ''} |"
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


def main() -> None:
    args = parse_args()
    texts = load_texts(args)
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
    rows: list[dict[str, Any]] = []
    for rank, (token, count) in enumerate(
        sorted(chem_counter.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        piece_info = tokenizer_piece_info(tokenizer, token)
        rows.append(
            {
                "rank": rank,
                "token": token,
                "category": chemical_token_category(token),
                "count": count,
                "frequency": count / total_chemical if total_chemical else 0.0,
                **piece_info,
            }
        )

    category_summary = distribution_summary(rows)
    recommended = recommend_tokens(
        rows,
        categories=set(args.recommend_categories),
        exclude_categories=set(args.exclude_recommend_categories),
        min_frequency=args.min_frequency,
        max_tokens=args.recommend_max_tokens,
        min_pieces=args.recommend_min_pieces,
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
    )
    print(f"Wrote analysis to {out_dir}")
    print(f"Recommended tokens: {len(recommended)}")


if __name__ == "__main__":
    main()
