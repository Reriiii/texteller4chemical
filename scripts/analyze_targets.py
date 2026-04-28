from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.tokenizer_utils import load_targets_from_metadata, whitespace_tokenize
from chemtexteller.utils import ensure_dir, save_json, setup_logging


logger = setup_logging()

PATTERNS = {
    "chemfig": re.compile(r"\\chemfig"),
    "Chemabove": re.compile(r"\\Chemabove"),
    "branch": re.compile(r"^branch$"),
    "reconnection_mark": re.compile(r"^\?\[[^\]]+\]$"),
    "single_bond_angle": re.compile(r"^-\[:[-+]?\d+(?:\.\d+)?\]"),
    "double_bond_angle": re.compile(r"^=\[:[-+]?\d+(?:\.\d+)?\]"),
    "wedge_bond_angle": re.compile(r"^<\[:[-+]?\d+(?:\.\d+)?\]"),
    "hashed_wedge_angle": re.compile(r"^<:\[:[-+]?\d+(?:\.\d+)?\]"),
    "circle": re.compile(r"\\circle"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EDU-CHEMC target strings.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--target_key", type=str, default="target")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--top_k", type=int, default=100)
    return parser.parse_args()


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1 - frac) + values[hi] * frac)


def main() -> None:
    args = parse_args()
    targets = load_targets_from_metadata(args.metadata, args.target_key)
    token_counts: Counter[str] = Counter()
    pattern_counts = {name: 0 for name in PATTERNS}
    lengths: list[int] = []

    for target in tqdm(targets, desc="Analyzing targets"):
        tokens = whitespace_tokenize(target)
        token_counts.update(tokens)
        lengths.append(len(tokens))
        for token in tokens:
            for name, pattern in PATTERNS.items():
                if pattern.search(token):
                    pattern_counts[name] += 1

    report = {
        "metadata": str(args.metadata),
        "num_samples": len(targets),
        "unique_token_count": len(token_counts),
        "length_mean": mean(lengths) if lengths else 0.0,
        "length_p50": median(lengths) if lengths else 0.0,
        "length_p95": percentile(lengths, 0.95),
        "length_max": max(lengths) if lengths else 0,
        "top_tokens": token_counts.most_common(args.top_k),
        "special_pattern_counts": pattern_counts,
    }

    ensure_dir(args.out_dir)
    save_json(report, args.out_dir / "target_analysis.json")
    md_lines = [
        "# EDU-CHEMC Target Analysis",
        "",
        f"- Metadata: `{args.metadata}`",
        f"- Samples: {report['num_samples']}",
        f"- Unique whitespace tokens: {report['unique_token_count']}",
        f"- Length mean/p50/p95/max: {report['length_mean']:.2f} / "
        f"{report['length_p50']} / {report['length_p95']:.2f} / {report['length_max']}",
        "",
        "## Special Chemical Markup Tokens",
        "",
    ]
    for name, count in pattern_counts.items():
        md_lines.append(f"- `{name}`: {count}")
    md_lines.extend(["", "## Top Tokens", ""])
    for token, count in token_counts.most_common(args.top_k):
        md_lines.append(f"- `{token}`: {count}")

    md_path = args.out_dir / "target_analysis.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s and target_analysis.json", md_path)


if __name__ == "__main__":
    main()
