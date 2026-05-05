from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.tokenizer_utils import (
    add_chemical_tokens,
    load_hf_tokenizer,
    load_targets_from_metadata,
    load_vocab_file,
    token_counter,
    tokenizer_unknown_stats,
)
from chemtexteller.utils import ensure_dir, save_json, setup_logging


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pretrained tokenizer coverage.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--target_key", type=str, default="target")
    parser.add_argument("--max_decoder_length", type=int, default=768)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--extend_tokenizer", action="store_true")
    parser.add_argument("--vocab_file", type=Path, default=None)
    parser.add_argument("--output_tokenizer_dir", type=Path, default=Path("outputs/tokenizer_edu_chemc"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--top_k_risky", type=int, default=200)
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
    raw_counter = token_counter(targets)
    report: dict[str, object] = {
        "metadata": str(args.metadata),
        "num_samples": len(targets),
        "raw_whitespace_token_count": sum(raw_counter.values()),
        "raw_unique_token_count": len(raw_counter),
    }

    tokenizer_source = args.tokenizer_path or args.pretrained_model_name_or_path
    if not tokenizer_source:
        if args.extend_tokenizer:
            raise SystemExit("--extend_tokenizer requires --tokenizer_path or --pretrained_model_name_or_path")
        logger.warning("No tokenizer/model path provided; only raw target stats will be written.")
        ensure_dir(args.out_dir)
        save_json(report, args.out_dir / "tokenizer_coverage.json")
        return

    tokenizer = load_hf_tokenizer(
        model_name_or_path=args.pretrained_model_name_or_path or tokenizer_source,
        tokenizer_path=args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    stats = tokenizer_unknown_stats(tokenizer, tqdm(targets, desc="Tokenizing targets"))
    lengths = stats["tokenized_lengths"]
    over_length_count = sum(1 for length in lengths if length > args.max_decoder_length)
    risky_tokens = stats["risky_tokens"].most_common(args.top_k_risky)

    report.update(
        {
            "tokenizer_source": str(tokenizer_source),
            "tokenizer_vocab_size": len(tokenizer),
            "tokenized_length_mean": mean(lengths) if lengths else 0.0,
            "tokenized_length_p95": percentile(lengths, 0.95),
            "tokenized_length_max": max(lengths) if lengths else 0,
            "max_decoder_length": args.max_decoder_length,
            "num_sequences_over_max_decoder_length": over_length_count,
            "unknown_token_count": stats["unknown_token_count"],
            "unknown_token_ratio": stats["unknown_token_ratio"],
            "risky_tokens": risky_tokens,
        }
    )

    if args.extend_tokenizer:
        extra_tokens = list(raw_counter.keys())
        if args.vocab_file is not None:
            extra_tokens.extend(load_vocab_file(args.vocab_file))
        added = add_chemical_tokens(tokenizer, extra_tokens)
        ensure_dir(args.output_tokenizer_dir)
        tokenizer.save_pretrained(args.output_tokenizer_dir)
        report["extended_tokenizer_dir"] = str(args.output_tokenizer_dir)
        report["num_added_tokens"] = added
        report["extended_tokenizer_vocab_size"] = len(tokenizer)
        logger.info("Added %s tokens and saved tokenizer to %s", added, args.output_tokenizer_dir)

    ensure_dir(args.out_dir)
    save_json(report, args.out_dir / "tokenizer_coverage.json")

    md_lines = [
        "# Tokenizer Coverage",
        "",
        f"- Samples: {report['num_samples']}",
        f"- Raw whitespace tokens: {report['raw_whitespace_token_count']}",
        f"- Raw unique tokens: {report['raw_unique_token_count']}",
        f"- Tokenizer vocab size: {report.get('tokenizer_vocab_size', 'n/a')}",
        f"- Tokenized length mean/p95/max: {report.get('tokenized_length_mean', 0):.2f} / "
        f"{report.get('tokenized_length_p95', 0):.2f} / {report.get('tokenized_length_max', 0)}",
        f"- Unknown token ratio: {report.get('unknown_token_ratio', 0):.6f}",
        f"- Sequences over max decoder length: {over_length_count}",
        "",
        "## Risky Tokens",
        "",
    ]
    for token, count in risky_tokens[: args.top_k_risky]:
        md_lines.append(f"- `{token}`: {count}")
    (args.out_dir / "tokenizer_coverage.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote tokenizer coverage report to %s", args.out_dir)


if __name__ == "__main__":
    main()
