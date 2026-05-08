from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

from tqdm import tqdm
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.tokenizer_utils import load_hf_tokenizer
from chemtexteller.utils import ensure_dir, read_jsonl, save_json, setup_logging


logger = setup_logging()

DEFAULT_CUTOFFS = (512, 768, 896, 1024, 1280, 1536)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tokenized target lengths and practical max_target_length cutoffs."
    )
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/processed/edu_chemc_normed"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--target_key", type=str, default="target")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="OleehyO/TexTeller")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--top_examples", type=int, default=20)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1 - frac) + sorted_values[high] * frac)


def target_from_row(row: dict[str, Any], target_key: str) -> str:
    target = row.get(target_key)
    if isinstance(target, str) and target.strip():
        return target.strip()
    raw_targets = row.get("targets")
    if isinstance(raw_targets, dict):
        nested_key = target_key.split(".", 1)[1] if target_key.startswith("targets.") else target_key
        nested = raw_targets.get(nested_key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    raise KeyError(f"Missing non-empty target for key {target_key!r}")


def load_split_records(dataset_dir: Path, split: str, target_key: str) -> list[dict[str, Any]]:
    metadata_path = dataset_dir / split / "metadata.jsonl"
    rows = read_jsonl(metadata_path)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        target = target_from_row(row, target_key)
        image_name = row.get("image_name") or row.get("file_name") or f"{split}_{idx:06d}"
        records.append(
            {
                "split": split,
                "row_index": idx,
                "image_name": str(image_name),
                "target": target,
            }
        )
    return records


def tokenized_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(records), batch_size), desc="Tokenizing targets"):
        batch = records[start : start + batch_size]
        encoded = tokenizer(
            [item["target"] for item in batch],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )
        for item, input_ids in zip(batch, encoded["input_ids"], strict=True):
            copied = dict(item)
            copied["tokenized_length"] = len(input_ids)
            copied.pop("target", None)
            output.append(copied)
    return output


def cutoff_stats(lengths: list[int], cutoffs: list[int]) -> dict[str, dict[str, float | int]]:
    total = len(lengths)
    stats: dict[str, dict[str, float | int]] = {}
    for cutoff in cutoffs:
        dropped = sum(1 for length in lengths if length > cutoff)
        kept = total - dropped
        stats[str(cutoff)] = {
            "kept": kept,
            "dropped": dropped,
            "dropped_pct": (100.0 * dropped / total) if total else 0.0,
        }
    return stats


def summarize(records: list[dict[str, Any]], cutoffs: list[int], top_examples: int) -> dict[str, Any]:
    lengths = [int(item["tokenized_length"]) for item in records]
    longest = sorted(records, key=lambda item: int(item["tokenized_length"]), reverse=True)
    return {
        "num_samples": len(records),
        "length_mean": mean(lengths) if lengths else 0.0,
        "length_p50": median(lengths) if lengths else 0.0,
        "length_p90": percentile(lengths, 0.90),
        "length_p95": percentile(lengths, 0.95),
        "length_p99": percentile(lengths, 0.99),
        "length_p995": percentile(lengths, 0.995),
        "length_p999": percentile(lengths, 0.999),
        "length_max": max(lengths) if lengths else 0,
        "cutoffs": cutoff_stats(lengths, cutoffs),
        "longest_examples": [
            {
                "split": item["split"],
                "row_index": item["row_index"],
                "image_name": item["image_name"],
                "tokenized_length": item["tokenized_length"],
            }
            for item in longest[:top_examples]
        ],
    }


def decoder_max_position_embeddings(args: argparse.Namespace) -> int | None:
    source = args.tokenizer_path or args.pretrained_model_name_or_path
    if not source:
        return None
    try:
        config = AutoConfig.from_pretrained(str(source), trust_remote_code=args.trust_remote_code)
    except Exception as exc:
        logger.warning("Could not load model config from %s: %s", source, exc)
        return None
    decoder_cfg = getattr(config, "decoder", None)
    value = getattr(decoder_cfg, "max_position_embeddings", None) if decoder_cfg is not None else None
    if value is None:
        value = getattr(config, "max_position_embeddings", None)
    return int(value) if value is not None else None


def recommended_cutoff(cutoffs: list[int], decoder_limit: int | None) -> int:
    if decoder_limit is None:
        return max(cutoffs)
    candidates = [cutoff for cutoff in cutoffs if cutoff <= decoder_limit]
    if candidates:
        return max(candidates)
    return decoder_limit


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Target Length Cutoffs",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Tokenizer: `{report['tokenizer_source']}`",
        f"- Decoder max_position_embeddings: {report['decoder_max_position_embeddings']}",
        f"- Recommended max_target_length: {report['recommended_max_target_length']}",
        "",
    ]
    for split_name, split_report in report["splits"].items():
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- Samples: {split_report['num_samples']}",
                f"- Mean/p50/p95/p99/max: {split_report['length_mean']:.2f} / "
                f"{split_report['length_p50']} / {split_report['length_p95']:.2f} / "
                f"{split_report['length_p99']:.2f} / {split_report['length_max']}",
                "",
                "| Cutoff | Kept | Dropped | Dropped % |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for cutoff, stats in split_report["cutoffs"].items():
            lines.append(
                f"| {cutoff} | {stats['kept']} | {stats['dropped']} | {stats['dropped_pct']:.4f} |"
            )
        lines.append("")
    all_report = report["all"]
    lines.extend(
        [
            "## Longest Examples",
            "",
            "| Split | Row | Image | Length |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for item in all_report["longest_examples"]:
        lines.append(
            f"| {item['split']} | {item['row_index']} | `{item['image_name']}` | "
            f"{item['tokenized_length']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cutoffs = sorted(set(args.cutoffs))
    tokenizer_source = args.tokenizer_path or args.pretrained_model_name_or_path
    tokenizer = load_hf_tokenizer(
        model_name_or_path=args.pretrained_model_name_or_path,
        tokenizer_path=args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    decoder_limit = decoder_max_position_embeddings(args)
    all_records: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}

    for split in args.splits:
        records = load_split_records(args.dataset_dir, split, args.target_key)
        tokenized = tokenized_records(records, tokenizer, args.batch_size)
        all_records.extend(tokenized)
        split_reports[split] = summarize(tokenized, cutoffs, args.top_examples)

    report = {
        "dataset_dir": str(args.dataset_dir),
        "splits_requested": args.splits,
        "target_key": args.target_key,
        "tokenizer_source": str(tokenizer_source),
        "tokenizer_vocab_size": len(tokenizer),
        "decoder_max_position_embeddings": decoder_limit,
        "recommended_max_target_length": recommended_cutoff(cutoffs, decoder_limit),
        "cutoffs": cutoffs,
        "splits": split_reports,
        "all": summarize(all_records, cutoffs, args.top_examples),
    }

    ensure_dir(args.out_dir)
    save_json(report, args.out_dir / "target_length_cutoffs.json")
    write_markdown(report, args.out_dir / "target_length_cutoffs.md")
    logger.info(
        "Recommended max_target_length=%s; wrote reports to %s",
        report["recommended_max_target_length"],
        args.out_dir,
    )


if __name__ == "__main__":
    main()
