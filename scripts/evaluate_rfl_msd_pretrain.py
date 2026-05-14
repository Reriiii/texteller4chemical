from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.graph_matching_eval import (  # noqa: E402
    run_graph_matching_tool,
    validate_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.rfl_adapter import restore_rfl_text_to_chemfig  # noqa: E402
from chemtexteller.tokenizer_utils import build_special_token_kwargs  # noqa: E402
from chemtexteller.utils import ensure_dir, read_jsonl, save_json, setup_logging  # noqa: E402


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-train diagnostics for the RFL-MSD pipeline. This does not train a model; "
            "it checks whether the prepared RFL targets, auxiliary MSD labels, tokenizer "
            "alignment, and graph-restore oracle are healthy enough to justify training."
        )
    )
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/processed/edu_chemc_rfl_msd"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--target_key", type=str, default="targets.ssml_rfl")
    parser.add_argument("--rfl_aux_field", type=str, default="rfl")
    parser.add_argument("--graph_label_key", type=str, default="ssml_rfl_graph_norm")
    parser.add_argument("--max_rows_per_split", type=int, default=1000)
    parser.add_argument("--oracle_graph_samples", type=int, default=200)
    parser.add_argument("--rfl_tool_dir", type=Path, default=Path("external/RFL-MSD"))
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=Path("external/GraphMatchingTool"))
    parser.add_argument("--graph_num_workers", type=int, default=0)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="OleehyO/TexTeller")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_tokenizer_check", action="store_true")
    parser.add_argument(
        "--out_report",
        type=Path,
        default=Path("outputs/reports/rfl_msd_pretrain_eval.json"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if the report contains blockers.",
    )
    return parser.parse_args()


def lookup_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is not None:
        return value
    targets = row.get("targets")
    if isinstance(targets, dict):
        nested_key = key.split(".", 1)[1] if key.startswith("targets.") else key
        return targets.get(nested_key)
    return None


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def summarize_ints(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0.0, "p95": 0.0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def msd_connection_token_pairs(msd: dict[str, Any]) -> list[tuple[int, int]]:
    branch_token_indices = list(msd.get("branch_token_indices") or [])
    bond_token_indices = list(msd.get("bond_token_indices") or [])
    pairs: list[tuple[int, int]] = []
    for pair in msd.get("branch_connection_pairs") or []:
        if not (isinstance(pair, list) and len(pair) == 2):
            continue
        branch_idx, bond_idx = pair
        if not isinstance(branch_idx, int) or not isinstance(bond_idx, int):
            continue
        if branch_idx >= len(branch_token_indices) or bond_idx >= len(bond_token_indices):
            continue
        pairs.append((int(branch_token_indices[branch_idx]), int(bond_token_indices[bond_idx])))
    return pairs


def prior_candidate_count(branch_token_indices: list[int], bond_token_indices: list[int]) -> int:
    return sum(
        sum(1 for bond_token_idx in bond_token_indices if bond_token_idx < branch_token_idx)
        for branch_token_idx in branch_token_indices
    )


def load_tokenizer(args: argparse.Namespace) -> Any | None:
    if args.skip_tokenizer_check:
        return None
    try:
        from transformers import AutoTokenizer

        source = args.tokenizer_path or args.pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=args.trust_remote_code)
        special_kwargs = build_special_token_kwargs(tokenizer)
        if special_kwargs:
            tokenizer.add_special_tokens(special_kwargs)
        return tokenizer
    except Exception as exc:
        logger.warning("Tokenizer check skipped because tokenizer could not be loaded: %s", exc)
        return None


def tokenizer_word_ids(encoded: Any) -> list[int | None] | None:
    if not hasattr(encoded, "word_ids"):
        return None
    try:
        return list(encoded.word_ids(0))
    except TypeError:
        try:
            return list(encoded.word_ids(batch_index=0))
        except Exception:
            return None
    except Exception:
        return None


def inspect_tokenizer(tokenizer: Any, tokens: list[str]) -> dict[str, Any]:
    if tokenizer is None:
        return {"checked": False}
    try:
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=True,
            return_tensors=None,
        )
        word_ids = tokenizer_word_ids(encoded)
    except Exception as exc:
        return {"checked": False, "error": str(exc)}
    if word_ids is None:
        return {"checked": False, "error": "Tokenizer did not expose word_ids()."}
    piece_counts = [0 for _ in tokens]
    missing_words = set(range(len(tokens)))
    for word_id in word_ids:
        if word_id is None or word_id < 0 or word_id >= len(tokens):
            continue
        piece_counts[word_id] += 1
        missing_words.discard(word_id)
    split_tokens = [tokens[idx] for idx, count in enumerate(piece_counts) if count > 1]
    return {
        "checked": True,
        "token_count": len(tokens),
        "split_token_count": len(split_tokens),
        "missing_word_count": len(missing_words),
        "split_tokens": split_tokens,
    }


def inspect_split(
    args: argparse.Namespace,
    split: str,
    tokenizer: Any | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata_path = args.dataset_dir / split / "metadata.jsonl"
    if not metadata_path.is_file():
        return (
            {
                "split": split,
                "metadata_path": str(metadata_path),
                "exists": False,
                "rows_scanned": 0,
                "blockers": [f"Missing metadata: {metadata_path}"],
            },
            [],
        )

    rows = read_jsonl(metadata_path)
    if args.max_rows_per_split > 0:
        rows = rows[: args.max_rows_per_split]

    counters: Counter[str] = Counter()
    target_lengths: list[int] = []
    branch_counts: list[int] = []
    bond_counts: list[int] = []
    positive_counts: list[int] = []
    all_candidate_counts: list[int] = []
    prior_candidate_counts: list[int] = []
    split_token_counter: Counter[str] = Counter()
    graph_rows: list[dict[str, str]] = []
    examples: dict[str, list[dict[str, Any]]] = {
        "missing_target": [],
        "missing_aux": [],
        "missing_graph_label": [],
        "restore_failures": [],
        "invalid_positive_pairs": [],
    }

    graph_budget = args.oracle_graph_samples
    for row_idx, row in enumerate(tqdm(rows, desc=f"Pretrain RFL {split}")):
        image_name = str(row.get("image_name") or row.get("file_name") or f"{split}_{row_idx}")
        target = lookup_value(row, args.target_key)
        if not isinstance(target, str) or not target.strip():
            counters["missing_target"] += 1
            if len(examples["missing_target"]) < 5:
                examples["missing_target"].append({"row": row_idx, "image_name": image_name})
            continue
        target_tokens = target.split()
        target_lengths.append(len(target_tokens))

        aux = row.get(args.rfl_aux_field)
        if not isinstance(aux, dict):
            counters["missing_aux"] += 1
            if len(examples["missing_aux"]) < 5:
                examples["missing_aux"].append({"row": row_idx, "image_name": image_name})
            continue
        msd = aux.get("msd")
        if not isinstance(msd, dict):
            counters["missing_aux_msd"] += 1
            continue

        graph_label = lookup_value(row, args.graph_label_key)
        if not isinstance(graph_label, str) or not graph_label.strip():
            counters["missing_graph_label"] += 1
            if len(examples["missing_graph_label"]) < 5:
                examples["missing_graph_label"].append({"row": row_idx, "image_name": image_name})

        branch_token_indices = [
            int(item) for item in msd.get("branch_token_indices", []) if isinstance(item, int)
        ]
        bond_token_indices = [
            int(item) for item in msd.get("bond_token_indices", []) if isinstance(item, int)
        ]
        branch_counts.append(len(branch_token_indices))
        bond_counts.append(len(bond_token_indices))
        pairs = msd_connection_token_pairs(msd)
        positive_counts.append(len(pairs))
        all_candidates = len(branch_token_indices) * len(bond_token_indices)
        prior_candidates = prior_candidate_count(branch_token_indices, bond_token_indices)
        all_candidate_counts.append(all_candidates)
        prior_candidate_counts.append(prior_candidates)
        invalid_pairs = [
            [branch_token_idx, bond_token_idx]
            for branch_token_idx, bond_token_idx in pairs
            if bond_token_idx >= branch_token_idx
        ]
        if invalid_pairs:
            counters["invalid_positive_pairs"] += len(invalid_pairs)
            if len(examples["invalid_positive_pairs"]) < 5:
                examples["invalid_positive_pairs"].append(
                    {"row": row_idx, "image_name": image_name, "pairs": invalid_pairs[:5]}
                )

        token_report = inspect_tokenizer(tokenizer, target_tokens)
        if token_report.get("checked"):
            counters["tokenizer_rows_checked"] += 1
            counters["tokenizer_split_tokens"] += int(token_report.get("split_token_count", 0))
            counters["tokenizer_total_tokens"] += int(token_report.get("token_count", 0))
            counters["tokenizer_missing_words"] += int(token_report.get("missing_word_count", 0))
            split_token_counter.update(token_report.get("split_tokens", []))
        elif "error" in token_report:
            counters["tokenizer_errors"] += 1

        if graph_budget > 0 and isinstance(graph_label, str) and graph_label.strip():
            restore = restore_rfl_text_to_chemfig(
                target,
                args.rfl_tool_dir,
                branch_pairs=pairs,
                cond_data=aux.get("cond_data"),
            )
            if restore.success:
                graph_rows.append(
                    {
                        "image_name": image_name,
                        "prediction": restore.chemfig,
                        "graph_label": graph_label,
                    }
                )
                graph_budget -= 1
            else:
                counters["restore_failures"] += 1
                if len(examples["restore_failures"]) < 5:
                    examples["restore_failures"].append(
                        {
                            "row": row_idx,
                            "image_name": image_name,
                            "error": restore.error,
                        }
                    )

    positive_total = sum(positive_counts)
    all_candidate_total = sum(all_candidate_counts)
    prior_candidate_total = sum(prior_candidate_counts)
    report = {
        "split": split,
        "metadata_path": str(metadata_path),
        "exists": True,
        "rows_scanned": len(rows),
        "missing_target": counters["missing_target"],
        "missing_aux": counters["missing_aux"],
        "missing_aux_msd": counters["missing_aux_msd"],
        "missing_graph_label": counters["missing_graph_label"],
        "restore_failures": counters["restore_failures"],
        "target_length_tokens": summarize_ints(target_lengths),
        "branch_count": summarize_ints(branch_counts),
        "bond_count": summarize_ints(bond_counts),
        "positive_branch_pairs": summarize_ints(positive_counts),
        "all_candidate_pairs": summarize_ints(all_candidate_counts),
        "prior_candidate_pairs": summarize_ints(prior_candidate_counts),
        "positive_rate_all_mask": (
            positive_total / all_candidate_total if all_candidate_total else 0.0
        ),
        "positive_rate_prior_mask": (
            positive_total / prior_candidate_total if prior_candidate_total else 0.0
        ),
        "extra_negative_pairs_if_all_mask": all_candidate_total - prior_candidate_total,
        "invalid_positive_pairs_not_prior": counters["invalid_positive_pairs"],
        "tokenizer": {
            "rows_checked": counters["tokenizer_rows_checked"],
            "total_tokens": counters["tokenizer_total_tokens"],
            "split_tokens": counters["tokenizer_split_tokens"],
            "split_token_rate": (
                counters["tokenizer_split_tokens"] / counters["tokenizer_total_tokens"]
                if counters["tokenizer_total_tokens"]
                else 0.0
            ),
            "missing_words": counters["tokenizer_missing_words"],
            "errors": counters["tokenizer_errors"],
            "top_split_tokens": split_token_counter.most_common(30),
        },
        "examples": examples,
    }
    return report, graph_rows


def run_oracle_graph_eval(
    args: argparse.Namespace,
    graph_rows_by_split: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    if not args.graph_matching_tool_dir.exists():
        return {
            "enabled": False,
            "reason": f"Missing GraphMatchingTool: {args.graph_matching_tool_dir}",
        }
    try:
        validate_graph_matching_tool(args.graph_matching_tool_dir)
    except Exception as exc:
        return {"enabled": False, "reason": str(exc)}

    oracle_report: dict[str, Any] = {"enabled": True, "splits": {}}
    temp_dir = ensure_dir(args.out_report.parent / "rfl_msd_pretrain_oracle_graph")
    for split, graph_rows in graph_rows_by_split.items():
        if not graph_rows:
            oracle_report["splits"][split] = {"samples": 0, "skipped": True}
            continue
        rec_path = temp_dir / f"{split}.rec.txt"
        lab_path = temp_dir / f"{split}.lab.txt"
        out_path = temp_dir / f"{split}.graph_result.txt"
        write_graph_matching_files(graph_rows, rec_path, lab_path)
        result = run_graph_matching_tool(
            tool_dir=args.graph_matching_tool_dir,
            rec_path=rec_path,
            lab_path=lab_path,
            output_path=out_path,
            num_workers=args.graph_num_workers,
        )
        oracle_report["splits"][split] = {
            "samples": len(graph_rows),
            "result_path": str(result.output_path),
            **result.metrics,
        }
    return oracle_report


def build_recommendations(report: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    next_steps: list[str] = []
    for split in report["splits"]:
        split_name = split["split"]
        if not split.get("exists"):
            blockers.extend(split.get("blockers", []))
            continue
        if split["missing_target"]:
            blockers.append(f"{split_name}: missing RFL target rows.")
        if split["missing_aux"] or split["missing_aux_msd"]:
            blockers.append(f"{split_name}: missing RFL MSD auxiliary labels.")
        if split["missing_graph_label"]:
            blockers.append(f"{split_name}: missing graph label {report['graph_label_key']!r}.")
        if split["restore_failures"]:
            blockers.append(f"{split_name}: RFL oracle restore failures.")
        if split["invalid_positive_pairs_not_prior"]:
            blockers.append(
                f"{split_name}: positive branch pairs include non-prior bonds; candidate mask assumptions need review."
            )
        tokenizer = split.get("tokenizer", {})
        split_rate = float(tokenizer.get("split_token_rate", 0.0))
        if split_rate > 0.10:
            warnings.append(
                f"{split_name}: tokenizer split rate is {split_rate:.2%}; RFL branch hidden states use first subtokens."
            )
        if float(split.get("positive_rate_all_mask", 0.0)) < 0.02 and split.get("all_candidate_pairs", {}).get("count"):
            warnings.append(
                f"{split_name}: branch classification is highly imbalanced under all-pair mask."
            )
        if split.get("extra_negative_pairs_if_all_mask", 0) > 0:
            warnings.append(
                f"{split_name}: all-pair training mask adds extra negative pairs versus prior-bond candidate mask."
            )

    oracle = report.get("oracle_graph_eval", {})
    if oracle.get("enabled"):
        for split_name, split_result in oracle.get("splits", {}).items():
            if split_result.get("samples", 0) and split_result.get("graph_em", 1.0) < 1.0:
                blockers.append(
                    f"{split_name}: oracle restored RFL label graph_em={split_result.get('graph_em')}; graph-label path is not lossless."
                )

    if blockers:
        next_steps.append("Fix blockers before any long training run.")
    else:
        next_steps.append("Run a tiny overfit on 16-64 samples to verify loss decreases and checkpoint reloads.")
        next_steps.append("Then run a short validation smoke before the full 30-epoch job.")
    if warnings:
        next_steps.append("Review warnings; mask/tokenizer issues can make RFL-MSD look worse than the stable baseline.")
    return blockers, warnings, next_steps


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args)
    split_reports: list[dict[str, Any]] = []
    graph_rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split in args.splits:
        split_report, graph_rows = inspect_split(args, split, tokenizer)
        split_reports.append(split_report)
        graph_rows_by_split[split] = graph_rows
    report: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir),
        "target_key": args.target_key,
        "rfl_aux_field": args.rfl_aux_field,
        "graph_label_key": args.graph_label_key,
        "max_rows_per_split": args.max_rows_per_split,
        "oracle_graph_samples": args.oracle_graph_samples,
        "tokenizer_checked": tokenizer is not None,
        "tokenizer_source": args.tokenizer_path or args.pretrained_model_name_or_path,
        "splits": split_reports,
    }
    report["oracle_graph_eval"] = run_oracle_graph_eval(args, graph_rows_by_split)
    blockers, warnings, next_steps = build_recommendations(report)
    report["blockers"] = blockers
    report["warnings"] = warnings
    report["next_steps"] = next_steps
    report["ready_for_tiny_train"] = not blockers
    save_json(report, args.out_report)
    logger.info(
        "RFL-MSD pretrain evaluation: %s",
        "READY_FOR_TINY_TRAIN" if report["ready_for_tiny_train"] else "BLOCKED",
    )
    for blocker in blockers:
        logger.info("blocker: %s", blocker)
    for warning in warnings:
        logger.info("warning: %s", warning)
    logger.info("Wrote %s", args.out_report)
    if args.strict and blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
