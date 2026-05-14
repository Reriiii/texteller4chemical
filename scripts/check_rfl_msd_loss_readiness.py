from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.utils import load_yaml, read_jsonl, save_json, setup_logging  # noqa: E402


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a prepared dataset/config can support the full RFL-MSD loss."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_edu_chemc.yaml"))
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--target_key", type=str, default=None)
    parser.add_argument("--graph_label_key", type=str, default=None)
    parser.add_argument("--rfl_aux_field", type=str, default="rfl")
    parser.add_argument("--max_rows_per_split", type=int, default=1000)
    parser.add_argument("--assume_rfl_decoder", action="store_true")
    parser.add_argument(
        "--out_report",
        type=Path,
        default=Path("outputs/reports/rfl_msd_loss_readiness.json"),
    )
    return parser.parse_args()


def resolve_target_key(config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    data_cfg = config.get("data")
    if isinstance(data_cfg, dict):
        return str(data_cfg.get("train_target_key", data_cfg.get("target_key", "target")))
    return str(config.get("target_key", "target"))


def lookup_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is not None:
        return value
    if key.startswith("targets."):
        targets = row.get("targets")
        if isinstance(targets, dict):
            return targets.get(key.split(".", 1)[1])
    return None


def is_bond_token(token: str) -> bool:
    return ("[:" in token and token.endswith("]")) or (
        token.startswith("?[") and token.endswith("]") and "," in token
    )


def nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def inspect_split(
    dataset_dir: Path,
    split: str,
    *,
    target_key: str,
    graph_label_key: str | None,
    rfl_aux_field: str,
    max_rows: int,
) -> dict[str, Any]:
    metadata_path = dataset_dir / split / "metadata.jsonl"
    if not metadata_path.is_file():
        return {
            "split": split,
            "metadata_path": str(metadata_path),
            "exists": False,
            "rows_scanned": 0,
            "failures": [f"Missing metadata: {metadata_path}"],
        }

    rows = read_jsonl(metadata_path)
    if max_rows > 0:
        rows = rows[:max_rows]

    counters: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "missing_aux": [],
        "bad_aux_lengths": [],
        "missing_target": [],
        "missing_graph_label": [],
    }
    token_lengths: list[int] = []
    branch_connection_counts: list[int] = []
    bond_counts: list[int] = []
    conn_token_rows = 0

    for row_index, row in enumerate(rows):
        target = lookup_value(row, target_key)
        if not isinstance(target, str) or not target.strip():
            counters["missing_target"] += 1
            if len(examples["missing_target"]) < 5:
                examples["missing_target"].append(
                    {"row": row_index, "image_name": row.get("image_name")}
                )
            continue
        if graph_label_key:
            graph_lookup_key = (
                graph_label_key
                if graph_label_key.startswith("targets.")
                else f"targets.{graph_label_key}"
            )
            graph_label = lookup_value(row, graph_lookup_key)
            if not isinstance(graph_label, str) or not graph_label.strip():
                counters["missing_graph_label"] += 1
                if len(examples["missing_graph_label"]) < 5:
                    examples["missing_graph_label"].append(
                        {"row": row_index, "image_name": row.get("image_name")}
                    )

        aux = row.get(rfl_aux_field)
        if not isinstance(aux, dict):
            counters["missing_aux"] += 1
            if len(examples["missing_aux"]) < 5:
                examples["missing_aux"].append(
                    {"row": row_index, "image_name": row.get("image_name")}
                )
            continue

        tokens = aux.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens):
            tokens = target.split()
        token_lengths.append(len(tokens))
        bond_counts.append(sum(1 for token in tokens if is_bond_token(token)))
        if any("conn" in token for token in tokens):
            conn_token_rows += 1

        bad_fields: dict[str, int] = {}
        for field in ("branch_info", "ring_branch_info", "cond_data"):
            value = aux.get(field)
            if isinstance(value, list) and len(value) != len(tokens):
                bad_fields[field] = len(value)
        if bad_fields:
            counters["bad_aux_lengths"] += 1
            if len(examples["bad_aux_lengths"]) < 5:
                examples["bad_aux_lengths"].append(
                    {
                        "row": row_index,
                        "image_name": row.get("image_name"),
                        "tokens": len(tokens),
                        "bad_fields": bad_fields,
                    }
                )

        ring_branch_info = aux.get("ring_branch_info")
        if isinstance(ring_branch_info, list):
            branch_connection_counts.append(
                sum(len(item) for item in ring_branch_info if nonempty_list(item))
            )

    rows_scanned = len(rows)
    usable_aux_rows = rows_scanned - counters["missing_aux"] - counters["missing_target"]
    return {
        "split": split,
        "metadata_path": str(metadata_path),
        "exists": True,
        "rows_scanned": rows_scanned,
        "target_key": target_key,
        "graph_label_key": graph_label_key,
        "rfl_aux_field": rfl_aux_field,
        "missing_target": counters["missing_target"],
        "missing_graph_label": counters["missing_graph_label"],
        "missing_aux": counters["missing_aux"],
        "bad_aux_lengths": counters["bad_aux_lengths"],
        "usable_aux_rows": usable_aux_rows,
        "conn_token_rows": conn_token_rows,
        "token_length_max": max(token_lengths) if token_lengths else 0,
        "bond_count_max": max(bond_counts) if bond_counts else 0,
        "branch_connection_total": sum(branch_connection_counts),
        "branch_connection_rows": sum(1 for count in branch_connection_counts if count > 0),
        "examples": examples,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    target_key = resolve_target_key(config, args.target_key)
    split_reports = [
        inspect_split(
            args.dataset_dir,
            split,
            target_key=target_key,
            graph_label_key=args.graph_label_key,
            rfl_aux_field=args.rfl_aux_field,
            max_rows=args.max_rows_per_split,
        )
        for split in args.splits
    ]

    blockers: list[str] = []
    warnings: list[str] = []
    if "rfl" not in target_key.lower():
        blockers.append(
            f"Active target_key is {target_key!r}; full RFL-MSD loss expects an RFL token target."
        )

    for split_report in split_reports:
        split = split_report["split"]
        if not split_report.get("exists"):
            blockers.extend(split_report.get("failures", []))
            continue
        if split_report["missing_target"]:
            blockers.append(f"{split}: {split_report['missing_target']} scanned rows missing target.")
        if split_report.get("missing_graph_label"):
            blockers.append(
                f"{split}: {split_report['missing_graph_label']} scanned rows missing "
                f"graph label {args.graph_label_key!r}."
            )
        if split_report["missing_aux"]:
            blockers.append(
                f"{split}: {split_report['missing_aux']} scanned rows missing {args.rfl_aux_field!r} auxiliary metadata."
            )
        if split_report["bad_aux_lengths"]:
            blockers.append(
                f"{split}: {split_report['bad_aux_lengths']} scanned rows have mismatched RFL auxiliary lengths."
            )
        if split_report["branch_connection_total"] == 0:
            warnings.append(f"{split}: no positive branch connections found in scanned rows.")

    model_ready = bool(args.assume_rfl_decoder)
    if not model_ready:
        blockers.append(
            "Current training path is TexTeller Seq2Seq token generation only; it does not produce "
            "branch_logits/branch_mask required by Lcls."
        )

    report = {
        "config": str(args.config),
        "dataset_dir": str(args.dataset_dir),
        "target_key": target_key,
        "graph_label_key": args.graph_label_key,
        "rfl_aux_field": args.rfl_aux_field,
        "assume_rfl_decoder": args.assume_rfl_decoder,
        "data_ready": not any(
            blocker.startswith(tuple(args.splits)) or blocker.startswith("Missing metadata")
            for blocker in blockers
        )
        and "rfl" in target_key.lower(),
        "model_ready": model_ready,
        "ready_for_full_rfl_msd_loss": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "splits": split_reports,
        "required_model_outputs": [
            "token_logits [B,L,V] or [L,B,V]",
            "labels [B,L]",
            "branch_logits [B,L_branch,L_bond,2]",
            "branch_labels [B,L_branch,L_bond]",
            "branch_mask [B,L_branch,L_bond]",
        ],
    }
    save_json(report, args.out_report)

    status = "READY" if report["ready_for_full_rfl_msd_loss"] else "NOT READY"
    logger.info("RFL-MSD loss readiness: %s", status)
    for blocker in blockers:
        logger.info("blocker: %s", blocker)
    for warning in warnings:
        logger.info("warning: %s", warning)
    logger.info("Wrote %s", args.out_report)


if __name__ == "__main__":
    main()
