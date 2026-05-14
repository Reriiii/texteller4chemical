from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.rfl_adapter import convert_ssml_to_rfl, restore_rfl_text_to_chemfig  # noqa: E402
from chemtexteller.utils import ensure_dir, read_jsonl, save_json, setup_logging, write_jsonl  # noqa: E402


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a prepared EDU-CHEMC dataset variant with RFL-MSD targets."
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--source_key", type=str, default="ssml_normed")
    parser.add_argument("--target_field", type=str, default="ssml_rfl")
    parser.add_argument(
        "--graph_label_field",
        type=str,
        default="ssml_rfl_graph_norm",
        help=(
            "Extra target field containing ChemFig restored from the ground-truth RFL "
            "sequence using exact MSD branch pairs. Use this field as the RFL graph-eval label."
        ),
    )
    parser.add_argument("--rfl_tool_dir", type=Path, default=Path("external/RFL-MSD"))
    parser.add_argument(
        "--aux_field",
        type=str,
        default="rfl",
        help="Metadata field used to store RFL-MSD auxiliary branch/ring labels.",
    )
    parser.add_argument(
        "--no_auxiliary",
        action="store_true",
        help="Only write the serialized RFL target, not branch/ring auxiliary labels.",
    )
    parser.add_argument(
        "--need_ring_count",
        action="store_true",
        help=(
            "Ask RFL-MSD to return ring_count when the cloned converter supports it. "
            "Some public revisions ignore this flag internally, so the default is off."
        ),
    )
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--on_error",
        choices=["fallback", "skip", "raise"],
        default="raise",
        help=(
            "How to handle RFL conversion failures. fallback keeps source_key as "
            "the target and is only for debugging."
        ),
    )
    return parser.parse_args()


def target_name(key: str) -> str:
    return key.split(".", 1)[1] if key.startswith("targets.") else key


def lookup_value(row: dict[str, Any], key: str) -> str:
    direct = row.get(key)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    targets = row.get("targets")
    if isinstance(targets, dict):
        nested_key = key.split(".", 1)[1] if key.startswith("targets.") else key
        nested = targets.get(nested_key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    raise KeyError(key)


def resolve_image_path(split_dir: Path, row: dict[str, Any]) -> str:
    file_name = row.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("metadata row is missing file_name")
    image_path = Path(file_name)
    if not image_path.is_absolute():
        image_path = split_dir / image_path
    return str(image_path.resolve())


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def validate_rfl_result(result: Any) -> str | None:
    if not result.success:
        return result.error or "RFL conversion failed"
    if not result.tokens:
        return "RFL converter returned no tokens"
    token_count = len(result.tokens)
    for field_name in ("branch_info", "ring_branch_info", "cond_data"):
        field_value = getattr(result, field_name)
        if field_value is not None and len(field_value) != token_count:
            return (
                f"RFL {field_name} length mismatch: "
                f"{len(field_value)} != {token_count}"
            )
    return None


def is_bond_token(token: str) -> bool:
    return ("[:" in token and token.endswith("]")) or (
        token.startswith("?[") and token.endswith("]") and "," in token
    )


def build_msd_auxiliary(result: Any) -> dict[str, Any]:
    tokens = list(result.tokens)
    bond_token_indices = [idx for idx, token in enumerate(tokens) if is_bond_token(token)]
    bond_index_by_token = {token_idx: bond_idx for bond_idx, token_idx in enumerate(bond_token_indices)}
    branch_token_indices: list[int] = []
    branch_connection_pairs: list[list[int]] = []
    skipped_connections: list[dict[str, Any]] = []

    ring_branch_info = result.ring_branch_info
    if isinstance(ring_branch_info, list):
        for token_idx, connections in enumerate(ring_branch_info):
            if not isinstance(connections, list) or not connections:
                continue
            branch_idx = len(branch_token_indices)
            branch_token_indices.append(token_idx)
            for connected_token_idx in connections:
                if not isinstance(connected_token_idx, int):
                    skipped_connections.append(
                        {
                            "branch_token_index": token_idx,
                            "connected_token_index": connected_token_idx,
                            "reason": "non_integer_connection",
                        }
                    )
                    continue
                bond_idx = bond_index_by_token.get(connected_token_idx)
                if bond_idx is None:
                    skipped_connections.append(
                        {
                            "branch_token_index": token_idx,
                            "connected_token_index": connected_token_idx,
                            "reason": "connection_token_is_not_a_bond",
                        }
                    )
                    continue
                branch_connection_pairs.append([branch_idx, bond_idx])

    return {
        "bond_token_indices": bond_token_indices,
        "branch_token_indices": branch_token_indices,
        "branch_connection_pairs": branch_connection_pairs,
        "branch_label_shape": [len(branch_token_indices), len(bond_token_indices)],
        "skipped_connections": skipped_connections,
    }


def msd_connection_token_pairs(msd_aux: dict[str, Any]) -> list[tuple[int, int]]:
    branch_token_indices = list(msd_aux.get("branch_token_indices") or [])
    bond_token_indices = list(msd_aux.get("bond_token_indices") or [])
    pairs: list[tuple[int, int]] = []
    for branch_idx, bond_idx in msd_aux.get("branch_connection_pairs") or []:
        if branch_idx >= len(branch_token_indices) or bond_idx >= len(bond_token_indices):
            continue
        pairs.append((int(branch_token_indices[branch_idx]), int(bond_token_indices[bond_idx])))
    return pairs


def build_auxiliary_metadata(
    args: argparse.Namespace,
    source_key: str,
    result: Any,
) -> dict[str, Any]:
    return {
        "source_key": source_key,
        "source_target_name": target_name(source_key),
        "target_field": args.target_field,
        "tool_dir": str(args.rfl_tool_dir),
        "tokens": list(result.tokens),
        "branch_info": result.branch_info,
        "ring_branch_info": result.ring_branch_info,
        "cond_data": result.cond_data,
        "ring_count": result.ring_count,
        "msd": build_msd_auxiliary(result),
    }


def convert_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    in_split_dir = args.dataset_dir / split
    rows = read_jsonl(in_split_dir / "metadata.jsonl")
    if args.max_samples_per_split is not None:
        rows = rows[: args.max_samples_per_split]

    out_split_dir = ensure_dir(args.out_dir / split)
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fallback_count = 0
    skipped_count = 0
    token_lengths: list[int] = []
    ring_counts: list[int] = []
    missing_ea_count = 0
    converted_count = 0
    branch_connection_counts: list[int] = []
    branch_candidate_counts: list[int] = []
    bond_candidate_counts: list[int] = []
    graph_label_count = 0
    source_target_name = target_name(args.source_key)

    for idx, row in enumerate(tqdm(rows, desc=f"RFL {split}")):
        image_name = row.get("image_name") or Path(str(row.get("file_name", f"{idx:06d}"))).name
        source = lookup_value(row, args.source_key)
        result = convert_ssml_to_rfl(
            source,
            args.rfl_tool_dir,
            need_ring_num=args.need_ring_count,
        )
        validation_error = validate_rfl_result(result)

        if validation_error is None:
            target = result.target
            status = "converted"
            converted_count += 1
            token_lengths.append(len(result.tokens))
            if result.ring_count is not None:
                ring_counts.append(result.ring_count)
            if "<ea>" not in result.tokens:
                missing_ea_count += 1
            msd_aux = build_msd_auxiliary(result)
            branch_connection_counts.append(len(msd_aux["branch_connection_pairs"]))
            branch_candidate_counts.append(len(msd_aux["branch_token_indices"]))
            bond_candidate_counts.append(len(msd_aux["bond_token_indices"]))
            graph_label = None
            if args.graph_label_field:
                restore = restore_rfl_text_to_chemfig(
                    target,
                    args.rfl_tool_dir,
                    branch_pairs=msd_connection_token_pairs(msd_aux),
                    cond_data=result.cond_data,
                )
                if not restore.success:
                    raise RuntimeError(
                        f"RFL graph-label restore failed for {split} row {idx} "
                        f"({image_name}): {restore.error}"
                    )
                graph_label = restore.chemfig
                graph_label_count += 1
        else:
            failures.append(
                {
                    "split": split,
                    "row": idx,
                    "image_name": image_name,
                    "source_key": args.source_key,
                    "error": validation_error,
                }
            )
            if args.on_error == "raise":
                raise RuntimeError(
                    f"RFL conversion failed for {split} row {idx} "
                    f"({image_name}): {validation_error}"
                )
            if args.on_error == "skip":
                skipped_count += 1
                continue
            target = source
            status = "fallback"
            fallback_count += 1
            msd_aux = {}
            graph_label = None

        targets = dict(row.get("targets") or {})
        targets[source_target_name] = source
        targets[args.target_field] = target
        if graph_label is not None:
            targets[args.graph_label_field] = graph_label
        out_row = dict(row)
        out_row["file_name"] = resolve_image_path(in_split_dir, row)
        out_row["image_name"] = image_name
        out_row["target"] = target
        out_row["target_field"] = args.target_field
        out_row["targets"] = targets
        out_row["target_status"] = status
        if not args.no_auxiliary and status == "converted":
            out_row[args.aux_field] = build_auxiliary_metadata(args, args.source_key, result)
        elif status == "fallback":
            out_row["conversion_error"] = failures[-1]["error"] if failures else None
        output_rows.append(out_row)

    write_jsonl(output_rows, out_split_dir / "metadata.jsonl")
    return {
        "split": split,
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "converted": converted_count,
        "fallback": fallback_count,
        "skipped": skipped_count,
        "failures_preview": failures[:50],
        "failure_count": len(failures),
        "target_len_min": min(token_lengths) if token_lengths else 0,
        "target_len_p50": percentile(token_lengths, 0.50),
        "target_len_p95": percentile(token_lengths, 0.95),
        "target_len_max": max(token_lengths) if token_lengths else 0,
        "ring_count_total": sum(ring_counts),
        "ring_count_max": max(ring_counts) if ring_counts else 0,
        "ring_sample_count": sum(1 for count in ring_counts if count > 0),
        "missing_ea_count": missing_ea_count,
        "branch_connection_total": sum(branch_connection_counts),
        "branch_connection_rows": sum(1 for count in branch_connection_counts if count > 0),
        "branch_candidate_max": max(branch_candidate_counts) if branch_candidate_counts else 0,
        "bond_candidate_max": max(bond_candidate_counts) if bond_candidate_counts else 0,
        "graph_label_field": args.graph_label_field,
        "graph_label_count": graph_label_count,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == dataset_dir or dataset_dir in out_dir.parents:
        raise SystemExit(
            f"Refusing to write inside the source dataset directory: {args.out_dir}"
        )
    if args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory exists. Pass --overwrite to replace: {args.out_dir}")
        shutil.rmtree(args.out_dir)
    ensure_dir(args.out_dir)

    summaries = [convert_split(args, split) for split in args.splits]
    failures = [
        failure
        for split_summary in summaries
        for failure in split_summary.pop("failures", [])
    ]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "source_key": args.source_key,
        "target_field": args.target_field,
        "graph_label_field": args.graph_label_field,
        "aux_field": None if args.no_auxiliary else args.aux_field,
        "need_ring_count": args.need_ring_count,
        "rfl_tool_dir": str(args.rfl_tool_dir),
        "on_error": args.on_error,
        "splits": summaries,
        "failure_count": len(failures),
    }
    save_json(summary, args.out_dir / "target_conversion_summary.json")
    if failures:
        write_jsonl(failures, args.out_dir / "rfl_conversion_failures.jsonl")
    logger.info("Wrote RFL target dataset to %s", args.out_dir)


if __name__ == "__main__":
    main()
