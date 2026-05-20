from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.graph_matching_eval import (  # noqa: E402
    lookup_target,
    run_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.metrics import sequence_metrics  # noqa: E402
from chemtexteller.rfl_adapter import restore_rfl_tokens_to_chemfig  # noqa: E402
from chemtexteller.rfl_vocab import RflVocab  # noqa: E402
from chemtexteller.texteller_msd_data import TextellerMsdCollator, TextellerMsdDataset  # noqa: E402
from chemtexteller.texteller_msd_model import load_texteller_msd_checkpoint  # noqa: E402
from chemtexteller.transforms import build_transform  # noqa: E402
from chemtexteller.utils import ensure_dir, load_yaml, save_json, setup_logging  # noqa: E402


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TexTeller encoder + MSD decoder/head.")
    parser.add_argument("--model_ckpt", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/processed/edu_chemc_rfl_msd"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--output_csv", type=Path, default=Path("outputs/eval_texteller_msd_test_beam5.csv"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--graph_eval", action="store_true")
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=Path("external/GraphMatchingTool"))
    parser.add_argument("--graph_label_key", type=str, default="ssml_rfl_graph_norm")
    parser.add_argument("--graph_num_workers", type=int, default=8)
    parser.add_argument("--rfl_tool_dir", type=Path, default=Path("external/RFL-MSD"))
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype | None:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    return None


def load_processor(source: str, trust_remote_code: bool) -> Any | None:
    for cls in (AutoProcessor, AutoImageProcessor):
        try:
            return cls.from_pretrained(source, trust_remote_code=trust_remote_code)
        except Exception:
            continue
    return None


def main() -> None:
    args = parse_args()
    config_path = args.config or args.model_ckpt / "train_config.yaml"
    config = load_yaml(config_path)
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_texteller_msd_checkpoint(
        args.model_ckpt,
        map_location="cpu",
        torch_dtype=None,
        trust_remote_code=args.trust_remote_code,
    )
    if dtype is not None and device.type == "cuda":
        model.to(dtype=dtype)
    model.to(device)
    model.eval()

    vocab = RflVocab.from_file(args.model_ckpt / "rfl_vocab.txt")
    base_source = getattr(model, "base_model_name_or_path", "OleehyO/TexTeller")
    processor = load_processor(str(base_source), args.trust_remote_code)
    data_cfg = dict(config.get("data") or {})
    dataset = TextellerMsdDataset(
        args.dataset_dir / args.split,
        vocab,
        build_transform(config, train=False, processor=processor),
        target_key=str(data_cfg.get("eval_target_key", data_cfg.get("target_key", "targets.ssml_rfl"))),
        rfl_aux_field=str(data_cfg.get("rfl_aux_field", "rfl")),
        max_target_length=int(config.get("max_target_length", 1024)),
        target_length_policy=str(data_cfg.get("target_length_policy", "filter")),
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TextellerMsdCollator(vocab, include_metadata=True),
    )

    rows: list[dict[str, str]] = []
    graph_rows: list[dict[str, str]] = []
    raw_predictions: list[str] = []
    restored_predictions: list[str] = []
    references: list[str] = []
    autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
        pixel_values = batch["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
            beam_outputs = model.generate(
                pixel_values,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
        for idx, beams in enumerate(beam_outputs):
            best = beams[0] if beams else None
            raw_prediction = best.text if best is not None else ""
            restore = (
                restore_rfl_tokens_to_chemfig(
                    best.tokens,
                    args.rfl_tool_dir,
                    branch_pairs=best.branch_pairs,
                    cond_data=best.cond_data,
                )
                if best is not None
                else None
            )
            restored_prediction = restore.chemfig if restore is not None and restore.success else raw_prediction
            reference = str(batch["target_texts"][idx])
            image_name = str(batch["image_names"][idx])
            metadata_targets = batch["metadata_targets"][idx]
            row = {
                "image_name": image_name,
                "ground_truth": reference,
                "prediction": raw_prediction,
                "restored_prediction": restored_prediction,
                "restore_status": "ok" if restore is not None and restore.success else "failed",
                "restore_error": "" if restore is None or restore.success else str(restore.error or ""),
                "score": "" if best is None else f"{best.score:.6f}",
                "branch_pairs": "" if best is None else repr(best.branch_pairs),
                "cond_data": "" if best is None else repr(best.cond_data),
            }
            rows.append(row)
            raw_predictions.append(raw_prediction)
            restored_predictions.append(restored_prediction)
            references.append(reference)
            if args.graph_eval:
                graph_rows.append(
                    {
                        "image_name": image_name,
                        "prediction": restored_prediction,
                        "graph_label": lookup_target(metadata_targets, args.graph_label_key),
                    }
                )

    ensure_dir(args.output_csv.parent)
    fieldnames = [
        "image_name",
        "ground_truth",
        "prediction",
        "restored_prediction",
        "restore_status",
        "restore_error",
        "score",
        "branch_pairs",
        "cond_data",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = sequence_metrics(raw_predictions, references)
    restored_metrics = {
        f"restored_{key}": value
        for key, value in sequence_metrics(restored_predictions, references).items()
    }
    metrics.update(restored_metrics)
    if args.graph_eval and graph_rows:
        rec_path = args.output_csv.with_suffix(".rec.txt")
        lab_path = args.output_csv.with_suffix(".lab.txt")
        graph_output = args.output_csv.with_suffix(".graph_result.txt")
        write_graph_matching_files(graph_rows, rec_path, lab_path)
        graph_result = run_graph_matching_tool(
            args.graph_matching_tool_dir,
            rec_path,
            lab_path,
            graph_output,
            args.graph_num_workers,
        )
        metrics.update(graph_result.metrics)
        metrics["graph_output_txt"] = str(graph_result.output_path)
    save_json(metrics, args.output_csv.with_suffix(".metrics.json"))
    logger.info("Metrics: %s", metrics)
    logger.info("Wrote predictions to %s", args.output_csv)


if __name__ == "__main__":
    main()
