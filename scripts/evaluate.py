from __future__ import annotations

import argparse
import csv
import contextlib
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.data import EduChemcDataset, VisionSeq2SeqCollator
from chemtexteller.graph_matching_eval import (
    lookup_target,
    run_graph_matching_tool,
    validate_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.metrics import per_sample_metrics, sequence_metrics
from chemtexteller.model_loader import load_pretrained_model_and_tokenizer
from chemtexteller.transforms import build_transform
from chemtexteller.utils import ensure_dir, load_yaml, save_json, setup_logging


logger = setup_logging()


TEMP_FIELDNAMES = [
    "sample_index",
    "image_name",
    "image_path",
    "ground_truth",
    "graph_label",
    "prediction",
    "exact_match",
    "normalized_exact_match",
    "token_edit_distance",
    "normalized_token_edit_distance",
    "char_edit_distance",
]

OUTPUT_FIELDNAMES = [name for name in TEMP_FIELDNAMES if name != "sample_index"]


class SingleProcessAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.process_index = 0
        self.num_processes = 1
        self.is_main_process = True

    def wait_for_everyone(self) -> None:
        return


def build_accelerator():
    try:
        from accelerate import Accelerator
    except ImportError:
        logger.warning(
            "accelerate is not installed; evaluation will run on a single process/GPU."
        )
        return SingleProcessAccelerator()
    return Accelerator()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned TexTeller checkpoint.")
    parser.add_argument("--model_ckpt", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/processed/edu_chemc"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples from the selected split.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Inference dtype. auto uses bf16 on supported CUDA GPUs, otherwise fp16 on CUDA.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--output_csv", type=Path, default=Path("outputs/eval_predictions.csv"))
    parser.add_argument("--graph_eval", action="store_true")
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=None)
    parser.add_argument("--graph_label_key", type=str, default="ssml_normed")
    parser.add_argument("--graph_num_workers", type=int, default=8)
    parser.add_argument("--graph_output_txt", type=Path, default=None)
    parser.add_argument("--graph_keep_temp", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    if args.config is not None:
        return load_yaml(args.config)
    candidate = args.model_ckpt / "train_config.yaml"
    if candidate.exists():
        return load_yaml(candidate)
    return {
        "max_target_length": args.max_new_tokens,
        "image_size": {"height": 384, "width": 768, "channels": 3},
        "augmentation": {"enabled": False},
    }


def resolve_inference_dtype(dtype_name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or dtype_name == "fp32":
        return None
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def enable_generation_cache(model: torch.nn.Module) -> None:
    candidates = [model]
    if hasattr(model, "get_base_model"):
        with contextlib.suppress(Exception):
            candidates.append(model.get_base_model())
    for attr in ("base_model", "model"):
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)

    for obj in candidates:
        for config_attr in ("config", "generation_config"):
            config_obj = getattr(obj, config_attr, None)
            if config_obj is None:
                continue
            if hasattr(config_obj, "use_cache"):
                config_obj.use_cache = True
            decoder_cfg = getattr(config_obj, "decoder", None)
            if decoder_cfg is not None and hasattr(decoder_cfg, "use_cache"):
                decoder_cfg.use_cache = True


def generation_kwargs(
    tokenizer,
    num_beams: int,
    max_new_tokens: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        kwargs["decoder_start_token_id"] = tokenizer.bos_token_id
    elif tokenizer.cls_token_id is not None:
        kwargs["decoder_start_token_id"] = tokenizer.cls_token_id
    return kwargs


def rank_output_path(output_csv: Path, process_index: int) -> Path:
    return output_csv.with_name(f"{output_csv.stem}.rank{process_index}{output_csv.suffix}")


def graph_output_paths(output_csv: Path, output_txt: Path | None) -> tuple[Path, Path, Path]:
    rec_path = output_csv.with_name(f"{output_csv.stem}.graph_rec.txt")
    lab_path = output_csv.with_name(f"{output_csv.stem}.graph_lab.txt")
    result_path = (
        output_txt
        if output_txt is not None
        else output_csv.with_name(f"{output_csv.stem}.graph_result.txt")
    )
    return rec_path, lab_path, result_path


def validate_graph_args(args: argparse.Namespace) -> None:
    if not args.graph_eval:
        return
    if args.graph_matching_tool_dir is None:
        raise SystemExit(
            "--graph_matching_tool_dir is required when --graph_eval is enabled."
        )
    validate_graph_matching_tool(args.graph_matching_tool_dir)


def validate_dataset_graph_labels(dataset: EduChemcDataset, label_key: str) -> None:
    for idx, sample in enumerate(dataset.samples):
        try:
            lookup_target(sample.targets, label_key)
        except KeyError as exc:
            raise ValueError(
                "Graph evaluation requires metadata label "
                f"{label_key!r}, but it is missing for sample {idx} "
                f"({sample.image_name}). Re-run scripts/prepare_edu_chemc.py so "
                "metadata.jsonl includes targets.ssml_normed, or pass a different "
                "--graph_label_key."
            ) from exc


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rank_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing distributed evaluation shard: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["sample_index"] = int(row["sample_index"])
                rows.append(row)
    rows.sort(key=lambda row: int(row["sample_index"]))
    return rows


def main() -> None:
    args = parse_args()
    accelerator = build_accelerator()
    validate_graph_args(args)
    device = accelerator.device
    config = load_config(args)
    bundle = load_pretrained_model_and_tokenizer(
        model_name_or_path=str(args.model_ckpt),
        tokenizer_path=args.tokenizer_path,
        device=str(device),
        trust_remote_code=args.trust_remote_code,
    )
    bundle.model.eval()
    enable_generation_cache(bundle.model)
    inference_dtype = resolve_inference_dtype(args.dtype, device)
    if inference_dtype is not None:
        bundle.model.to(dtype=inference_dtype)
        logger.info("Using %s inference for generation.", inference_dtype)

    transform = build_transform(config, train=False, processor=bundle.processor)
    dataset = EduChemcDataset(
        split_dir=args.dataset_dir / args.split,
        tokenizer=bundle.tokenizer,
        transform=transform,
        max_target_length=int(config.get("max_target_length", args.max_new_tokens)),
    )
    if args.graph_eval:
        validate_dataset_graph_labels(dataset, args.graph_label_key)
    sample_indices = list(range(len(dataset)))
    if args.max_samples is not None:
        sample_indices = sample_indices[: args.max_samples]
    process_indices = sample_indices[accelerator.process_index :: accelerator.num_processes]
    eval_dataset = Subset(dataset, process_indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VisionSeq2SeqCollator(bundle.tokenizer, include_metadata=True),
    )

    rows: list[dict[str, object]] = []
    if accelerator.is_main_process:
        logger.info(
            "Evaluating %s samples on %s process(es).",
            len(sample_indices),
            accelerator.num_processes,
        )
    logger.info(
        "Process %s evaluating %s samples on %s.",
        accelerator.process_index,
        len(eval_dataset),
        device,
    )
    gen_kwargs = generation_kwargs(
        bundle.tokenizer,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=inference_dtype)
        if inference_dtype is not None
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        progress = tqdm(
            loader,
            desc=f"Evaluating {args.split}",
            disable=not accelerator.is_main_process,
        )
        for batch_idx, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(device)
            if inference_dtype is not None:
                pixel_values = pixel_values.to(dtype=inference_dtype)
            try:
                generated = bundle.model.generate(
                    pixel_values=pixel_values,
                    **gen_kwargs,
                )
            except TypeError:
                generated = bundle.model.generate(
                    inputs=pixel_values,
                    **gen_kwargs,
                )
            decoded = bundle.tokenizer.batch_decode(generated, skip_special_tokens=True)
            start = batch_idx * args.batch_size
            batch_sample_indices = process_indices[start : start + len(decoded)]
            for sample_index, image_name, image_path, metadata_targets, ref, pred in zip(
                batch_sample_indices,
                batch["image_names"],
                batch["image_paths"],
                batch["metadata_targets"],
                batch["targets"],
                decoded,
            ):
                graph_label = ""
                if args.graph_eval:
                    graph_label = lookup_target(metadata_targets, args.graph_label_key)
                sample_metrics = per_sample_metrics(pred, ref)
                rows.append(
                    {
                        "sample_index": sample_index,
                        "image_name": image_name,
                        "image_path": image_path,
                        "ground_truth": ref,
                        "graph_label": graph_label,
                        "prediction": pred,
                        **sample_metrics,
                    }
                )

    rank_path = rank_output_path(args.output_csv, accelerator.process_index)
    write_rows(rank_path, rows, TEMP_FIELDNAMES)
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    rank_paths = [
        rank_output_path(args.output_csv, process_index)
        for process_index in range(accelerator.num_processes)
    ]
    rows = read_rank_rows(rank_paths)
    predictions = [str(row["prediction"]) for row in rows]
    references = [str(row["ground_truth"]) for row in rows]
    metrics = sequence_metrics(predictions, references)
    if args.graph_eval:
        rec_path, lab_path, result_path = graph_output_paths(
            args.output_csv,
            args.graph_output_txt,
        )
        write_graph_matching_files(rows, rec_path, lab_path)
        graph_result = run_graph_matching_tool(
            tool_dir=args.graph_matching_tool_dir,
            rec_path=rec_path,
            lab_path=lab_path,
            output_path=result_path,
            num_workers=args.graph_num_workers,
        )
        metrics.update(graph_result.metrics)
        metrics.update(
            {
                "graph_matching_tool_dir": str(args.graph_matching_tool_dir),
                "graph_label_key": args.graph_label_key,
                "graph_output_txt": str(graph_result.output_path),
            }
        )
        logger.info(
            "Graph matching metrics: EM(struct.line)=%.6f, Structure EM(struct)=%.6f",
            metrics["graph_em"],
            metrics["graph_structure_em"],
        )
        if not args.graph_keep_temp:
            with contextlib.suppress(FileNotFoundError):
                rec_path.unlink()
            with contextlib.suppress(FileNotFoundError):
                lab_path.unlink()
    output_rows = [
        {field: row[field] for field in OUTPUT_FIELDNAMES}
        for row in rows
    ]
    write_rows(args.output_csv, output_rows, OUTPUT_FIELDNAMES)
    metrics_path = args.output_csv.with_suffix(".metrics.json")
    save_json(metrics, metrics_path)
    logger.info("Metrics: %s", metrics)
    logger.info("Wrote predictions to %s", args.output_csv)


if __name__ == "__main__":
    main()
