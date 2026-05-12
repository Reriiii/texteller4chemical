from __future__ import annotations

import argparse
import csv
import contextlib
import random
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
from chemtexteller.component_decode import (
    TwoPassDecodeConfig,
    decode_components_for_image,
)
from chemtexteller.graph_matching_eval import (
    lookup_target,
    run_graph_matching_tool,
    validate_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.inference import (
    autocast_context,
    generate_from_pixel_values,
    generation_kwargs,
    load_inference_config,
    merge_lora_for_inference,
    move_pixel_values,
    resolve_inference_dtype,
    set_generation_cache,
)
from chemtexteller.metrics import per_sample_metrics, sequence_metrics
from chemtexteller.model_loader import load_pretrained_model_and_tokenizer
from chemtexteller.target_normalization import SSML_GRAPH_NORM_FIELD, normalize_target_for_field
from chemtexteller.transforms import build_transform
from chemtexteller.utils import ensure_dir, save_json, setup_logging


logger = setup_logging()


TEMP_FIELDNAMES = [
    "sample_index",
    "image_name",
    "image_path",
    "ground_truth",
    "graph_label",
    "raw_prediction",
    "first_pass_prediction",
    "prediction",
    "two_pass_used",
    "two_pass_reason",
    "two_pass_crops",
    "two_pass_stitched_prediction",
    "component_predictions",
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
    parser.add_argument(
        "--target_key",
        type=str,
        default=None,
        help=(
            "Metadata target key used for sequence metrics. Defaults to data.eval_target_key, "
            "data.validation_target_key, or data.target_key from the train config."
        ),
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=None,
        help="Optional generation length_penalty. Useful for beam-search decoding sweeps.",
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional beam-search early_stopping override.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=None,
        help="Optional generation min_new_tokens. Use carefully; it can force overlong outputs.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        help="Optional generation no_repeat_ngram_size override.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Optional generation repetition_penalty override.",
    )
    parser.add_argument(
        "--two_pass_decode",
        action="store_true",
        help=(
            "Enable experimental component-level second-pass decoding for wide/long "
            "reaction images. The first pass is still saved for comparison."
        ),
    )
    parser.add_argument(
        "--two_pass_min_aspect_ratio",
        type=float,
        default=2.4,
        help="Trigger second-pass crops when trimmed image width/height is at least this value.",
    )
    parser.add_argument(
        "--two_pass_window_aspect_ratio",
        type=float,
        default=2.2,
        help="Approximate width/height ratio for each horizontal component crop.",
    )
    parser.add_argument(
        "--two_pass_overlap_ratio",
        type=float,
        default=0.20,
        help="Fractional overlap between adjacent horizontal crops.",
    )
    parser.add_argument(
        "--two_pass_max_crops",
        type=int,
        default=5,
        help="Maximum number of horizontal crops in the second pass.",
    )
    parser.add_argument(
        "--two_pass_crop_max_new_tokens",
        type=int,
        default=512,
        help="Maximum generation length per component crop; set <=0 to reuse max_new_tokens.",
    )
    parser.add_argument(
        "--two_pass_selection",
        choices=["syntax_strict", "syntax_or_longer", "syntax", "always", "never"],
        default="syntax_strict",
        help="How to choose whether stitched component output replaces the first pass.",
    )
    parser.add_argument(
        "--two_pass_min_length_gain_tokens",
        type=int,
        default=20,
        help="Minimum stitched-token length gain used by syntax_or_longer selection.",
    )
    parser.add_argument(
        "--two_pass_min_length_ratio",
        type=float,
        default=0.70,
        help="Reject stitched output shorter than this token-count ratio versus the first pass.",
    )
    parser.add_argument(
        "--two_pass_max_length_ratio",
        type=float,
        default=1.25,
        help="Reject stitched output longer than this token-count ratio versus the first pass.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pin host memory for GPU transfer. Defaults to true on CUDA.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples from the selected split.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help=(
            "Shuffle the selected split with this seed before applying --max_samples. "
            "Use this for representative quick evaluation subsets."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Inference dtype. auto uses bf16 on supported CUDA GPUs, otherwise fp16 on CUDA.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_merge_lora", action="store_true")
    parser.add_argument("--output_csv", type=Path, default=Path("outputs/eval_predictions.csv"))
    parser.add_argument("--graph_eval", action="store_true")
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=None)
    parser.add_argument("--graph_label_key", type=str, default=SSML_GRAPH_NORM_FIELD)
    parser.add_argument(
        "--prediction_normalizer",
        type=str,
        default=None,
        help=(
            "Optional target-normalization field applied to decoded predictions before "
            "sequence metrics and graph evaluation, e.g. ssml_graph_sd or ssml_graph_norm."
        ),
    )
    parser.add_argument("--graph_num_workers", type=int, default=8)
    parser.add_argument("--graph_output_txt", type=Path, default=None)
    parser.add_argument("--graph_keep_temp", action="store_true")
    return parser.parse_args()


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
                "metadata.jsonl includes the requested graph label, or pass a different "
                "--graph_label_key."
            ) from exc


def resolve_eval_target_key(config: dict[str, Any], target_key: str | None) -> str:
    if target_key:
        return target_key
    data_cfg = config.get("data", {})
    if isinstance(data_cfg, dict):
        return str(
            data_cfg.get(
                "eval_target_key",
                data_cfg.get("validation_target_key", data_cfg.get("target_key", "target")),
            )
        )
    return "target"


def normalize_prediction(prediction: str, normalizer: str | None) -> str:
    if normalizer is None or normalizer.lower() in {"", "none", "off", "false"}:
        return prediction
    return normalize_target_for_field(prediction, normalizer)


def log_inference_preprocessing(config: dict[str, object]) -> None:
    image_cfg = config.get("image_size", {})
    if not isinstance(image_cfg, dict):
        logger.info("Inference preprocessing | image_size config is not a mapping: %r", image_cfg)
        return
    logger.info(
        "Inference preprocessing | size=%sx%s channels=%s resize_mode=%s "
        "pad_position=%s trim_white_border=%s normalize_mean=%s normalize_std=%s",
        image_cfg.get("height"),
        image_cfg.get("width"),
        image_cfg.get("channels"),
        image_cfg.get("resize_mode"),
        image_cfg.get("pad_position"),
        image_cfg.get("trim_white_border"),
        image_cfg.get("normalize_mean"),
        image_cfg.get("normalize_std"),
    )


def two_pass_config_from_args(args: argparse.Namespace) -> TwoPassDecodeConfig:
    crop_max_new_tokens = (
        None
        if args.two_pass_crop_max_new_tokens is None
        or args.two_pass_crop_max_new_tokens <= 0
        else int(args.two_pass_crop_max_new_tokens)
    )
    return TwoPassDecodeConfig(
        min_aspect_ratio=float(args.two_pass_min_aspect_ratio),
        window_aspect_ratio=float(args.two_pass_window_aspect_ratio),
        overlap_ratio=float(args.two_pass_overlap_ratio),
        max_crops=int(args.two_pass_max_crops),
        crop_max_new_tokens=crop_max_new_tokens,
        selection=str(args.two_pass_selection),
        min_length_gain_tokens=int(args.two_pass_min_length_gain_tokens),
        min_length_ratio=float(args.two_pass_min_length_ratio),
        max_length_ratio=float(args.two_pass_max_length_ratio),
    )


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
    try:
        inference_dtype = resolve_inference_dtype(args.dtype, device)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = load_inference_config(args.model_ckpt, args.config, args.max_new_tokens)
    log_inference_preprocessing(config)
    target_key = resolve_eval_target_key(config, args.target_key)
    bundle = load_pretrained_model_and_tokenizer(
        model_name_or_path=str(args.model_ckpt),
        tokenizer_path=args.tokenizer_path,
        device=str(device),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=inference_dtype,
    )
    bundle.model = merge_lora_for_inference(bundle.model, enabled=not args.no_merge_lora)
    if inference_dtype is not None:
        bundle.model.to(device=device, dtype=inference_dtype)
        logger.info("Using %s inference for generation.", inference_dtype)
    else:
        bundle.model.to(device)
    bundle.model.eval()
    set_generation_cache(bundle.model, enabled=True)

    transform = build_transform(config, train=False, processor=bundle.processor)
    dataset = EduChemcDataset(
        split_dir=args.dataset_dir / args.split,
        tokenizer=bundle.tokenizer,
        transform=transform,
        max_target_length=int(config.get("max_target_length", args.max_new_tokens)),
        target_key=target_key,
        tokenize_targets=False,
    )
    logger.info("Evaluating %s split with target_key=%s.", args.split, target_key)
    if args.graph_eval:
        validate_dataset_graph_labels(dataset, args.graph_label_key)
    sample_indices = list(range(len(dataset)))
    if args.sample_seed is not None:
        rng = random.Random(args.sample_seed)
        rng.shuffle(sample_indices)
    if args.max_samples is not None:
        sample_indices = sample_indices[: args.max_samples]
    process_indices = sample_indices[accelerator.process_index :: accelerator.num_processes]
    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    loader_kwargs: dict[str, object] = {
        "num_workers": args.dataloader_num_workers,
        "pin_memory": pin_memory,
    }
    if args.dataloader_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    eval_dataset = Subset(dataset, process_indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VisionSeq2SeqCollator(bundle.tokenizer, include_metadata=True),
        **loader_kwargs,
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
        bundle.model,
        bundle.tokenizer,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
        min_new_tokens=args.min_new_tokens,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
    )
    logger.info(
        "Generation kwargs | %s",
        ", ".join(f"{key}={value!r}" for key, value in sorted(gen_kwargs.items())),
    )
    two_pass_cfg = two_pass_config_from_args(args) if args.two_pass_decode else None
    if two_pass_cfg is not None:
        logger.info("Two-pass component decoding enabled | %s", two_pass_cfg)
    autocast_ctx = autocast_context(device, inference_dtype)
    with torch.inference_mode(), autocast_ctx:
        progress = tqdm(
            loader,
            desc=f"Evaluating {args.split}",
            disable=not accelerator.is_main_process,
        )
        for batch_idx, batch in enumerate(progress):
            pixel_values = move_pixel_values(batch["pixel_values"], device, inference_dtype)
            generated = generate_from_pixel_values(bundle.model, pixel_values, gen_kwargs)
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
                first_pass_prediction = pred
                component_predictions = ""
                two_pass_used = False
                two_pass_reason = ""
                two_pass_crops = 0
                two_pass_stitched_prediction = ""
                if two_pass_cfg is not None:
                    two_pass = decode_components_for_image(
                        image_path=image_path,
                        first_pass_prediction=first_pass_prediction,
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        transform=transform,
                        gen_kwargs=gen_kwargs,
                        device=device,
                        dtype=inference_dtype,
                        cfg=two_pass_cfg,
                    )
                    pred = two_pass.prediction
                    two_pass_used = two_pass.used
                    two_pass_reason = ";".join(two_pass.reasons)
                    two_pass_crops = two_pass.crop_count
                    two_pass_stitched_prediction = " ".join(
                        two_pass.stitched_prediction.split()
                    )
                    component_predictions = " <COMPONENT> ".join(
                        " ".join(text.split())
                        for text in two_pass.component_predictions
                    )
                normalized_pred = normalize_prediction(pred, args.prediction_normalizer)
                graph_label = ""
                if args.graph_eval:
                    graph_label = lookup_target(metadata_targets, args.graph_label_key)
                sample_metrics = per_sample_metrics(normalized_pred, ref)
                rows.append(
                    {
                        "sample_index": sample_index,
                        "image_name": image_name,
                        "image_path": image_path,
                        "ground_truth": ref,
                        "graph_label": graph_label,
                        "raw_prediction": pred,
                        "first_pass_prediction": first_pass_prediction,
                        "prediction": normalized_pred,
                        "two_pass_used": two_pass_used,
                        "two_pass_reason": two_pass_reason,
                        "two_pass_crops": two_pass_crops,
                        "two_pass_stitched_prediction": two_pass_stitched_prediction,
                        "component_predictions": component_predictions,
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
    if two_pass_cfg is not None and rows:
        two_pass_triggered = sum(bool(str(row["two_pass_reason"]).strip()) for row in rows)
        two_pass_cropped = sum(int(row["two_pass_crops"] or 0) >= 2 for row in rows)
        two_pass_used = sum(str(row["two_pass_used"]).lower() == "true" for row in rows)
        metrics["two_pass_triggered"] = two_pass_triggered
        metrics["two_pass_triggered_rate"] = two_pass_triggered / len(rows)
        metrics["two_pass_cropped"] = two_pass_cropped
        metrics["two_pass_cropped_rate"] = two_pass_cropped / len(rows)
        metrics["two_pass_used"] = two_pass_used
        metrics["two_pass_used_rate"] = two_pass_used / len(rows)
    if args.prediction_normalizer:
        metrics["prediction_normalizer"] = args.prediction_normalizer
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
            "Graph matching metrics: EM=%.6f, Structure EM=%.6f",
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
