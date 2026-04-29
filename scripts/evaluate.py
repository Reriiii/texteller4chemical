from __future__ import annotations

import argparse
import csv
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
from chemtexteller.metrics import per_sample_metrics, sequence_metrics
from chemtexteller.model_loader import load_pretrained_model_and_tokenizer
from chemtexteller.transforms import build_transform
from chemtexteller.utils import ensure_dir, load_yaml, save_json, setup_logging


logger = setup_logging()


TEMP_FIELDNAMES = [
    "sample_index",
    "image_path",
    "ground_truth",
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
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--output_csv", type=Path, default=Path("outputs/eval_predictions.csv"))
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


def rank_output_path(output_csv: Path, process_index: int) -> Path:
    return output_csv.with_name(f"{output_csv.stem}.rank{process_index}{output_csv.suffix}")


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
    device = accelerator.device
    config = load_config(args)
    bundle = load_pretrained_model_and_tokenizer(
        model_name_or_path=str(args.model_ckpt),
        tokenizer_path=args.tokenizer_path,
        device=str(device),
        trust_remote_code=args.trust_remote_code,
    )
    bundle.model.eval()

    transform = build_transform(config, train=False, processor=bundle.processor)
    dataset = EduChemcDataset(
        split_dir=args.dataset_dir / args.split,
        tokenizer=bundle.tokenizer,
        transform=transform,
        max_target_length=int(config.get("max_target_length", args.max_new_tokens)),
    )
    process_indices = list(range(accelerator.process_index, len(dataset), accelerator.num_processes))
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
            len(dataset),
            accelerator.num_processes,
        )
    logger.info(
        "Process %s evaluating %s samples on %s.",
        accelerator.process_index,
        len(eval_dataset),
        device,
    )
    with torch.no_grad():
        progress = tqdm(
            loader,
            desc=f"Evaluating {args.split}",
            disable=not accelerator.is_main_process,
        )
        for batch_idx, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(device)
            try:
                generated = bundle.model.generate(
                    pixel_values=pixel_values,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                )
            except TypeError:
                generated = bundle.model.generate(
                    inputs=pixel_values,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                )
            decoded = bundle.tokenizer.batch_decode(generated, skip_special_tokens=True)
            start = batch_idx * args.batch_size
            batch_sample_indices = process_indices[start : start + len(decoded)]
            for sample_index, image_path, ref, pred in zip(
                batch_sample_indices,
                batch["image_paths"],
                batch["targets"],
                decoded,
            ):
                sample_metrics = per_sample_metrics(pred, ref)
                rows.append(
                    {
                        "sample_index": sample_index,
                        "image_path": image_path,
                        "ground_truth": ref,
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
