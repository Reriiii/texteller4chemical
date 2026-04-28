from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned TexTeller checkpoint.")
    parser.add_argument("--model_ckpt", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/processed/edu_chemc"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
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


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args)
    bundle = load_pretrained_model_and_tokenizer(
        model_name_or_path=str(args.model_ckpt),
        tokenizer_path=args.tokenizer_path,
        device=device,
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VisionSeq2SeqCollator(bundle.tokenizer, include_metadata=True),
    )

    rows: list[dict[str, object]] = []
    predictions: list[str] = []
    references: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
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
            predictions.extend(decoded)
            references.extend(batch["targets"])
            for image_path, ref, pred in zip(batch["image_paths"], batch["targets"], decoded):
                sample_metrics = per_sample_metrics(pred, ref)
                rows.append(
                    {
                        "image_path": image_path,
                        "ground_truth": ref,
                        "prediction": pred,
                        **sample_metrics,
                    }
                )

    metrics = sequence_metrics(predictions, references)
    ensure_dir(args.output_csv.parent)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "ground_truth",
                "prediction",
                "exact_match",
                "normalized_exact_match",
                "token_edit_distance",
                "normalized_token_edit_distance",
                "char_edit_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = args.output_csv.with_suffix(".metrics.json")
    save_json(metrics, metrics_path)
    logger.info("Metrics: %s", metrics)
    logger.info("Wrote predictions to %s", args.output_csv)


if __name__ == "__main__":
    main()
