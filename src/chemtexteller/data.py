from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .utils import read_jsonl


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class EduChemcSample:
    image_path: Path
    target: str


def load_split_samples(split_dir: Path, target_key: str = "target") -> list[EduChemcSample]:
    metadata_path = split_dir / "metadata.jsonl"
    rows = read_jsonl(metadata_path)
    samples: list[EduChemcSample] = []
    for idx, row in enumerate(rows):
        file_name = row.get("file_name")
        target = row.get(target_key)
        if not file_name:
            raise ValueError(f"Missing file_name in {metadata_path}, row {idx}")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Missing string target in {metadata_path}, row {idx}")
        samples.append(EduChemcSample(image_path=split_dir / file_name, target=target))
    return samples


class EduChemcDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        split_dir: Path,
        tokenizer: PreTrainedTokenizerBase,
        transform: Any,
        max_target_length: int = 512,
        target_key: str = "target",
    ) -> None:
        self.samples = load_split_samples(split_dir, target_key=target_key)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        try:
            with Image.open(sample.image_path) as image:
                pixel_values = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {sample.image_path}: {exc}") from exc

        tokenized = self.tokenizer(
            sample.target,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )
        return {
            "pixel_values": pixel_values,
            "labels": tokenized["input_ids"],
            "target": sample.target,
            "image_path": str(sample.image_path),
        }


class VisionSeq2SeqCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        label_pad_token_id: int = -100,
        include_metadata: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.include_metadata = include_metadata
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id before training.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        label_lists = [f["labels"] for f in features]
        max_len = max(len(labels) for labels in label_lists)
        pad_id = int(self.tokenizer.pad_token_id)

        labels = torch.full(
            (len(label_lists), max_len),
            fill_value=self.label_pad_token_id,
            dtype=torch.long,
        )
        decoder_attention_mask = torch.zeros((len(label_lists), max_len), dtype=torch.long)
        for row_idx, label_ids in enumerate(label_lists):
            length = len(label_ids)
            padded = torch.tensor(label_ids, dtype=torch.long)
            labels[row_idx, :length] = padded
            decoder_attention_mask[row_idx, :length] = 1
            labels[row_idx, labels[row_idx] == pad_id] = self.label_pad_token_id

        batch: dict[str, Any] = {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask,
        }
        if self.include_metadata:
            batch["targets"] = [f["target"] for f in features]
            batch["image_paths"] = [f["image_path"] for f in features]
        return batch
