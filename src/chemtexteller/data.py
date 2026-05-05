from __future__ import annotations

from dataclasses import dataclass, field
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
    image_name: str
    targets: dict[str, str] = field(default_factory=dict)


def _metadata_targets(row: dict[str, Any], target_key: str, target: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    raw_targets = row.get("targets")
    if isinstance(raw_targets, dict):
        for key, value in raw_targets.items():
            if isinstance(value, str) and value.strip():
                targets[str(key)] = value.strip()

    for key, value in row.items():
        if key in {"file_name", "image_name", "target", "target_field", "targets"}:
            continue
        if isinstance(value, str) and value.strip():
            targets.setdefault(str(key), value.strip())

    target_field = row.get("target_field")
    if isinstance(target_field, str) and target_field:
        targets.setdefault(target_field, target)
    targets.setdefault(target_key, target)
    return targets


def _target_from_row(row: dict[str, Any], target_key: str) -> Any:
    target = row.get(target_key)
    if isinstance(target, str) and target.strip():
        return target
    raw_targets = row.get("targets")
    if not isinstance(raw_targets, dict):
        return target
    nested_key = target_key.split(".", 1)[1] if target_key.startswith("targets.") else target_key
    return raw_targets.get(nested_key, target)


def load_split_samples(split_dir: Path, target_key: str = "target") -> list[EduChemcSample]:
    metadata_path = split_dir / "metadata.jsonl"
    rows = read_jsonl(metadata_path)
    samples: list[EduChemcSample] = []
    for idx, row in enumerate(rows):
        file_name = row.get("file_name")
        target = _target_from_row(row, target_key)
        if not file_name:
            raise ValueError(f"Missing file_name in {metadata_path}, row {idx}")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Missing string target in {metadata_path}, row {idx}")
        image_path = Path(file_name)
        if not image_path.is_absolute():
            image_path = split_dir / image_path
        image_name = row.get("image_name")
        if not isinstance(image_name, str) or not image_name.strip():
            image_name = Path(file_name).name
        samples.append(
            EduChemcSample(
                image_path=image_path,
                target=target,
                image_name=image_name,
                targets=_metadata_targets(row, target_key, target),
            )
        )
    return samples


class EduChemcDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        split_dir: Path,
        tokenizer: PreTrainedTokenizerBase,
        transform: Any,
        max_target_length: int = 512,
        target_key: str = "target",
        tokenize_targets: bool = True,
        validate_target_lengths: bool = True,
        length_check_batch_size: int = 512,
    ) -> None:
        self.samples = load_split_samples(split_dir, target_key=target_key)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_target_length = max_target_length
        self.tokenize_targets = tokenize_targets
        self._target_token_lengths: list[int] | None = None
        if tokenize_targets and validate_target_lengths:
            self._validate_target_lengths(batch_size=length_check_batch_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        try:
            with Image.open(sample.image_path) as image:
                pixel_values = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {sample.image_path}: {exc}") from exc

        item: dict[str, Any] = {
            "pixel_values": pixel_values,
            "target": sample.target,
            "image_path": str(sample.image_path),
            "image_name": sample.image_name,
            "metadata_targets": sample.targets,
        }
        if self.tokenize_targets:
            tokenized = self.tokenizer(
                sample.target,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            item["labels"] = tokenized["input_ids"]
        return item

    def target_token_lengths(self, batch_size: int = 512) -> list[int]:
        if self._target_token_lengths is None:
            lengths: list[int] = []
            texts = [sample.target for sample in self.samples]
            for start in range(0, len(texts), batch_size):
                tokenized = self.tokenizer(
                    texts[start : start + batch_size],
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    verbose=False,
                )
                lengths.extend(len(input_ids) for input_ids in tokenized["input_ids"])
            self._target_token_lengths = lengths
        return self._target_token_lengths

    def _validate_target_lengths(self, batch_size: int) -> None:
        lengths = self.target_token_lengths(batch_size=batch_size)
        too_long = [
            (idx, length)
            for idx, length in enumerate(lengths)
            if length > self.max_target_length
        ]
        if not too_long:
            return
        examples = ", ".join(
            f"{self.samples[idx].image_name}:{length}"
            for idx, length in too_long[:5]
        )
        raise ValueError(
            f"{len(too_long)} targets exceed max_target_length={self.max_target_length}; "
            f"max_len={max(lengths)}; examples={examples}. Increase max_target_length "
            "or filter long targets instead of silently truncating labels."
        )


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
        batch: dict[str, Any] = {
            "pixel_values": pixel_values,
        }
        has_labels = ["labels" in f for f in features]
        if any(has_labels):
            if not all(has_labels):
                raise ValueError("Batch mixes samples with and without labels.")
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
            batch["labels"] = labels
            batch["decoder_attention_mask"] = decoder_attention_mask
        if self.include_metadata:
            batch["targets"] = [f["target"] for f in features]
            batch["image_paths"] = [f["image_path"] for f in features]
            batch["image_names"] = [f["image_name"] for f in features]
            batch["metadata_targets"] = [f["metadata_targets"] for f in features]
        return batch
