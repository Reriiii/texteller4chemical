from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .utils import get_logger, read_jsonl


logger = get_logger("data")


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class EduChemcSample:
    image_path: Path
    target: str
    image_name: str
    targets: dict[str, str] = field(default_factory=dict)
    rfl_aux: dict[str, Any] | None = None


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
    nested_target_key = target_key.split(".", 1)[1] if target_key.startswith("targets.") else target_key
    targets.setdefault(nested_target_key, target)
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


def load_split_samples(
    split_dir: Path,
    target_key: str = "target",
    rfl_aux_field: str | None = None,
) -> list[EduChemcSample]:
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
                rfl_aux=row.get(rfl_aux_field) if rfl_aux_field else None,
            )
        )
    return samples


def _is_bond_token(token: str) -> bool:
    return ("[:" in token and token.endswith("]")) or (
        token.startswith("?[") and token.endswith("]") and "," in token
    )


def _rfl_tokens_from_aux(aux: dict[str, Any], target: str) -> list[str]:
    tokens = aux.get("tokens")
    if isinstance(tokens, list) and all(isinstance(item, str) for item in tokens):
        return list(tokens)
    return target.split()


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def _rfl_msd_metadata(aux: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    msd = aux.get("msd")
    if isinstance(msd, dict):
        return msd

    bond_token_indices = [idx for idx, token in enumerate(tokens) if _is_bond_token(token)]
    bond_index_by_token = {token_idx: bond_idx for bond_idx, token_idx in enumerate(bond_token_indices)}
    branch_token_indices: list[int] = []
    branch_connection_pairs: list[list[int]] = []
    ring_branch_info = aux.get("ring_branch_info")
    if isinstance(ring_branch_info, list):
        for token_idx, connections in enumerate(ring_branch_info):
            if not _nonempty_list(connections):
                continue
            branch_idx = len(branch_token_indices)
            branch_token_indices.append(token_idx)
            for connected_token_idx in connections:
                if not isinstance(connected_token_idx, int):
                    continue
                bond_idx = bond_index_by_token.get(connected_token_idx)
                if bond_idx is not None:
                    branch_connection_pairs.append([branch_idx, bond_idx])
    return {
        "bond_token_indices": bond_token_indices,
        "branch_token_indices": branch_token_indices,
        "branch_connection_pairs": branch_connection_pairs,
    }


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
        target_length_policy: str = "error",
        rfl_aux_field: str | None = None,
    ) -> None:
        self.samples = load_split_samples(
            split_dir,
            target_key=target_key,
            rfl_aux_field=rfl_aux_field,
        )
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_target_length = max_target_length
        self.tokenize_targets = tokenize_targets
        self.target_length_policy = target_length_policy
        self.rfl_aux_field = rfl_aux_field
        self._target_token_lengths: list[int] | None = None
        if tokenize_targets and validate_target_lengths:
            self._handle_target_lengths(batch_size=length_check_batch_size)

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
        rfl_tokens = (
            _rfl_tokens_from_aux(sample.rfl_aux, sample.target)
            if isinstance(sample.rfl_aux, dict)
            else None
        )
        word_ids: list[int | None] | None = None
        if self.tokenize_targets:
            tokenized, word_ids = self._tokenize_target(sample.target, rfl_tokens)
            item["labels"] = tokenized["input_ids"]
        if isinstance(sample.rfl_aux, dict) and rfl_tokens is not None:
            item.update(
                self._build_rfl_item_features(
                    sample.rfl_aux,
                    rfl_tokens,
                    word_ids,
                )
            )
        return item

    def _tokenize_target(
        self,
        target: str,
        rfl_tokens: list[str] | None = None,
    ) -> tuple[Any, list[int | None] | None]:
        truncation = self.target_length_policy.lower() == "truncate"
        tokenizer_kwargs: dict[str, Any] = {
            "truncation": truncation,
            "padding": False,
            "add_special_tokens": True,
        }
        if truncation:
            tokenizer_kwargs["max_length"] = self.max_target_length
        if rfl_tokens is None:
            return self.tokenizer(target, **tokenizer_kwargs), None

        try:
            tokenized = self.tokenizer(
                rfl_tokens,
                is_split_into_words=True,
                **tokenizer_kwargs,
            )
            word_ids = tokenized.word_ids()
            return tokenized, list(word_ids)
        except Exception:
            tokenized = self.tokenizer(target, **tokenizer_kwargs)
            return tokenized, None

    def _build_rfl_item_features(
        self,
        aux: dict[str, Any],
        tokens: list[str],
        word_ids: list[int | None] | None,
    ) -> dict[str, Any]:
        if word_ids is None:
            return {
                "rfl_token_alignment_ok": False,
                "rfl_branch_token_positions": [],
                "rfl_bond_token_positions": [],
                "rfl_branch_labels": [],
                "rfl_branch_mask": [],
            }

        first_label_pos_by_word = [-1 for _ in tokens]
        token_piece_counts = [0 for _ in tokens]
        for label_pos, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx < 0 or word_idx >= len(tokens):
                continue
            token_piece_counts[word_idx] += 1
            if first_label_pos_by_word[word_idx] < 0:
                first_label_pos_by_word[word_idx] = label_pos

        msd = _rfl_msd_metadata(aux, tokens)
        branch_token_indices = [
            int(idx)
            for idx in msd.get("branch_token_indices", [])
            if isinstance(idx, int) and 0 <= idx < len(tokens)
        ]
        bond_token_indices = [
            int(idx)
            for idx in msd.get("bond_token_indices", [])
            if isinstance(idx, int) and 0 <= idx < len(tokens)
        ]
        valid_branch_map: dict[int, int] = {}
        branch_positions: list[int] = []
        for old_idx, token_idx in enumerate(branch_token_indices):
            label_pos = first_label_pos_by_word[token_idx]
            if label_pos >= 0:
                valid_branch_map[old_idx] = len(branch_positions)
                branch_positions.append(label_pos)

        valid_bond_map: dict[int, int] = {}
        bond_positions: list[int] = []
        for old_idx, token_idx in enumerate(bond_token_indices):
            label_pos = first_label_pos_by_word[token_idx]
            if label_pos >= 0:
                valid_bond_map[old_idx] = len(bond_positions)
                bond_positions.append(label_pos)

        labels = [[0 for _ in bond_positions] for _ in branch_positions]
        for pair in msd.get("branch_connection_pairs", []):
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            branch_idx, bond_idx = pair
            if not isinstance(branch_idx, int) or not isinstance(bond_idx, int):
                continue
            mapped_branch = valid_branch_map.get(branch_idx)
            mapped_bond = valid_bond_map.get(bond_idx)
            if mapped_branch is not None and mapped_bond is not None:
                labels[mapped_branch][mapped_bond] = 1
        mask = [[1 for _ in bond_positions] for _ in branch_positions]
        return {
            "rfl_token_alignment_ok": True,
            "rfl_split_token_count": sum(count > 1 for count in token_piece_counts),
            "rfl_branch_token_positions": branch_positions,
            "rfl_bond_token_positions": bond_positions,
            "rfl_branch_labels": labels,
            "rfl_branch_mask": mask,
        }

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

    def _handle_target_lengths(self, batch_size: int) -> None:
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
        message = (
            f"{len(too_long)} targets exceed max_target_length={self.max_target_length}; "
            f"max_len={max(lengths)}; examples={examples}."
        )
        policy = self.target_length_policy.lower()
        if policy in {"error", "raise"}:
            raise ValueError(
                f"{message} Increase max_target_length or set data.target_length_policy "
                "to 'filter' or 'truncate'."
            )
        if policy == "truncate":
            logger.warning("%s These labels will be truncated.", message)
            return
        if policy != "filter":
            raise ValueError(
                "Unsupported target_length_policy "
                f"{self.target_length_policy!r}; expected 'error', 'filter', or 'truncate'."
            )

        drop_indices = {idx for idx, _ in too_long}
        self.samples = [
            sample
            for idx, sample in enumerate(self.samples)
            if idx not in drop_indices
        ]
        self._target_token_lengths = [
            length
            for idx, length in enumerate(lengths)
            if idx not in drop_indices
        ]
        if not self.samples:
            raise ValueError(f"All samples were filtered by max_target_length={self.max_target_length}.")
        logger.warning("%s Filtered these samples from this split.", message)


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
        if any("rfl_branch_token_positions" in f for f in features):
            self._add_rfl_msd_batch(batch, features)
        return batch

    def _add_rfl_msd_batch(
        self,
        batch: dict[str, Any],
        features: list[dict[str, Any]],
    ) -> None:
        branch_lists = [f.get("rfl_branch_token_positions", []) for f in features]
        bond_lists = [f.get("rfl_bond_token_positions", []) for f in features]
        max_branch = max((len(items) for items in branch_lists), default=0)
        max_bond = max((len(items) for items in bond_lists), default=0)
        batch_size = len(features)
        branch_positions = torch.full((batch_size, max_branch), -1, dtype=torch.long)
        bond_positions = torch.full((batch_size, max_bond), -1, dtype=torch.long)
        branch_labels = torch.zeros((batch_size, max_branch, max_bond), dtype=torch.long)
        branch_mask = torch.zeros((batch_size, max_branch, max_bond), dtype=torch.float32)
        alignment_ok = torch.zeros((batch_size,), dtype=torch.bool)
        split_token_count = torch.zeros((batch_size,), dtype=torch.long)

        for row_idx, feature in enumerate(features):
            branch_items = [int(item) for item in feature.get("rfl_branch_token_positions", [])]
            bond_items = [int(item) for item in feature.get("rfl_bond_token_positions", [])]
            if branch_items:
                branch_positions[row_idx, : len(branch_items)] = torch.tensor(branch_items, dtype=torch.long)
            if bond_items:
                bond_positions[row_idx, : len(bond_items)] = torch.tensor(bond_items, dtype=torch.long)
            raw_labels = feature.get("rfl_branch_labels", [])
            raw_mask = feature.get("rfl_branch_mask", [])
            for branch_idx, row in enumerate(raw_labels[:max_branch]):
                if not isinstance(row, list):
                    continue
                for bond_idx, value in enumerate(row[:max_bond]):
                    branch_labels[row_idx, branch_idx, bond_idx] = int(value)
            for branch_idx, row in enumerate(raw_mask[:max_branch]):
                if not isinstance(row, list):
                    continue
                for bond_idx, value in enumerate(row[:max_bond]):
                    branch_mask[row_idx, branch_idx, bond_idx] = float(value)
            alignment_ok[row_idx] = bool(feature.get("rfl_token_alignment_ok", False))
            split_token_count[row_idx] = int(feature.get("rfl_split_token_count", 0) or 0)

        batch["rfl_branch_token_positions"] = branch_positions
        batch["rfl_bond_token_positions"] = bond_positions
        batch["rfl_branch_labels"] = branch_labels
        batch["rfl_branch_mask"] = branch_mask
        batch["rfl_token_alignment_ok"] = alignment_ok
        batch["rfl_split_token_count"] = split_token_count
