from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import load_split_samples
from .rfl_adapter import is_rfl_bond_token, is_rfl_super_token
from .rfl_vocab import RflVocab


def _tokens_from_aux(aux: dict[str, Any] | None, target: str) -> list[str]:
    if isinstance(aux, dict):
        tokens = aux.get("tokens")
        if isinstance(tokens, list) and all(isinstance(item, str) for item in tokens):
            return list(tokens)
    return target.split()


def _sequence_field(aux: dict[str, Any] | None, key: str, length: int, default: Any) -> list[Any]:
    if isinstance(aux, dict):
        value = aux.get(key)
        if isinstance(value, list):
            values = list(value[:length])
            if len(values) < length:
                values.extend([default for _ in range(length - len(values))])
            return values
    return [default for _ in range(length)]


def build_msd_item_features(
    tokens_without_eos: list[str],
    aux: dict[str, Any] | None,
    vocab: RflVocab,
) -> dict[str, Any]:
    tokens = [*tokens_without_eos, "</s>"]
    seq_len = len(tokens)
    ring_branch_info = _sequence_field(aux, "ring_branch_info", len(tokens_without_eos), None)
    ring_branch_info.append(None)
    cond_source = _sequence_field(aux, "cond_data", len(tokens_without_eos), -1)
    cond_source.append(-1)

    memory_index: list[int] = []
    memory_index_by_token: dict[int, int] = {}
    memory_used_indices: list[list[int]] = []
    mem_update_info = [-1 for _ in range(seq_len)]

    bond_index: list[int] = []
    bond_index_by_token: dict[int, int] = {}
    bond_update_info = [-1 for _ in range(seq_len)]

    branch_update_info = [-1 for _ in range(seq_len)]
    branch_index_by_token: dict[int, int] = {}
    branch_pairs: list[list[int]] = []
    remaining_memory: list[int] = []
    cond_data: list[int] = []

    for token_idx, token in enumerate(tokens):
        if is_rfl_super_token(token):
            memory_index_by_token[token_idx] = len(memory_index)
            memory_index.append(token_idx)
            mem_update_info[token_idx] = len(memory_index)
            remaining_memory.append(memory_index_by_token[token_idx])

        if is_rfl_bond_token(token):
            bond_index_by_token[token_idx] = len(bond_index)
            bond_index.append(token_idx)
            bond_update_info[token_idx] = len(bond_index) - 1

        connections = ring_branch_info[token_idx]
        if isinstance(connections, list) and connections:
            branch_index_by_token[token_idx] = len(branch_index_by_token)
            branch_update_info[token_idx] = branch_index_by_token[token_idx]
            for connected_token_idx in connections:
                if not isinstance(connected_token_idx, int):
                    continue
                bond_idx = bond_index_by_token.get(connected_token_idx)
                if bond_idx is not None:
                    branch_pairs.append([branch_index_by_token[token_idx], bond_idx])

        source_cond = cond_source[token_idx]
        cond_idx = memory_index_by_token.get(source_cond, -1) if isinstance(source_cond, int) else -1
        cond_data.append(cond_idx + 1)
        memory_used_indices.append(list(remaining_memory))

    branch_count = len(branch_index_by_token)
    bond_count = len(bond_index)
    branch_label = [[0 for _ in range(bond_count)] for _ in range(branch_count)]
    for branch_idx, bond_idx in branch_pairs:
        if 0 <= branch_idx < branch_count and 0 <= bond_idx < bond_count:
            branch_label[branch_idx][bond_idx] = 1

    token_ids = vocab.encode(tokens_without_eos, add_eos=True)
    return {
        "target": token_ids,
        "target_mask": [1 for _ in token_ids],
        "tokens": tokens,
        "cond_data": cond_data,
        "mem_index_data": memory_index,
        "mem_used_indices": memory_used_indices,
        "mem_update_info": mem_update_info,
        "branch_label": branch_label,
        "branch_update_info": branch_update_info,
        "bond_index_data": bond_index,
        "bond_update_info": bond_update_info,
        "branch_pairs": branch_pairs,
    }


class TextellerMsdDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        split_dir: Path,
        vocab: RflVocab,
        transform: Any,
        *,
        target_key: str = "targets.ssml_rfl",
        rfl_aux_field: str = "rfl",
        max_target_length: int = 1024,
        target_length_policy: str = "filter",
        max_samples: int | None = None,
    ) -> None:
        samples = load_split_samples(split_dir, target_key=target_key, rfl_aux_field=rfl_aux_field)
        if max_samples is not None:
            samples = samples[: max(0, int(max_samples))]
        self.samples = samples
        self.vocab = vocab
        self.transform = transform
        self.max_target_length = int(max_target_length)
        self.target_length_policy = target_length_policy.lower()
        self._filter_by_target_length()
        self.target_lengths = self._target_lengths()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        try:
            with Image.open(sample.image_path) as image:
                pixel_values = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {sample.image_path}: {exc}") from exc

        tokens = _tokens_from_aux(sample.rfl_aux, sample.target)
        msd = build_msd_item_features(tokens, sample.rfl_aux, self.vocab)
        return {
            "pixel_values": pixel_values,
            "target_text": sample.target,
            "image_path": str(sample.image_path),
            "image_name": sample.image_name,
            "metadata_targets": sample.targets,
            **msd,
        }

    def _target_lengths(self) -> list[int]:
        lengths: list[int] = []
        for sample in self.samples:
            tokens = _tokens_from_aux(sample.rfl_aux, sample.target)
            lengths.append(len(tokens) + 1)
        return lengths

    def _filter_by_target_length(self) -> None:
        lengths = self._target_lengths()
        too_long = [idx for idx, length in enumerate(lengths) if length > self.max_target_length]
        if not too_long:
            return
        if self.target_length_policy in {"error", "raise"}:
            max_len = max(lengths) if lengths else 0
            raise ValueError(
                f"{len(too_long)} RFL targets exceed max_target_length={self.max_target_length}; "
                f"max_len={max_len}."
            )
        if self.target_length_policy == "truncate":
            return
        if self.target_length_policy != "filter":
            raise ValueError(
                f"Unsupported target_length_policy={self.target_length_policy!r}; "
                "expected error, filter, or truncate."
            )
        drop = set(too_long)
        self.samples = [sample for idx, sample in enumerate(self.samples) if idx not in drop]


class TextellerMsdCollator:
    def __init__(self, vocab: RflVocab, *, include_metadata: bool = False) -> None:
        self.vocab = vocab
        self.include_metadata = include_metadata

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(features)
        max_len = max(max(1, len(feature["target"])) for feature in features)
        max_mem = max(1, max((len(feature["mem_index_data"]) for feature in features), default=0) + 1)
        max_branch = max(1, max((len(feature["branch_label"]) for feature in features), default=0))
        max_bond = max(
            1,
            max(
                (
                    len(feature["branch_label"][0])
                    if feature["branch_label"] and isinstance(feature["branch_label"][0], list)
                    else len(feature["bond_index_data"])
                )
                for feature in features
            ),
        )

        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        target = torch.full((batch_size, max_len), self.vocab.pad_id, dtype=torch.long)
        target_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
        cond_data = torch.zeros((batch_size, max_len), dtype=torch.long)
        mem_index_data = torch.full((batch_size, max_mem), -1, dtype=torch.long)
        mem_used_mask = torch.zeros((batch_size, max_len, max_mem), dtype=torch.float32)
        mem_used_mask[:, :, 0] = 1.0
        mem_update_info = torch.full((batch_size, max_len), -1, dtype=torch.long)
        branch_label = torch.zeros((batch_size, max_branch, max_bond), dtype=torch.long)
        branch_update_info = torch.full((batch_size, max_len), -1, dtype=torch.long)
        bond_index_data = torch.full((batch_size, max_bond), -1, dtype=torch.long)
        bond_update_info = torch.full((batch_size, max_len), -1, dtype=torch.long)

        for row_idx, feature in enumerate(features):
            ids = list(feature["target"])
            length = min(len(ids), max_len)
            target[row_idx, :length] = torch.tensor(ids[:length], dtype=torch.long)
            target_mask[row_idx, :length] = 1.0

            cond = list(feature["cond_data"])[:length]
            if cond:
                cond_data[row_idx, : len(cond)] = torch.tensor(cond, dtype=torch.long)

            memory_indices = list(feature["mem_index_data"])
            if memory_indices:
                mem_index_data[row_idx, 1 : len(memory_indices) + 1] = torch.tensor(
                    memory_indices,
                    dtype=torch.long,
                )

            for token_idx, used_indices in enumerate(feature["mem_used_indices"][:length]):
                for mem_idx in used_indices:
                    mem_pos = int(mem_idx) + 1
                    if 0 <= mem_pos < max_mem:
                        mem_used_mask[row_idx, token_idx, mem_pos] = 1.0

            updates = list(feature["mem_update_info"])[:length]
            if updates:
                mem_update_info[row_idx, : len(updates)] = torch.tensor(updates, dtype=torch.long)

            branches = list(feature["branch_update_info"])[:length]
            if branches:
                branch_update_info[row_idx, : len(branches)] = torch.tensor(branches, dtype=torch.long)

            bonds = list(feature["bond_index_data"])
            if bonds:
                bond_index_data[row_idx, : len(bonds)] = torch.tensor(bonds, dtype=torch.long)

            bond_updates = list(feature["bond_update_info"])[:length]
            if bond_updates:
                bond_update_info[row_idx, : len(bond_updates)] = torch.tensor(
                    bond_updates,
                    dtype=torch.long,
                )

            for branch_idx, row in enumerate(feature["branch_label"][:max_branch]):
                for bond_idx, value in enumerate(row[:max_bond]):
                    branch_label[row_idx, branch_idx, bond_idx] = int(value)

        batch: dict[str, Any] = {
            "pixel_values": pixel_values,
            "target": target,
            "target_mask": target_mask,
            "cond_data": cond_data,
            "mem_index_data": mem_index_data,
            "mem_used_mask": mem_used_mask,
            "mem_update_info": mem_update_info,
            "branch_label": branch_label,
            "branch_update_info": branch_update_info,
            "bond_index_data": bond_index_data,
            "bond_update_info": bond_update_info,
        }
        if self.include_metadata:
            batch["target_texts"] = [feature["target_text"] for feature in features]
            batch["image_paths"] = [feature["image_path"] for feature in features]
            batch["image_names"] = [feature["image_name"] for feature in features]
            batch["metadata_targets"] = [feature["metadata_targets"] for feature in features]
        return batch
