from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RflMsdLossOutput:
    loss: torch.Tensor
    sequence_loss: torch.Tensor
    branch_loss: torch.Tensor
    sequence_tokens: int
    branch_pairs: int


class RflMsdBranchClassifier(nn.Module):
    """Binary classifier for RFL-MSD branch/bond connection pairs."""

    def __init__(self, hidden_size: int, match_size: int | None = None) -> None:
        super().__init__()
        match_dim = int(match_size or hidden_size)
        self.branch_proj = nn.Linear(hidden_size, match_dim, bias=True)
        self.bond_proj = nn.Linear(hidden_size, match_dim, bias=False)
        self.classifier = nn.Linear(match_dim, 2, bias=True)

    def forward(self, branch_hidden: torch.Tensor, bond_hidden: torch.Tensor) -> torch.Tensor:
        if branch_hidden.ndim != 3:
            raise ValueError(f"Expected branch_hidden [B,L_branch,D], got {tuple(branch_hidden.shape)}")
        if bond_hidden.ndim != 3:
            raise ValueError(f"Expected bond_hidden [B,L_bond,D], got {tuple(bond_hidden.shape)}")
        branch = self.branch_proj(branch_hidden).unsqueeze(2)
        bond = self.bond_proj(bond_hidden).unsqueeze(1)
        return self.classifier(torch.tanh(branch + bond))


def _nested_config_value(config: Any, *names: str) -> Any:
    current = config
    for name in names:
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def infer_decoder_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    candidates = (
        _nested_config_value(config, "decoder", "hidden_size"),
        _nested_config_value(config, "decoder", "d_model"),
        _nested_config_value(config, "text_config", "hidden_size"),
        _nested_config_value(config, "text_config", "d_model"),
        _nested_config_value(config, "decoder_hidden_size"),
        _nested_config_value(config, "hidden_size"),
        _nested_config_value(config, "d_model"),
    )
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        "Could not infer decoder hidden size from model.config. Expected "
        "decoder.hidden_size/d_model, text_config.hidden_size/d_model, "
        "decoder_hidden_size, hidden_size, or d_model."
    )


def _output_value(outputs: Any, name: str) -> Any:
    if hasattr(outputs, name):
        return getattr(outputs, name)
    if isinstance(outputs, dict):
        return outputs.get(name)
    return None


def decoder_last_hidden_state(outputs: Any) -> torch.Tensor:
    hidden_states = _output_value(outputs, "decoder_hidden_states")
    if hidden_states is None:
        hidden_states = _output_value(outputs, "hidden_states")
    if not hidden_states:
        raise RuntimeError(
            "RFL-MSD branch inference requires decoder hidden states, but the "
            "model did not return them for output_hidden_states=True."
        )
    last_hidden = hidden_states[-1]
    if not isinstance(last_hidden, torch.Tensor) or last_hidden.ndim != 3:
        raise RuntimeError(
            "Expected decoder hidden state [B,L,D], "
            f"got {type(last_hidden).__name__} with shape {getattr(last_hidden, 'shape', None)}."
        )
    return last_hidden


def gather_token_states(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim != 2:
        raise ValueError(f"Expected positions [B,N], got {tuple(positions.shape)}")
    if positions.shape[1] == 0:
        return hidden.new_zeros((hidden.shape[0], 0, hidden.shape[-1]))
    valid = positions >= 0
    clamped = positions.clamp(min=0, max=max(0, hidden.shape[1] - 1))
    gathered = hidden.gather(1, clamped.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
    return gathered * valid.unsqueeze(-1).to(dtype=hidden.dtype)


def _as_batch_major_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected token logits with shape [B, L, V] or [L, B, V], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"Expected token labels with shape [B, L], got {tuple(labels.shape)}")
    if tuple(logits.shape[:2]) == tuple(labels.shape):
        return logits
    if logits.shape[0] == labels.shape[1] and logits.shape[1] == labels.shape[0]:
        return logits.transpose(0, 1)
    raise ValueError(
        "Token logits and labels have incompatible shapes: "
        f"logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
    )


def sequence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, int]:
    """Cross-entropy for the skeleton-generation part of RFL-MSD.

    RFL-MSD's public code trains token generation with cross entropy over the
    autoregressive RFL token sequence. This helper accepts either batch-major
    logits [B, L, V] or the original implementation's time-major [L, B, V].
    """
    logits = _as_batch_major_logits(logits, labels)
    labels = labels.long()
    if label_mask is not None:
        if tuple(label_mask.shape) != tuple(labels.shape):
            raise ValueError(
                "label_mask must match labels: "
                f"label_mask={tuple(label_mask.shape)} labels={tuple(labels.shape)}"
            )
        labels = labels.masked_fill(label_mask <= 0, ignore_index)
    valid = labels.ne(ignore_index)
    if valid.sum().item() == 0:
        return logits.new_zeros(()), 0
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss, int(valid.sum().item())


def branch_classification_loss(
    branch_logits: torch.Tensor,
    branch_labels: torch.Tensor,
    *,
    branch_mask: torch.Tensor | None = None,
    ignore_index: int = -1,
) -> tuple[torch.Tensor, int]:
    """Binary branch-connection loss used by RFL-MSD.

    Args:
        branch_logits: [B, L_branch, L_bond, 2] logits for disconnected/connected.
        branch_labels: [B, L_branch, L_bond] labels where 1 means connected.
        branch_mask: [B, L_branch, L_bond] active candidate-pair mask.
    """
    if branch_logits.ndim != 4 or branch_logits.shape[-1] != 2:
        raise ValueError(
            "Expected branch_logits with shape [B, L_branch, L_bond, 2], "
            f"got {tuple(branch_logits.shape)}"
        )
    expected_label_shape = tuple(branch_logits.shape[:3])
    if tuple(branch_labels.shape) != expected_label_shape:
        raise ValueError(
            "branch_labels must match branch_logits without class dim: "
            f"branch_labels={tuple(branch_labels.shape)} logits={tuple(branch_logits.shape)}"
        )
    labels = branch_labels.long()
    if branch_mask is not None:
        if tuple(branch_mask.shape) != expected_label_shape:
            raise ValueError(
                "branch_mask must match branch_labels: "
                f"branch_mask={tuple(branch_mask.shape)} labels={tuple(branch_labels.shape)}"
            )
        labels = labels.masked_fill(branch_mask <= 0, ignore_index)
        active_pairs = int((branch_mask > 0).sum().item())
    else:
        active_pairs = int(labels.ne(ignore_index).sum().item())
    if active_pairs == 0:
        return branch_logits.new_zeros(()), 0
    loss = F.cross_entropy(
        branch_logits.reshape(-1, 2),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss, active_pairs


class RflMsdLoss(nn.Module):
    """RFL-MSD objective: O = lambda_seq * Lce + lambda_branch * Lcls."""

    def __init__(
        self,
        *,
        lambda_sequence: float = 1.0,
        lambda_branch: float = 1.0,
        label_ignore_index: int = -100,
        branch_ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.lambda_sequence = float(lambda_sequence)
        self.lambda_branch = float(lambda_branch)
        self.label_ignore_index = int(label_ignore_index)
        self.branch_ignore_index = int(branch_ignore_index)

    def forward(
        self,
        token_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        label_mask: torch.Tensor | None = None,
        branch_logits: torch.Tensor | None = None,
        branch_labels: torch.Tensor | None = None,
        branch_mask: torch.Tensor | None = None,
    ) -> RflMsdLossOutput:
        seq_loss, sequence_tokens = sequence_cross_entropy(
            token_logits,
            labels,
            label_mask=label_mask,
            ignore_index=self.label_ignore_index,
        )
        if branch_logits is None or branch_labels is None:
            raise ValueError(
                "RFL-MSD loss requires branch_logits and branch_labels from an MSD decoder. "
                "The current TexTeller sequence decoder only provides token logits."
            )
        cls_loss, branch_pairs = branch_classification_loss(
            branch_logits,
            branch_labels,
            branch_mask=branch_mask,
            ignore_index=self.branch_ignore_index,
        )
        total = self.lambda_sequence * seq_loss + self.lambda_branch * cls_loss
        return RflMsdLossOutput(
            loss=total,
            sequence_loss=seq_loss,
            branch_loss=cls_loss,
            sequence_tokens=sequence_tokens,
            branch_pairs=branch_pairs,
        )
