from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel

from .rfl_vocab import RflVocab
from .utils import save_json


@dataclass(frozen=True)
class MsdDecoderConfig:
    encoder_hidden_size: int = 768
    encoder_dim: int = 384
    decoder_state_dim: int = 256
    decoder_embed_dim: int = 128
    decoder_att_dim: int = 128
    decoder_merge_dim: int = 384
    decoder_chatt_dim: int = 384
    decoder_cover_kernel: tuple[int, int] = (11, 11)
    decoder_cover_padding: tuple[int, int] = (5, 5)
    decoder_dropout: float = 0.2
    decoder_embed_drop: float = 0.15
    decoder_mem_match_dim: int = 256
    lambda_sequence: float = 1.0
    lambda_memory: float = 1.0


@dataclass(frozen=True)
class MsdPrediction:
    tokens: list[str]
    score: float
    branch_pairs: list[tuple[int, int]]
    cond_data: list[int]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


class ImageAttention(nn.Module):
    def __init__(self, cfg: MsdDecoderConfig) -> None:
        super().__init__()
        self.state_dim = cfg.decoder_state_dim
        self.encoder_dim = cfg.encoder_dim
        self.att_dim = cfg.decoder_att_dim
        self.mem_dim = cfg.decoder_embed_dim + cfg.decoder_state_dim + cfg.encoder_dim
        self.energy = nn.Conv2d(self.att_dim, 1, 1, bias=False)
        self.weight_trans = nn.Conv2d(
            1,
            self.att_dim,
            kernel_size=cfg.decoder_cover_kernel,
            padding=cfg.decoder_cover_padding,
            bias=False,
        )
        self.cum_weight_trans = nn.Conv2d(
            1,
            self.att_dim,
            kernel_size=cfg.decoder_cover_kernel,
            padding=cfg.decoder_cover_padding,
            bias=False,
        )
        self.cond_weight_trans = nn.Conv2d(
            2,
            2 * self.att_dim,
            kernel_size=cfg.decoder_cover_kernel,
            padding=cfg.decoder_cover_padding,
            bias=False,
            groups=2,
        )
        self.state_trans = nn.Linear(self.state_dim, self.att_dim, bias=False)
        self.context_trans = nn.Linear(self.encoder_dim, self.att_dim, bias=False)
        self.cond_trans = nn.Linear(self.mem_dim, self.att_dim, bias=False)

    def forward(
        self,
        encode: torch.Tensor,
        encode_pro: torch.Tensor,
        encode_mask: torch.Tensor,
        state: torch.Tensor,
        pre_weight: torch.Tensor,
        cum_weight: torch.Tensor,
        context: torch.Tensor,
        cond_mem: torch.Tensor,
        cond_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = self.weight_trans(pre_weight)
        cum = self.cum_weight_trans(cum_weight)
        cond = self.cond_weight_trans(cond_weight)
        bsz, _, height, width = cond.shape
        cond = cond.view(bsz, 2, self.att_dim, height, width).sum(1)
        state_ctx = (
            self.state_trans(state)
            + self.context_trans(context)
            + self.cond_trans(cond_mem)
        ).view(bsz, self.att_dim, 1, 1)
        energies = self.energy(torch.tanh(state_ctx + pre + cum + cond + encode_pro))
        energies = energies + (encode_mask - 1) * 1e8
        att_weights = F.softmax(energies.flatten(1), dim=1)
        new_weight = att_weights.view(bsz, 1, encode.shape[2], encode.shape[3])
        new_cum_weight = cum_weight + new_weight
        ctx = torch.bmm(att_weights.unsqueeze(1), encode.flatten(2).transpose(1, 2)).squeeze(1)
        return ctx, new_weight, new_cum_weight


class MyGRUTransition(nn.Module):
    def __init__(self, state_dim: int, input_dim: int) -> None:
        super().__init__()
        self.input_to_state = nn.Linear(input_dim, state_dim)
        self.input_to_gate = nn.Linear(input_dim, state_dim * 2)
        self.state_to_gate_h2h = nn.Linear(state_dim, state_dim * 2, bias=False)
        self.h2h = nn.Linear(state_dim, state_dim, bias=False)

    def forward(self, data: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        gate = self.input_to_gate(data) + self.state_to_gate_h2h(state)
        update_gate, reset_gate = gate.chunk(2, dim=1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)
        state_hat = torch.tanh(self.input_to_state(data) + self.h2h(state * reset_gate))
        return update_gate * state_hat + (1 - update_gate) * state


class MlpHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_mlp_l1 = nn.Linear(input_dim, hidden_dim)
        self.linear_mlp_l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_mlp_l2(self.dropout(F.relu(self.linear_mlp_l1(x))))


class MemoryClsHead(nn.Module):
    def __init__(self, input_dim: int, match_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, match_dim)
        self.mem_proj = nn.Linear(input_dim, match_dim, bias=False)
        self.energy_proj = nn.Linear(match_dim, 2)

    def forward(self, selected_branchs: torch.Tensor, selected_bonds: torch.Tensor) -> torch.Tensor:
        branch = self.input_proj(selected_branchs).unsqueeze(2)
        bond = self.mem_proj(selected_bonds).unsqueeze(1)
        return self.energy_proj(torch.tanh(branch + bond))


class Readout(nn.Module):
    def __init__(self, vocab_size: int, cfg: MsdDecoderConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.decoder_embed_dim)
        self.embed_dropout = nn.Dropout(cfg.decoder_embed_drop)
        self.token_head = MlpHead(
            cfg.decoder_embed_dim + cfg.decoder_state_dim + cfg.encoder_dim,
            cfg.decoder_merge_dim,
            vocab_size,
            cfg.decoder_dropout,
        )

    def step(self, output: torch.Tensor, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.token_head(torch.cat([output, state, context], dim=1))

    def get_embed(self, label_id: torch.Tensor) -> torch.Tensor:
        return self.embed_dropout(self.embed(label_id.long()))

    def forward(
        self,
        output: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
        label_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(output, state, context), self.get_embed(label_id)


class PreDecoder(nn.Module):
    def __init__(self, cfg: MsdDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mem_dim = cfg.decoder_embed_dim + cfg.decoder_state_dim + cfg.encoder_dim
        self.attention_conv = nn.Conv2d(cfg.encoder_dim, cfg.decoder_att_dim, 1, bias=False)
        self.transition_gru_state_init = nn.Linear(cfg.encoder_dim, cfg.decoder_state_dim)

    def enc_init_states(self, encode: torch.Tensor, encode_mask: torch.Tensor) -> list[torch.Tensor]:
        masked = encode * encode_mask
        pooled = masked.sum(dim=[2, 3]) / encode_mask.sum(dim=[2, 3]).clamp_min(1.0)
        return [self.transition_gru_state_init(pooled), pooled]

    def zero_init_states(
        self,
        encode: torch.Tensor,
        mem_index_data: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        batch_size, _, height, width = encode.shape
        memory_length = 1 if mem_index_data is None else int(mem_index_data.shape[1])
        device = encode.device
        dtype = encode.dtype
        weight = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
        cum_weight = torch.zeros_like(weight)
        output = torch.zeros(batch_size, self.cfg.decoder_embed_dim, device=device, dtype=dtype)
        memory = torch.zeros(batch_size, memory_length, self.mem_dim, device=device, dtype=dtype)
        memory_mask = torch.zeros(batch_size, memory_length, device=device, dtype=dtype)
        memory_weight = torch.zeros(batch_size, memory_length, 2, height, width, device=device, dtype=dtype)
        memory_mask[:, 0] = 1
        return [weight, cum_weight, output, memory, memory_mask, memory_weight]

    def forward(
        self,
        encode: torch.Tensor,
        encode_mask: torch.Tensor,
        mem_index_data: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        return (
            self.attention_conv(encode),
            self.enc_init_states(encode, encode_mask),
            self.zero_init_states(encode, mem_index_data),
        )


class SequenceGenerator(nn.Module):
    def __init__(self, vocab: RflVocab, cfg: MsdDecoderConfig) -> None:
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.vocab_size = vocab.vocab_size
        self.mem_dim = cfg.decoder_embed_dim + cfg.decoder_state_dim + cfg.encoder_dim
        self.query_dim = self.mem_dim
        self.attention = ImageAttention(cfg)
        self.transition = MyGRUTransition(
            cfg.decoder_state_dim,
            cfg.decoder_embed_dim + cfg.encoder_dim + self.mem_dim,
        )
        self.readout = Readout(self.vocab_size, cfg)
        self.mem_cls = MemoryClsHead(self.query_dim, cfg.decoder_mem_match_dim)
        self.ea_idx = vocab.get_id("<ea>")
        self.connbranch_idx = vocab.get_id(r"\connbranch")
        self.bond_ids = tuple(
            token_id
            for token_id, word in vocab.id2word.items()
            if ("[:" in word and word.endswith("]"))
            or (word.startswith("?[") and word.endswith("]") and "," in word)
        )
        self.super_ids = tuple(
            token_id
            for token_id, word in vocab.id2word.items()
            if "@" in word or r"\Superatom" in word
        )

    def step(
        self,
        encode: torch.Tensor,
        encode_pro: torch.Tensor,
        encode_mask: torch.Tensor,
        states: list[torch.Tensor],
        label: torch.Tensor,
        cond_mem: torch.Tensor,
        next_cond: torch.Tensor,
        time_t: int,
        mem_index_data: torch.Tensor,
        cur_mem_update_info: torch.Tensor,
        cond_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        weight, cum_weight, output, memory, memory_mask, memory_weight, state, context = states
        cur_context, cur_weight, cur_cum_weight = self.attention(
            encode,
            encode_pro,
            encode_mask,
            state,
            weight,
            cum_weight,
            context,
            cond_mem,
            cond_weight,
        )
        cur_energy, cur_output = self.readout(output, state, cur_context, label)
        tmp_states = [cur_weight, cur_cum_weight, cur_output, memory, memory_mask, memory_weight, state, cur_context]
        cur_memory, cur_memory_mask, cur_memory_weight = self.update_memory_for_train(
            time_t,
            mem_index_data,
            tmp_states,
            cur_mem_update_info,
        )
        batch_size = encode.shape[0]
        next_cond_mem = torch.gather(
            cur_memory.detach(),
            1,
            next_cond.view(batch_size, 1, 1).repeat(1, 1, self.mem_dim).long(),
        ).squeeze(1)
        cur_state = self.transition(torch.cat([cur_output, cur_context, next_cond_mem], dim=1), state)
        return cur_energy, [
            cur_weight,
            cur_cum_weight,
            cur_output,
            cur_memory,
            cur_memory_mask,
            cur_memory_weight,
            cur_state,
            cur_context,
        ]

    def update_memory_for_train(
        self,
        time_t: int,
        mem_index_data: torch.Tensor,
        states: list[torch.Tensor],
        cur_mem_update_info: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del time_t, mem_index_data
        cur_weight, cur_cum_weight, cur_output, cur_memory, cur_memory_mask, cur_memory_weight, cur_state, cur_ctx = states
        update_mask = (cur_mem_update_info != -1).detach()
        if update_mask.sum() <= 0:
            return cur_memory, cur_memory_mask, cur_memory_weight
        batch_size = cur_mem_update_info.shape[0]
        mem_dim = cur_memory.shape[-1]
        _, _, _, height, width = cur_memory_weight.shape
        new_feature = torch.cat([cur_output, cur_state, cur_ctx], dim=1).unsqueeze(1)
        new_feature = new_feature * update_mask.view(batch_size, 1, 1)
        new_weight = torch.cat([cur_weight, cur_cum_weight], dim=1).unsqueeze(1)
        new_weight = new_weight * update_mask.view(batch_size, 1, 1, 1, 1)
        index = (cur_mem_update_info * update_mask).long().detach()
        memory = torch.scatter_add(
            cur_memory,
            1,
            index.view(batch_size, 1, 1).repeat(1, 1, mem_dim),
            new_feature,
        )
        memory_mask = torch.scatter_add(
            cur_memory_mask,
            1,
            index.view(batch_size, 1),
            update_mask.float().view(batch_size, 1).to(cur_memory_mask.dtype),
        )
        memory_weight = torch.scatter_add(
            cur_memory_weight,
            1,
            index.view(batch_size, 1, 1, 1, 1).repeat(1, 1, 2, height, width),
            new_weight,
        )
        return memory, memory_mask, memory_weight

    def sequence_loss(
        self,
        readout: torch.Tensor,
        label: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = readout.permute(1, 0, 2).reshape(-1, readout.shape[-1])
        targets = label.reshape(-1).long()
        mask = label_mask.reshape(-1) > 0
        if mask.sum() <= 0:
            return logits.sum() * 0
        targets = targets.masked_fill(~mask, -1)
        return F.cross_entropy(logits, targets, ignore_index=-1)

    def memory_loss(
        self,
        mem_cls_logits: torch.Tensor,
        branch_target: torch.Tensor,
        mem_cls_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mem_cls_mask > 0
        if mask.sum() <= 0:
            return mem_cls_logits.sum() * 0
        targets = branch_target.long().masked_fill(~mask, -1)
        return F.cross_entropy(mem_cls_logits.reshape(-1, 2), targets.reshape(-1), ignore_index=-1)

    def forward(
        self,
        encode: torch.Tensor,
        encode_mask: torch.Tensor,
        encode_pro: torch.Tensor,
        label: torch.Tensor,
        label_mask: torch.Tensor,
        target_branch: torch.Tensor,
        cond_data: torch.Tensor,
        mem_index_data: torch.Tensor,
        bond_index_data: torch.Tensor,
        mem_used_mask: torch.Tensor,
        mem_update_info: torch.Tensor,
        branch_update_info: torch.Tensor,
        bond_update_info: torch.Tensor,
        init_states: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size, tgt_len = label.shape
        states = init_states
        label = label * label_mask.long()
        cond_data = cond_data * label_mask.long()
        mem_used_mask = mem_used_mask * label_mask.unsqueeze(-1)
        mem_used_mask[:, :, 0] = 1
        outputs_energy: list[torch.Tensor] = []

        max_branch_len = target_branch.shape[1]
        max_bond_len = bond_index_data.shape[1]
        selected_branchs = torch.zeros(batch_size, max_branch_len, self.query_dim, device=encode.device, dtype=encode.dtype)
        selected_branchs_mask = torch.zeros(batch_size, max_branch_len, device=encode.device, dtype=encode.dtype)
        selected_bonds = torch.zeros(batch_size, max_bond_len, self.query_dim, device=encode.device, dtype=encode.dtype)
        selected_bonds_mask = torch.zeros(batch_size, max_bond_len, device=encode.device, dtype=encode.dtype)
        default_next_cond = torch.zeros(batch_size, device=encode.device, dtype=torch.long)

        for time_idx in range(tgt_len):
            cur_memory = states[3]
            cur_memory_weight = states[5]
            old_state = states[6]
            _, _, _, height, width = cur_memory_weight.shape
            cur_cond = cond_data[:, time_idx].long()
            cond_mem = torch.gather(
                cur_memory.detach(),
                1,
                cur_cond.view(batch_size, 1, 1).repeat(1, 1, self.mem_dim),
            ).squeeze(1)
            cond_weight = torch.gather(
                cur_memory_weight.detach(),
                1,
                cur_cond.view(batch_size, 1, 1, 1, 1).repeat(1, 1, 2, height, width),
            ).squeeze(1)
            next_cond = cond_data[:, time_idx + 1].long() if time_idx < tgt_len - 1 else default_next_cond
            cur_energy, states = self.step(
                encode,
                encode_pro,
                encode_mask,
                states,
                label[:, time_idx],
                cond_mem,
                next_cond,
                time_idx,
                mem_index_data,
                mem_update_info[:, time_idx],
                cond_weight,
            )
            bond_update = bond_update_info[:, time_idx]
            bond_mask = (bond_update != -1).detach()
            if bond_mask.sum() > 0:
                new_feature = torch.cat([states[2], old_state, states[7]], dim=1)
                new_feature = (new_feature * bond_mask.unsqueeze(1)).unsqueeze(1)
                index = (bond_update * bond_mask).long().detach()
                selected_bonds.scatter_add_(
                    1,
                    index.view(batch_size, 1, 1).repeat(1, 1, self.query_dim),
                    new_feature,
                )
                selected_bonds_mask.scatter_add_(
                    1,
                    index.view(batch_size, 1),
                    bond_mask.float().view(batch_size, 1).to(selected_bonds_mask.dtype),
                )
            branch_update = branch_update_info[:, time_idx]
            branch_mask = (branch_update != -1).detach()
            if branch_mask.sum() > 0:
                new_feature = torch.cat([states[2], old_state, states[7]], dim=1)
                new_feature = (new_feature * branch_mask.unsqueeze(1)).unsqueeze(1)
                index = (branch_update * branch_mask).long().detach()
                selected_branchs.scatter_add_(
                    1,
                    index.view(batch_size, 1, 1).repeat(1, 1, self.query_dim),
                    new_feature,
                )
                selected_branchs_mask.scatter_add_(
                    1,
                    index.view(batch_size, 1),
                    branch_mask.float().view(batch_size, 1).to(selected_branchs_mask.dtype),
                )
            outputs_energy.append(cur_energy)

        logits = torch.stack(outputs_energy, dim=0)
        mem_cls_logits = self.mem_cls(selected_branchs, selected_bonds)
        mem_cls_mask = selected_branchs_mask.unsqueeze(2) * selected_bonds_mask.unsqueeze(1)
        mem_cls_logits = mem_cls_logits * mem_cls_mask.unsqueeze(-1)
        seq_loss = self.sequence_loss(logits, label, label_mask)
        mem_loss = self.memory_loss(mem_cls_logits, target_branch, mem_cls_mask)
        return {
            "loss": self.cfg.lambda_sequence * seq_loss + self.cfg.lambda_memory * mem_loss,
            "sequence_loss": seq_loss.detach(),
            "memory_loss": mem_loss.detach(),
            "logits": logits,
            "memory_logits": mem_cls_logits,
            "memory_mask": mem_cls_mask,
        }

    def eval_step_pred(
        self,
        encode: torch.Tensor,
        encode_pro: torch.Tensor,
        encode_mask: torch.Tensor,
        states: list[torch.Tensor],
        decode_states: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        weight, cum_weight, output, memory, memory_mask, memory_weight, state, context = states
        encode_batch = encode.shape[0]
        decode_batch = state.shape[0]
        if decode_batch > encode_batch:
            repeat = decode_batch // encode_batch
            encode = encode.unsqueeze(1).repeat(1, repeat, 1, 1, 1).view(decode_batch, *encode.shape[1:])
            encode_pro = encode_pro.unsqueeze(1).repeat(1, repeat, 1, 1, 1).view(decode_batch, *encode_pro.shape[1:])
            encode_mask = encode_mask.unsqueeze(1).repeat(1, repeat, 1, 1, 1).view(decode_batch, *encode_mask.shape[1:])
        cur_context, cur_weight, cur_cum_weight = self.attention(
            encode,
            encode_pro,
            encode_mask,
            state,
            weight,
            cum_weight,
            context,
            decode_states["cond_mem"],
            decode_states["cond_weight"],
        )
        logits = self.readout.step(output, state, cur_context)
        return F.log_softmax(logits, dim=1), [cur_weight, cur_cum_weight, output, memory, memory_mask, memory_weight, state, cur_context], decode_states

    @torch.no_grad()
    def eval_step_update_memory(
        self,
        all_outputs: torch.Tensor,
        states: list[torch.Tensor],
        decode_states: dict[str, torch.Tensor],
        *,
        time_t: int,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        cur_weight, cur_cum_weight, _, memory, memory_mask, memory_weight, state, cur_context = states
        label = all_outputs[-1].long()
        batch_beam = label.shape[0]
        cur_output = self.readout.get_embed(label)
        device = label.device

        super_ids = torch.tensor(self.super_ids or (-999999,), device=device).view(1, -1)
        super_hit = (label.view(-1, 1) == super_ids).sum(1)
        cur_len = memory_mask.sum(dim=1)
        pad = int((cur_len + super_hit - memory_mask.shape[1]).max().clamp_min(0).item())
        cur_memory = F.pad(memory, (0, 0, 0, pad))
        cur_memory_weight = F.pad(memory_weight, (0, 0, 0, 0, 0, 0, 0, pad))
        cur_memory_mask = F.pad(memory_mask, (0, pad))
        cur_memory_used_mask = F.pad(decode_states["memory_used_mask"], (0, pad))
        cur_mem_index = F.pad(decode_states["mem_index"], (0, pad), value=-1)

        super_rows = super_hit.nonzero().long().squeeze(1)
        if super_rows.numel() > 0:
            delta_memory = torch.cat([cur_output[super_rows], state[super_rows], cur_context[super_rows]], dim=1)
            delta_weight = torch.cat([cur_weight[super_rows], cur_cum_weight[super_rows]], dim=1)
            row_ids = torch.arange(batch_beam, device=device).unsqueeze(1)
            cmp_res = row_ids == super_rows
            offset_idx = (cmp_res.cumsum(dim=1) * cmp_res).sum(0)
            offset = (offset_idx + cur_len[super_rows] - 1).long()
            for item_id, (row, col) in enumerate(zip(super_rows.tolist(), offset.tolist())):
                cur_memory[row, col, :] = delta_memory[item_id]
                cur_memory_weight[row, col, :] = delta_weight[item_id]
                cur_memory_mask[row, col] = 1
                cur_memory_used_mask[row, col] = 1
                cur_mem_index[row, col] = time_t

        bond_ids = torch.tensor(self.bond_ids or (-999999,), device=device).view(1, -1)
        bond_hit = (label.view(-1, 1) == bond_ids).sum(1)
        bonds_mask = decode_states["bonds_mask"]
        selected_bonds = decode_states["selected_bonds"]
        cur_bond_len = bonds_mask.sum(dim=1)
        bond_pad = int((cur_bond_len + bond_hit - bonds_mask.shape[1]).max().clamp_min(0).item())
        cur_bonds = F.pad(selected_bonds, (0, 0, 0, bond_pad))
        cur_bonds_mask = F.pad(bonds_mask, (0, bond_pad))
        cur_bond_index = F.pad(decode_states["bond_index"], (0, bond_pad), value=-1)
        bond_rows = bond_hit.nonzero().long().squeeze(1)
        if bond_rows.numel() > 0:
            delta_bond = torch.cat([cur_output[bond_rows], state[bond_rows], cur_context[bond_rows]], dim=1)
            row_ids = torch.arange(batch_beam, device=device).unsqueeze(1)
            cmp_res = row_ids == bond_rows
            offset_idx = (cmp_res.cumsum(dim=1) * cmp_res).sum(0)
            offset = (offset_idx + cur_bond_len[bond_rows] - 1).long()
            for item_id, (row, col) in enumerate(zip(bond_rows.tolist(), offset.tolist())):
                cur_bonds[row, col, :] = delta_bond[item_id]
                cur_bonds_mask[row, col] = 1
                cur_bond_index[row, col] = time_t

        branch_hit = (label == self.connbranch_idx).long()
        branchs_mask = decode_states["branchs_mask"]
        cur_branch_len = branchs_mask.sum(dim=1)
        branch_pad = int((cur_branch_len + branch_hit - branchs_mask.shape[1]).max().clamp_min(0).item())
        cur_branchs_mask = F.pad(branchs_mask, (0, branch_pad))
        cur_branch_index = F.pad(decode_states["branch_index"], (0, branch_pad), value=-1)
        branch_rows = branch_hit.nonzero().long().squeeze(1)
        if branch_rows.numel() > 0:
            row_ids = torch.arange(batch_beam, device=device).unsqueeze(1)
            cmp_res = row_ids == branch_rows
            offset_idx = (cmp_res.cumsum(dim=1) * cmp_res).sum(0)
            offset = (offset_idx + cur_branch_len[branch_rows] - 1).long()
            for row, col in zip(branch_rows.tolist(), offset.tolist()):
                cur_branchs_mask[row, col] = 1
                cur_branch_index[row, col] = max(cur_bond_index[row])

        new_states = [cur_weight, cur_cum_weight, cur_output, cur_memory, cur_memory_mask, cur_memory_weight, state, cur_context]
        decode_states["memory_used_mask"] = cur_memory_used_mask
        decode_states["mem_index"] = cur_mem_index
        decode_states["bond_index"] = cur_bond_index
        decode_states["selected_bonds"] = cur_bonds
        decode_states["bonds_mask"] = cur_bonds_mask
        decode_states["branch_index"] = cur_branch_index
        decode_states["branchs_mask"] = cur_branchs_mask

        branch_mask = label == self.connbranch_idx
        prev_branch_mask = all_outputs[-2].long() == self.connbranch_idx
        multi_branch = prev_branch_mask & branch_mask
        mem_cls_res = -torch.ones((batch_beam, 2), device=device)
        if branch_mask.sum() > 0:
            last_bonds = decode_states["bonds_mask"].sum(dim=1) - 1
            selected_branch = decode_states["selected_bonds"][
                torch.arange(batch_beam, device=device),
                last_bonds.long().clamp_min(0),
            ].unsqueeze(1)
            selected_bonds_for_cls = decode_states["selected_bonds"][:, 1:]
            bonds_mask_for_cls = decode_states["bonds_mask"][:, 1:]
            cls_mask = branch_mask.float().view(batch_beam, 1, 1) * bonds_mask_for_cls.unsqueeze(1)
            cls_logits = self.mem_cls(selected_branch, selected_bonds_for_cls) * cls_mask.unsqueeze(-1)
            cls_prob = F.softmax(cls_logits, dim=3)
            for row in branch_mask.nonzero().long().squeeze(1).tolist():
                last_bond = int(last_bonds[row].item())
                if last_bond <= 0:
                    continue
                probs = cls_prob[row, 0, :last_bond, 1]
                if probs.numel() == 0:
                    continue
                if multi_branch[row]:
                    suffix = all_outputs[:, row].long().tolist()[::-1]
                    rank = 0
                    for token_id in suffix:
                        if token_id == self.connbranch_idx:
                            rank += 1
                        else:
                            break
                    rank = min(max(rank, 1), probs.numel())
                    bond_pos = torch.topk(probs, k=rank).indices[-1]
                else:
                    bond_pos = probs.argmax(dim=0)
                bond_index = decode_states["bond_index"][row, 1:][bond_pos]
                branch_index = max(decode_states["branch_index"][row])
                if bond_index != -1 and branch_index != -1:
                    mem_cls_res[row, 0] = branch_index
                    mem_cls_res[row, 1] = bond_index
        decode_states["mem_cls_res"] = mem_cls_res
        return new_states, decode_states

    def eval_step_update(
        self,
        states: list[torch.Tensor],
        decode_states: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        cur_weight, cur_cum_weight, cur_output, cur_memory, cur_memory_mask, cur_memory_weight, state, cur_context = states
        cur_state = self.transition(torch.cat([cur_output, cur_context, decode_states["cond_mem"]], dim=1), state)
        return [cur_weight, cur_cum_weight, cur_output, cur_memory, cur_memory_mask, cur_memory_weight, cur_state, cur_context], decode_states

    @torch.no_grad()
    def expand_state_dim(self, state: torch.Tensor, num: int) -> torch.Tensor:
        old_shape = list(state.shape)
        repeated = state.unsqueeze(1).repeat(1, int(num), *([1] * (state.dim() - 1)))
        old_shape[0] *= int(num)
        return repeated.view(*old_shape)

    @torch.no_grad()
    def find_cand(
        self,
        outputs: torch.Tensor,
        masks: torch.Tensor,
        mem_index: torch.Tensor,
        mem_used_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length, batch = outputs.shape
        del length
        cur_cand = torch.zeros((batch, 1), device=outputs.device, dtype=mem_index.dtype)
        cur_cand_mask = torch.zeros((batch, 1), device=outputs.device, dtype=masks.dtype)
        for row in range(batch):
            if masks[-1, row] == 0:
                continue
            token_ids = [int(item) for item in outputs[1:, row].detach().cpu().tolist()]
            words = [self.vocab.get_word(token_id) for token_id in token_ids]
            cur_idx = len(words) - 1
            if cur_idx < 0 or words[cur_idx] != "<ea>":
                continue
            cur_idx -= 1
            super_memory: dict[str, int] = {}
            for token_idx in range(cur_idx, 0, -1):
                word = words[token_idx]
                if word == r"\chemfig":
                    break
                if "@" not in word and r"\Superatom" not in word:
                    continue
                if "@" in word and word in super_memory:
                    super_memory.pop(word)
                    continue
                mids = (mem_index[row] == token_idx).nonzero().flatten().long().cpu().tolist()
                if len(mids) != 1:
                    break
                mid = mids[0]
                super_memory[word] = mid
            cand_count = 0
            for mid in super_memory.values():
                if not bool(mem_used_mask[row, mid]):
                    continue
                capacity = cur_cand.shape[1]
                if cand_count + 1 > capacity:
                    delta = cand_count + 1 - capacity
                    cur_cand = F.pad(cur_cand, (0, delta))
                    cur_cand_mask = F.pad(cur_cand_mask, (0, delta))
                cur_cand[row, cand_count] = mid
                cur_cand_mask[row, cand_count] = 1
                cand_count += 1
        return cur_cand, cur_cand_mask


class TexTellerMsdModel(nn.Module):
    def __init__(
        self,
        base_model: VisionEncoderDecoderModel,
        vocab: RflVocab,
        cfg: MsdDecoderConfig | None = None,
        *,
        base_model_name_or_path: str = "OleehyO/TexTeller",
    ) -> None:
        super().__init__()
        self.base_model_name_or_path = base_model_name_or_path
        self.vocab = vocab
        self.cfg = cfg or MsdDecoderConfig()
        self.encoder = base_model.encoder
        self.encoder_norm = nn.LayerNorm(self.cfg.encoder_hidden_size)
        self.encoder_adapter = nn.Conv2d(self.cfg.encoder_hidden_size, self.cfg.encoder_dim, 1)
        self.pre_decoder = PreDecoder(self.cfg)
        self.decoder = SequenceGenerator(vocab, self.cfg)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        vocab: RflVocab,
        cfg: MsdDecoderConfig | None = None,
        *,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
    ) -> "TexTellerMsdModel":
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        base = VisionEncoderDecoderModel.from_pretrained(model_name_or_path, **kwargs)
        return cls(base, vocab, cfg, base_model_name_or_path=model_name_or_path)

    def freeze_encoder(self, frozen: bool = True) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not frozen

    def enable_encoder_gradient_checkpointing(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def encode(
        self,
        pixel_values: torch.Tensor,
        mem_index_data: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
        hidden = encoder_outputs.last_hidden_state
        if hidden.shape[-1] != self.cfg.encoder_hidden_size:
            raise RuntimeError(
                f"TexTeller encoder hidden size {hidden.shape[-1]} does not match "
                f"MSD config encoder_hidden_size={self.cfg.encoder_hidden_size}."
            )
        seq_len = hidden.shape[1]
        if int(math.isqrt(seq_len - 1)) ** 2 == seq_len - 1:
            hidden = hidden[:, 1:, :]
            seq_len -= 1
        side = int(math.isqrt(seq_len))
        if side * side != seq_len:
            raise RuntimeError(f"TexTeller encoder token count is not a square grid: {seq_len}")
        hidden = self.encoder_norm(hidden)
        features = hidden.transpose(1, 2).contiguous().view(hidden.shape[0], hidden.shape[2], side, side)
        encoded = self.encoder_adapter(features)
        encoded_mask = torch.ones(
            encoded.shape[0],
            1,
            encoded.shape[2],
            encoded.shape[3],
            device=encoded.device,
            dtype=encoded.dtype,
        )
        encoded_proj, enc_init_states, zero_init_states = self.pre_decoder(encoded, encoded_mask, mem_index_data)
        return encoded, encoded_mask, encoded_proj, zero_init_states + enc_init_states

    def forward(
        self,
        pixel_values: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        cond_data: torch.Tensor,
        mem_index_data: torch.Tensor,
        mem_used_mask: torch.Tensor,
        mem_update_info: torch.Tensor,
        branch_label: torch.Tensor,
        branch_update_info: torch.Tensor,
        bond_index_data: torch.Tensor,
        bond_update_info: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded, encoded_mask, encoded_proj, init_states = self.encode(pixel_values, mem_index_data)
        return self.decoder(
            encoded,
            encoded_mask,
            encoded_proj,
            target,
            target_mask,
            branch_label,
            cond_data,
            mem_index_data,
            bond_index_data,
            mem_used_mask,
            mem_update_info,
            branch_update_info,
            bond_update_info,
            init_states,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        *,
        num_beams: int = 5,
        max_new_tokens: int = 1024,
    ) -> list[list[MsdPrediction]]:
        if num_beams < 1:
            raise ValueError("num_beams must be >= 1")
        encoded, encoded_mask, encoded_proj, init_states = self.encode(pixel_values)
        batch_size = encoded.shape[0]
        beam_size = int(num_beams)
        beam_batch = batch_size * beam_size
        device = encoded.device
        dtype = encoded.dtype
        height, width = encoded.shape[2], encoded.shape[3]

        states = []
        for state in init_states:
            shape = [beam_batch, *state.shape[1:]]
            repeats = [1, beam_size, *([1] * (state.dim() - 1))]
            states.append(state.unsqueeze(1).repeat(*repeats).view(*shape))

        query_dim = states[3].shape[2]
        decode_states: dict[str, torch.Tensor] = {
            "memory_used_mask": torch.zeros(beam_batch, states[3].shape[1], device=device, dtype=dtype),
            "mem_index": -torch.ones(beam_batch, states[3].shape[1], device=device, dtype=dtype),
            "cond_mem": torch.zeros(beam_batch, query_dim, device=device, dtype=dtype),
            "cond_weight": torch.zeros(beam_batch, 2, height, width, device=device, dtype=dtype),
            "bond_index": -torch.ones(beam_batch, 1, device=device, dtype=dtype),
            "selected_bonds": torch.zeros(beam_batch, 1, query_dim, device=device, dtype=dtype),
            "bonds_mask": torch.ones(beam_batch, 1, device=device, dtype=dtype),
            "branch_index": -torch.ones(beam_batch, 1, device=device, dtype=dtype),
            "branchs_mask": torch.ones(beam_batch, 1, device=device, dtype=dtype),
        }
        batch_beam_mask = torch.ones(beam_batch, device=device, dtype=dtype)
        all_outputs = torch.full((1, beam_batch), self.vocab.sos_id, device=device, dtype=torch.long)
        all_masks = torch.ones_like(all_outputs, dtype=dtype)
        all_costs = torch.zeros_like(all_masks)
        all_branch_outputs = -torch.ones(1, beam_batch, 2, device=device, dtype=dtype)
        all_cond_inputs = -torch.ones(1, beam_batch, device=device, dtype=dtype)
        bond_ids = torch.tensor(self.decoder.bond_ids or (-999999,), device=device, dtype=torch.long)
        conn_pre_token = torch.cat([bond_ids, torch.tensor([self.decoder.connbranch_idx], device=device)])

        for step_idx in range(int(max_new_tokens)):
            if all_masks[-1].sum() == 0:
                break
            log_probs, pred_states, decode_states = self.decoder.eval_step_pred(
                encoded,
                encoded_proj,
                encoded_mask,
                states,
                decode_states,
            )
            log_probs = log_probs * batch_beam_mask.unsqueeze(1) - 1e6 * (1 - batch_beam_mask).unsqueeze(1)
            pre_conn = (all_outputs[-1].unsqueeze(1) == conn_pre_token).sum(1)
            log_probs[(pre_conn == 0).nonzero().flatten(), self.decoder.connbranch_idx] = -1e8
            next_costs = all_costs[-1, :, None] + log_probs * all_masks[-1, :, None]
            finished = (all_masks[-1] == 0).nonzero().flatten()
            if finished.numel() > 0:
                next_costs[finished, : self.vocab.eos_id] = -torch.inf
                next_costs[finished, self.vocab.eos_id + 1 :] = -torch.inf
            cur_beam = log_probs.shape[0] // batch_size
            avg_costs = next_costs / all_masks.sum(axis=0).clamp_min(1).view(-1, 1)
            avg_costs = avg_costs.view(batch_size, -1)
            if step_idx == 0:
                first = avg_costs.view(batch_size, cur_beam, -1)[:, 0, :]
                _, outputs = torch.topk(first, beam_size, dim=1)
                out_inds = (
                    torch.arange(batch_size, device=device).view(batch_size, 1) * cur_beam
                ).repeat(1, beam_size)
            else:
                _, flat_inds = torch.topk(avg_costs, beam_size, dim=1)
                outputs = flat_inds % self.vocab.vocab_size
                out_inds = flat_inds // self.vocab.vocab_size
                out_inds = out_inds + torch.arange(batch_size, device=device).view(batch_size, 1) * cur_beam
            outputs = outputs.reshape(1, -1)
            out_inds = out_inds.reshape(-1)
            all_outputs = torch.cat([all_outputs[:, out_inds], outputs.long()], dim=0)
            all_masks = torch.cat(
                [all_masks[:, out_inds], (outputs != self.vocab.eos_id).to(dtype)],
                dim=0,
            )
            all_costs = torch.cat([all_costs[:, out_inds], next_costs[out_inds, outputs.flatten()].view(1, -1)], dim=0)
            all_cond_inputs = all_cond_inputs[:, out_inds]
            all_branch_outputs = all_branch_outputs[:, out_inds, :]
            pred_states = [state[out_inds] for state in pred_states]
            decode_states = {key: value[out_inds] for key, value in decode_states.items()}

            new_states, new_decode_states = self.decoder.eval_step_update_memory(
                all_outputs,
                pred_states,
                decode_states,
                time_t=step_idx,
            )
            all_branch_outputs = torch.cat(
                [all_branch_outputs, new_decode_states["mem_cls_res"].unsqueeze(0)],
                dim=0,
            )
            y_t = all_outputs[-1]
            ea_mask = (y_t == self.decoder.ea_idx) * all_masks[-1].bool()
            if ea_mask.sum() > 0:
                cur_cand, cur_cand_mask = self.decoder.find_cand(
                    all_outputs,
                    all_masks,
                    new_decode_states["mem_index"],
                    new_decode_states["memory_used_mask"],
                )
                cand_num = cur_cand_mask.sum(1) * ea_mask
                expand_beam = int(cand_num.max().long().cpu().item()) if cand_num.numel() else 1
                expand_beam = max(1, expand_beam)
                cur_batch_beam = y_t.shape[0]
                if expand_beam > 1:
                    new_states = [self.decoder.expand_state_dim(state, expand_beam) for state in new_states]
                    new_decode_states = {
                        key: self.decoder.expand_state_dim(value, expand_beam)
                        for key, value in new_decode_states.items()
                    }
                    cur_length = all_outputs.shape[0]
                    all_outputs = all_outputs.unsqueeze(2).repeat(1, 1, expand_beam).view(cur_length, -1)
                    all_masks = all_masks.unsqueeze(2).repeat(1, 1, expand_beam).view(cur_length, -1)
                    all_costs = all_costs.unsqueeze(2).repeat(1, 1, expand_beam).view(cur_length, -1)
                    all_cond_inputs = (
                        all_cond_inputs.unsqueeze(2)
                        .repeat(1, 1, expand_beam)
                        .view(cur_length - 1, cur_batch_beam * expand_beam)
                    )
                    branch_dim = all_branch_outputs.shape[2]
                    all_branch_outputs = (
                        all_branch_outputs.unsqueeze(2)
                        .repeat(1, 1, expand_beam, 1)
                        .view(cur_length, cur_batch_beam * expand_beam, branch_dim)
                    )

                old_cond_mem = new_decode_states["cond_mem"].view(cur_batch_beam, expand_beam, -1)
                old_cond_weight = new_decode_states["cond_weight"].view(cur_batch_beam, expand_beam, 2, height, width)
                next_cond_mem = torch.zeros_like(old_cond_mem)
                next_cond_weight = torch.zeros_like(old_cond_weight)
                next_cond_input = -torch.ones(cur_batch_beam, expand_beam, device=device, dtype=dtype)
                batch_beam_mask_2d = torch.zeros(cur_batch_beam, expand_beam, device=device, dtype=dtype)

                memory = new_states[3].view(cur_batch_beam, expand_beam, new_states[3].shape[1], new_states[3].shape[2])
                memory_weight = new_states[5].view(cur_batch_beam, expand_beam, new_states[5].shape[1], 2, height, width)
                memory_used_mask = new_decode_states["memory_used_mask"].clone().view(cur_batch_beam, expand_beam, -1)
                mem_index = new_decode_states["mem_index"].view(cur_batch_beam, expand_beam, -1)

                cand_dict: dict[int, list[int]] = {}
                for row in range(cur_cand.shape[0]):
                    cand_len = int(cur_cand_mask[row].sum().long().cpu().item())
                    if cand_len > 0:
                        cand_dict[row] = [int(item) for item in cur_cand[row, :cand_len].long().cpu().tolist()]

                for row in range(cur_batch_beam):
                    if not bool(ea_mask[row]):
                        batch_beam_mask_2d[row, 0] = 1
                        continue
                    candidates = cand_dict.get(row, [])
                    if candidates:
                        for cand_id, mem_id in enumerate(candidates[:expand_beam]):
                            batch_beam_mask_2d[row, cand_id] = 1
                            next_cond_mem[row, cand_id, :] = memory[row, cand_id, mem_id]
                            next_cond_weight[row, cand_id, :] = memory_weight[row, cand_id, mem_id]
                            next_cond_input[row, cand_id] = mem_index[row, cand_id, mem_id]
                            memory_used_mask[row, cand_id, mem_id] = 0
                    else:
                        remaining = (memory_used_mask[row, 0] == 1).nonzero().flatten().long().cpu().tolist()
                        if remaining:
                            for cand_id, mem_id in enumerate(remaining[:expand_beam]):
                                batch_beam_mask_2d[row, cand_id] = 1
                                next_cond_mem[row, cand_id, :] = memory[row, cand_id, mem_id]
                                next_cond_weight[row, cand_id, :] = memory_weight[row, cand_id, mem_id]
                                next_cond_input[row, cand_id] = mem_index[row, cand_id, mem_id]
                                memory_used_mask[row, cand_id, mem_id] = 0
                        else:
                            batch_beam_mask_2d[row, 0] = 1
                            memory_used_mask[row, 0, :] = 0

                new_decode_states["cond_mem"] = next_cond_mem.view(cur_batch_beam * expand_beam, -1)
                new_decode_states["cond_weight"] = next_cond_weight.view(cur_batch_beam * expand_beam, 2, height, width)
                new_decode_states["memory_used_mask"] = memory_used_mask.view(cur_batch_beam * expand_beam, -1)
                all_cond_inputs = torch.cat([all_cond_inputs, next_cond_input.view(1, cur_batch_beam * expand_beam)], dim=0)
                batch_beam_mask = batch_beam_mask_2d.view(-1)
            else:
                next_cond_input = -torch.ones(1, y_t.shape[0], device=device, dtype=dtype)
                all_cond_inputs = torch.cat([all_cond_inputs, next_cond_input], dim=0)
                batch_beam_mask = torch.ones(y_t.shape[0], device=device, dtype=dtype)
                new_decode_states["cond_mem"][:] = 0
                new_decode_states["cond_weight"][:] = 0
            states, decode_states = self.decoder.eval_step_update(new_states, new_decode_states)

        return self._finalize_predictions(all_outputs, all_masks, all_costs, all_branch_outputs, all_cond_inputs, batch_size, beam_size)

    def _finalize_predictions(
        self,
        all_outputs: torch.Tensor,
        all_masks: torch.Tensor,
        all_costs: torch.Tensor,
        all_branch_outputs: torch.Tensor,
        all_cond_inputs: torch.Tensor,
        batch_size: int,
        beam_size: int,
    ) -> list[list[MsdPrediction]]:
        outputs_cpu = all_outputs.detach().cpu()
        masks_cpu = all_masks.detach().cpu()
        costs_cpu = all_costs.detach().cpu()
        branches_cpu = all_branch_outputs.detach().cpu()
        cond_cpu = all_cond_inputs.detach().cpu()
        actual_beam_size = max(1, outputs_cpu.shape[1] // batch_size)
        results: list[list[MsdPrediction]] = []
        for batch_idx in range(batch_size):
            beams: list[MsdPrediction] = []
            for beam_idx in range(actual_beam_size):
                col = batch_idx * actual_beam_size + beam_idx
                token_ids: list[int] = []
                for token_id, mask in zip(outputs_cpu[1:, col].tolist(), masks_cpu[1:, col].tolist()):
                    token_id = int(token_id)
                    if token_id == self.vocab.eos_id or float(mask) <= 0:
                        break
                    token_ids.append(token_id)
                tokens = self.vocab.decode(token_ids, stop_at_eos=True)
                branch_pairs: list[tuple[int, int]] = []
                for pair in branches_cpu[1:, col].tolist():
                    if len(pair) == 2 and pair[0] != -1 and pair[1] != -1:
                        branch_pairs.append((int(pair[0]), int(pair[1])))
                raw_cond = [int(item) for item in cond_cpu[1:, col].tolist()]
                cond_data = [-1, *raw_cond[: max(0, len(tokens) - 1)]]
                valid_len = max(1, len(token_ids))
                score = float(costs_cpu[: valid_len + 1, col].sum().item() / valid_len)
                beams.append(MsdPrediction(tokens=tokens, score=score, branch_pairs=branch_pairs, cond_data=cond_data[: len(tokens)]))
            beams.sort(key=lambda item: item.score, reverse=True)
            results.append(beams[:beam_size])
        return results

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": asdict(self.cfg),
                "base_model_name_or_path": self.base_model_name_or_path,
            },
            output_dir / "pytorch_model.bin",
        )
        self.vocab.save(output_dir / "rfl_vocab.txt")
        save_json(
            {
                "model_type": "texteller_msd",
                "base_model_name_or_path": self.base_model_name_or_path,
                "config": asdict(self.cfg),
            },
            output_dir / "texteller_msd_config.json",
        )


def load_texteller_msd_checkpoint(
    checkpoint_dir: Path,
    *,
    map_location: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
) -> TexTellerMsdModel:
    payload = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=map_location)
    vocab = RflVocab.from_file(checkpoint_dir / "rfl_vocab.txt")
    cfg_kwargs = dict(payload.get("config") or {})
    for key in ("decoder_cover_kernel", "decoder_cover_padding"):
        if key in cfg_kwargs and isinstance(cfg_kwargs[key], list):
            cfg_kwargs[key] = tuple(cfg_kwargs[key])
    cfg = MsdDecoderConfig(**cfg_kwargs)
    model = TexTellerMsdModel.from_pretrained(
        str(payload.get("base_model_name_or_path") or "OleehyO/TexTeller"),
        vocab,
        cfg,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.load_state_dict(payload["state_dict"])
    return model
