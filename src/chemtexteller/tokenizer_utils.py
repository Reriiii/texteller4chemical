from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import read_jsonl


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def whitespace_tokenize(text: str) -> list[str]:
    return text.strip().split()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def load_vocab_file(path: Path) -> list[str]:
    tokens: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    return tokens


def load_targets_from_metadata(metadata_path: Path, target_key: str = "target") -> list[str]:
    rows = read_jsonl(metadata_path)
    targets: list[str] = []
    for idx, row in enumerate(rows):
        if target_key not in row:
            raise KeyError(f"Missing key '{target_key}' in row {idx} of {metadata_path}")
        target = row[target_key]
        if not isinstance(target, str):
            raise TypeError(f"Expected string target in row {idx}, got {type(target).__name__}")
        targets.append(target)
    return targets


def token_counter(targets: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for target in targets:
        counter.update(whitespace_tokenize(target))
    return counter


def build_special_token_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    if tokenizer.pad_token is None:
        kwargs["pad_token"] = "<pad>"
    if tokenizer.bos_token is None:
        kwargs["bos_token"] = "<bos>"
    if tokenizer.eos_token is None:
        kwargs["eos_token"] = "<eos>"
    if tokenizer.unk_token is None:
        kwargs["unk_token"] = "<unk>"
    return kwargs


def load_hf_tokenizer(
    model_name_or_path: str | Path,
    tokenizer_path: str | Path | None = None,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    source = str(tokenizer_path or model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    special_kwargs = build_special_token_kwargs(tokenizer)
    if special_kwargs:
        tokenizer.add_special_tokens(special_kwargs)
    return tokenizer


def add_chemical_tokens(
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[str],
) -> int:
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [tok for tok in dict.fromkeys(tokens) if tok and tok not in existing_vocab]
    if not new_tokens:
        return 0
    return tokenizer.add_tokens(new_tokens, special_tokens=False)


def tokenizer_unknown_stats(
    tokenizer: PreTrainedTokenizerBase,
    targets: list[str],
) -> dict[str, Any]:
    unk_id = tokenizer.unk_token_id
    raw_token_count = 0
    unknown_count = 0
    risky: Counter[str] = Counter()
    sequence_lengths: list[int] = []

    for target in targets:
        pieces = tokenizer(target, add_special_tokens=True, truncation=False)["input_ids"]
        sequence_lengths.append(len(pieces))
        for token in whitespace_tokenize(target):
            raw_token_count += 1
            encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
            has_unknown = unk_id is not None and unk_id in encoded
            if has_unknown:
                unknown_count += 1
                risky[token] += 1
            elif len(encoded) > 4:
                risky[token] += 1

    unknown_ratio = unknown_count / raw_token_count if raw_token_count else 0.0
    return {
        "raw_whitespace_token_count": raw_token_count,
        "unknown_token_count": unknown_count,
        "unknown_token_ratio": unknown_ratio,
        "tokenized_lengths": sequence_lengths,
        "risky_tokens": risky,
    }
