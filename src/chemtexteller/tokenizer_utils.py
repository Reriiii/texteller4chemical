from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import re
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import read_jsonl


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
CHEMICAL_MARKUP_RE = re.compile(
    r"branch\(|branch\)"
    r"|\\[A-Za-z]+"
    r"|\?\[[^\]]+\]"
    r"|[-=~<>:|]*\[:\s*-?\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\]"
)

DEFAULT_CHEMICAL_TOKENS = [
    r"\chemfig",
    r"\circle",
    r"\Chemabove",
    r"\Charge",
    r"\lewis",
    r"\rightarrow",
    r"\xrightarrow",
    r"\xrightleftharpoons",
    "branch(",
    "branch)",
]
TOKEN_CATEGORIES = {"branch", "macro", "ring_marker", "bond_geometry", "other"}


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


def target_from_metadata_row(
    row: dict[str, Any],
    target_key: str = "target",
) -> str:
    value: Any = row
    if target_key.startswith("targets."):
        value = row.get("targets", {})
        target_key = target_key.split(".", 1)[1]

    if "." not in target_key:
        direct = row.get(target_key)
        if isinstance(direct, str) and direct.strip():
            return direct
        raw_targets = row.get("targets")
        if isinstance(raw_targets, dict):
            nested = raw_targets.get(target_key)
            if isinstance(nested, str) and nested.strip():
                return nested

    for part in target_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(target_key)
        value = value[part]
    if not isinstance(value, str) or not value.strip():
        raise KeyError(target_key)
    return value


def load_targets_from_metadata(metadata_path: Path, target_key: str = "target") -> list[str]:
    rows = read_jsonl(metadata_path)
    targets: list[str] = []
    for idx, row in enumerate(rows):
        try:
            targets.append(target_from_metadata_row(row, target_key))
        except KeyError as exc:
            raise KeyError(
                f"Missing non-empty target for key '{target_key}' in row {idx} of {metadata_path}"
            ) from exc
    return targets


def token_counter(targets: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for target in targets:
        counter.update(whitespace_tokenize(target))
    return counter


def _nested_get(row: dict[str, Any], key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def chemical_markup_tokens(text: str) -> list[str]:
    return CHEMICAL_MARKUP_RE.findall(str(text or ""))


def is_chemical_token_candidate(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if token in {"branch(", "branch)"}:
        return True
    if token.startswith("\\") and len(token) > 1:
        return True
    if token.startswith("?[") and token.endswith("]"):
        return True
    if "[:" in token and token.endswith("]"):
        return True
    return False


def chemical_token_category(token: str) -> str:
    token = token.strip()
    if token in {"branch(", "branch)"}:
        return "branch"
    if token.startswith("\\") and len(token) > 1:
        return "macro"
    if token.startswith("?[") and token.endswith("]"):
        return "ring_marker"
    if "[:" in token and token.endswith("]"):
        return "bond_geometry"
    return "other"


def chemical_token_counter(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in whitespace_tokenize(text) + chemical_markup_tokens(text):
            token = token.strip()
            if is_chemical_token_candidate(token):
                counter[token] += 1
    return counter


def load_texts_from_csv(
    path: Path,
    fields: list[str],
    max_rows: int | None = None,
) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if max_rows is not None and row_idx >= max_rows:
                break
            for field in fields:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    texts.append(value)
    return texts


def load_texts_from_metadata(
    path: Path,
    target_keys: list[str],
    max_rows: int | None = None,
) -> list[str]:
    rows = read_jsonl(path)
    texts: list[str] = []
    for row_idx, row in enumerate(rows):
        if max_rows is not None and row_idx >= max_rows:
            break
        for key in target_keys:
            try:
                value = target_from_metadata_row(row, key)
            except KeyError:
                nested = _nested_get(row, key)
                value = nested if isinstance(nested, str) else ""
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def extract_chemical_tokens_from_sources(
    sources: list[Path],
    *,
    csv_fields: list[str],
    metadata_target_keys: list[str],
    min_frequency: int = 10,
    max_tokens: int | None = None,
    include_categories: set[str] | None = None,
    exclude_categories: set[str] | None = None,
    max_tokens_per_category: dict[str, int] | None = None,
    max_rows_per_source: int | None = None,
    include_default_tokens: bool = True,
) -> tuple[list[str], Counter[str]]:
    counter: Counter[str] = Counter()
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Chemical-token source does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix == ".csv":
            texts = load_texts_from_csv(source, csv_fields, max_rows=max_rows_per_source)
        elif suffix in {".jsonl", ".json"}:
            texts = load_texts_from_metadata(
                source,
                metadata_target_keys,
                max_rows=max_rows_per_source,
            )
        else:
            with source.open("r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
                if max_rows_per_source is not None:
                    texts = texts[:max_rows_per_source]
        counter.update(chemical_token_counter(texts))

    if include_default_tokens:
        for token in DEFAULT_CHEMICAL_TOKENS:
            counter.setdefault(token, min_frequency)

    if include_categories:
        unknown = set(include_categories) - TOKEN_CATEGORIES
        if unknown:
            raise ValueError(f"Unknown include_categories: {sorted(unknown)}")
    if exclude_categories:
        unknown = set(exclude_categories) - TOKEN_CATEGORIES
        if unknown:
            raise ValueError(f"Unknown exclude_categories: {sorted(unknown)}")
    per_category_counts: Counter[str] = Counter()
    tokens: list[str] = []
    for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if count < min_frequency:
            continue
        category = chemical_token_category(token)
        if include_categories and category not in include_categories:
            continue
        if exclude_categories and category in exclude_categories:
            continue
        category_limit = (
            max_tokens_per_category.get(category)
            if isinstance(max_tokens_per_category, dict)
            else None
        )
        if category_limit is not None and per_category_counts[category] >= int(category_limit):
            continue
        tokens.append(token)
        per_category_counts[category] += 1
    if max_tokens is not None and max_tokens > 0:
        tokens = tokens[:max_tokens]
    return tokens, counter


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
