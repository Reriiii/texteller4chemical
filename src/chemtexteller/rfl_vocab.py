from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RflVocab:
    word2id: dict[str, int]
    id2word: dict[int, str]
    unk_token: str = r"\unk"

    @classmethod
    def from_file(cls, path: Path, *, unk_token: str = r"\unk") -> "RflVocab":
        word2id: dict[str, int] = {}
        id2word: dict[int, str] = {}
        next_id = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 1:
                    word = parts[0]
                    token_id = next_id
                    next_id += 1
                elif len(parts) == 2:
                    word = parts[0]
                    try:
                        token_id = int(parts[1])
                    except ValueError as exc:
                        raise ValueError(f"Bad token id in {path}:{line_no}: {parts[1]!r}") from exc
                else:
                    raise ValueError(f"Bad vocab line in {path}:{line_no}: {raw_line!r}")
                word2id[word] = token_id
                id2word[token_id] = word
        if not word2id:
            raise ValueError(f"Empty RFL vocab: {path}")
        return cls(word2id=word2id, id2word=id2word, unk_token=unk_token)

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        *,
        special_tokens: tuple[str, ...] = (r"\unk", "<s>", "</s>"),
        unk_token: str = r"\unk",
    ) -> "RflVocab":
        ordered: list[str] = []
        seen: set[str] = set()
        for token in [*special_tokens, *tokens]:
            token = str(token)
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return cls(
            word2id={token: idx for idx, token in enumerate(ordered)},
            id2word={idx: token for idx, token in enumerate(ordered)},
            unk_token=unk_token,
        )

    @property
    def unk_id(self) -> int:
        return self.word2id.get(self.unk_token, 0)

    @property
    def pad_id(self) -> int:
        return self.unk_id

    @property
    def sos_id(self) -> int:
        return self.get_id("<s>")

    @property
    def eos_id(self) -> int:
        return self.get_id("</s>")

    @property
    def vocab_size(self) -> int:
        return max(self.id2word) + 1

    def get_id(self, token: str) -> int:
        return self.word2id.get(token, self.unk_id)

    def get_word(self, token_id: int) -> str:
        return self.id2word.get(int(token_id), self.unk_token)

    def encode(self, tokens: Iterable[str], *, add_eos: bool = True) -> list[int]:
        ids = [self.get_id(token) for token in tokens]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], *, stop_at_eos: bool = True) -> list[str]:
        tokens: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if stop_at_eos and token_id == self.eos_id:
                break
            tokens.append(self.get_word(token_id))
        return tokens

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for token_id in sorted(self.id2word):
                handle.write(f"{self.id2word[token_id]}\t{token_id}\n")
