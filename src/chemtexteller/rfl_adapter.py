from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
import re


@dataclass(frozen=True)
class RflConversionResult:
    success: bool
    target: str
    tokens: tuple[str, ...] = ()
    branch_info: list[Any] | None = None
    ring_branch_info: list[Any] | None = None
    cond_data: list[int] | None = None
    ring_count: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class RflRestoreResult:
    success: bool
    chemfig: str
    tokens: tuple[str, ...] = ()
    ring_branch_info: list[Any] | None = None
    cond_data: list[int] | None = None
    branch_pairs: tuple[tuple[int, int], ...] = ()
    strategy: str = ""
    error: str | None = None


RFL_CONNBRANCH_TOKEN = r"\connbranch"
RFL_SUPERATOM_TOKEN = r"\Superatom"
RFL_CHEMFIG_TOKEN = r"\chemfig"
RFL_EXTRA_ATOM_TOKEN = "<ea>"
RFL_COMPACT_BOND_RE = re.compile(
    r"^(?P<prefix>.+\[:)(?P<angle>-?\d+(?:\.\d+)?)(?P<suffix>\])$"
)


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path.resolve())
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def resolve_rfl_module_dir(rfl_tool_dir: Path) -> Path:
    """Return the import directory that contains RFL-MSD's RFL.py module."""
    if (rfl_tool_dir / "RFL.py").is_file():
        return rfl_tool_dir
    nested = rfl_tool_dir / "RFL"
    if (nested / "RFL.py").is_file():
        return nested
    return rfl_tool_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def split_rfl_text(text: str) -> list[str]:
    return text.split()


def is_rfl_bond_token(token: str) -> bool:
    return "[:" in token and any(char in token for char in ("-", "=", ":", "~", "<", ">"))


def is_rfl_super_token(token: str) -> bool:
    return RFL_SUPERATOM_TOKEN in token or "@" in token


def infer_rfl_bond_token_indices(tokens: Sequence[str]) -> list[int]:
    return [idx for idx, token in enumerate(tokens) if is_rfl_bond_token(token)]


def infer_rfl_branch_token_indices(tokens: Sequence[str]) -> list[int]:
    """Return the bond-token indices that own generated \\connbranch markers."""
    branches: list[int] = []
    for idx, token in enumerate(tokens):
        if token != RFL_CONNBRANCH_TOKEN:
            continue
        for prev_idx in range(idx - 1, -1, -1):
            if is_rfl_bond_token(tokens[prev_idx]):
                branches.append(prev_idx)
                break
    return branches


def infer_rfl_cond_data(tokens: Sequence[str]) -> list[int]:
    """Best-effort reconstruction of RFL condition guidance for graph restore.

    RFL stores one condition pointer from the first ring token after <ea> to the
    corresponding \\Superatom/@ token in the stem. During generation TexTeller
    emits only the token sequence, so evaluation has to reconstruct that pointer
    from the emitted <ea> delimiters.
    """
    cond_data = [-1] * len(tokens)
    super_token_indices: list[int] = []
    next_super = 0
    pending_super: int | None = None
    for idx, token in enumerate(tokens):
        if is_rfl_super_token(token):
            super_token_indices.append(idx)
            continue
        if token == RFL_EXTRA_ATOM_TOKEN:
            if next_super < len(super_token_indices):
                pending_super = super_token_indices[next_super]
                next_super += 1
            else:
                pending_super = None
            continue
        if pending_super is None:
            continue
        if token in {"{", "}", RFL_CHEMFIG_TOKEN}:
            continue
        cond_data[idx] = pending_super
        pending_super = None
    return cond_data


def infer_rfl_branch_pairs(
    tokens: Sequence[str],
    *,
    strategy: str = "previous_bond",
) -> list[tuple[int, int]]:
    if strategy.lower() in {"", "none", "off", "false"}:
        return []
    if strategy != "previous_bond":
        raise ValueError(f"Unsupported RFL restore branch strategy: {strategy}")
    bond_indices = infer_rfl_bond_token_indices(tokens)
    branch_indices = infer_rfl_branch_token_indices(tokens)
    pairs: list[tuple[int, int]] = []
    for branch_idx in branch_indices:
        previous_bonds = [bond_idx for bond_idx in bond_indices if bond_idx < branch_idx]
        if previous_bonds:
            pairs.append((branch_idx, previous_bonds[-1]))
    return pairs


def build_rfl_ring_branch_info(
    tokens: Sequence[str],
    branch_pairs: Sequence[tuple[int, int]] | None,
) -> list[Any]:
    ring_branch_info: list[Any] = [None] * len(tokens)
    if not branch_pairs:
        return ring_branch_info
    for branch_idx, bond_idx in branch_pairs:
        if branch_idx < 0 or branch_idx >= len(tokens):
            continue
        if bond_idx < 0 or bond_idx >= len(tokens):
            continue
        existing = ring_branch_info[branch_idx]
        if existing is None:
            ring_branch_info[branch_idx] = [int(bond_idx)]
        elif isinstance(existing, list):
            existing.append(int(bond_idx))
    return ring_branch_info


def canonicalize_rfl_restored_bonds(text: str) -> str:
    """Expand compact RFL bonds like -[:90] to graph-eval-friendly -[:90.0,1.0]."""
    tokens: list[str] = []
    for token in text.split():
        if "," in token:
            tokens.append(token)
            continue
        match = RFL_COMPACT_BOND_RE.match(token)
        if not match:
            tokens.append(token)
            continue
        angle = float(match.group("angle"))
        tokens.append(f"{match.group('prefix')}{angle:.1f},1.0{match.group('suffix')}")
    return " ".join(tokens)


def restore_rfl_tokens_to_chemfig(
    tokens: Sequence[str],
    rfl_tool_dir: Path,
    *,
    branch_pairs: Sequence[tuple[int, int]] | None = None,
    branch_strategy: str = "previous_bond",
    cond_data: Sequence[int] | None = None,
) -> RflRestoreResult:
    rfl_module_dir = resolve_rfl_module_dir(rfl_tool_dir)
    token_list = [str(token) for token in tokens]
    if not token_list:
        return RflRestoreResult(
            success=False,
            chemfig="",
            tokens=(),
            error="Empty RFL token sequence.",
        )
    if not rfl_module_dir.is_dir():
        return RflRestoreResult(
            success=False,
            chemfig=" ".join(token_list),
            tokens=tuple(token_list),
            error=f"RFL tool directory does not exist: {rfl_tool_dir}",
        )
    try:
        pairs = (
            [(int(branch_idx), int(bond_idx)) for branch_idx, bond_idx in branch_pairs]
            if branch_pairs is not None
            else infer_rfl_branch_pairs(token_list, strategy=branch_strategy)
        )
        ring_branch_info = build_rfl_ring_branch_info(token_list, pairs)
        restore_cond_data = (
            [int(item) for item in cond_data]
            if cond_data is not None
            else infer_rfl_cond_data(token_list)
        )
        if len(restore_cond_data) != len(token_list):
            raise ValueError(
                "cond_data length must match RFL token length: "
                f"{len(restore_cond_data)} != {len(token_list)}"
            )
        import numpy

        if not hasattr(numpy, "float"):
            numpy.float = float  # type: ignore[attr-defined]

        with _prepend_sys_path(rfl_module_dir):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                from RFL import chemstem2chemfig  # type: ignore

                restored = chemstem2chemfig(
                    list(token_list),
                    [None] * len(token_list),
                    list(ring_branch_info),
                    list(restore_cond_data),
                    show=False,
                    add_extra_token=True,
                )
    except Exception as exc:
        return RflRestoreResult(
            success=False,
            chemfig=" ".join(token_list),
            tokens=tuple(token_list),
            branch_pairs=tuple(branch_pairs or ()),
            strategy=branch_strategy,
            error=str(exc),
        )

    restored = restored.replace("(", "branch(").replace(")", "branch)")
    restored = canonicalize_rfl_restored_bonds(restored)
    return RflRestoreResult(
        success=True,
        chemfig=restored,
        tokens=tuple(token_list),
        ring_branch_info=ring_branch_info,
        cond_data=restore_cond_data,
        branch_pairs=tuple(pairs),
        strategy=branch_strategy,
    )


def restore_rfl_text_to_chemfig(
    text: str,
    rfl_tool_dir: Path,
    *,
    branch_pairs: Sequence[tuple[int, int]] | None = None,
    branch_strategy: str = "previous_bond",
    cond_data: Sequence[int] | None = None,
) -> RflRestoreResult:
    return restore_rfl_tokens_to_chemfig(
        split_rfl_text(text),
        rfl_tool_dir,
        branch_pairs=branch_pairs,
        branch_strategy=branch_strategy,
        cond_data=cond_data,
    )


def convert_ssml_to_rfl(
    source: str,
    rfl_tool_dir: Path,
    *,
    need_ring_num: bool = False,
) -> RflConversionResult:
    """Convert SSML/ChemFig-like markup to RFL using the external RFL-MSD code.

    The RFL-MSD repository is intentionally kept in external/ instead of being
    vendored into src/. This adapter gives our data pipeline a stable boundary:
    import the official converter when available, return structured failures
    when a sample cannot be converted, and leave the original labels untouched.
    """
    rfl_module_dir = resolve_rfl_module_dir(rfl_tool_dir)
    if not rfl_module_dir.is_dir():
        return RflConversionResult(
            success=False,
            target="",
            error=f"RFL tool directory does not exist: {rfl_tool_dir}",
        )
    try:
        import numpy

        if not hasattr(numpy, "float"):
            numpy.float = float  # type: ignore[attr-defined]

        with _prepend_sys_path(rfl_module_dir):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                from RFL import cs_main  # type: ignore

                actual_need_ring_num = need_ring_num
                try:
                    result = cs_main(source, is_show=False, need_ring_num=actual_need_ring_num)
                except ValueError as exc:
                    if (
                        actual_need_ring_num
                        and "not enough values to unpack" in str(exc)
                    ):
                        actual_need_ring_num = False
                        result = cs_main(source, is_show=False, need_ring_num=False)
                    else:
                        raise
    except Exception as exc:
        return RflConversionResult(success=False, target="", error=str(exc))

    if not result or not result[0]:
        return RflConversionResult(success=False, target="", error="RFL conversion failed")
    cs_string = result[1]
    if not isinstance(cs_string, list) or not all(isinstance(item, str) for item in cs_string):
        return RflConversionResult(
            success=False,
            target="",
            error=f"Unexpected RFL converter output type: {type(cs_string).__name__}",
        )
    branch_info = _jsonable(result[2]) if len(result) > 2 else None
    ring_branch_info = _jsonable(result[3]) if len(result) > 3 else None
    cond_data = _jsonable(result[4]) if len(result) > 4 else None
    ring_count = int(result[5]) if actual_need_ring_num and len(result) > 5 else None
    return RflConversionResult(
        success=True,
        target=" ".join(cs_string),
        tokens=tuple(cs_string),
        branch_info=branch_info,
        ring_branch_info=ring_branch_info,
        cond_data=cond_data,
        ring_count=ring_count,
    )
