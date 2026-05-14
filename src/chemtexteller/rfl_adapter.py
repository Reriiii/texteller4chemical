from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


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
