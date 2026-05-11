from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RflConversionResult:
    success: bool
    target: str
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


def convert_ssml_to_rfl(source: str, rfl_tool_dir: Path) -> RflConversionResult:
    """Convert SSML/ChemFig-like markup to RFL using the external RFL-MSD code.

    The RFL-MSD repository is intentionally kept in external/ instead of being
    vendored into src/. This adapter gives our data pipeline a stable boundary:
    import the official converter when available, return structured failures
    when a sample cannot be converted, and leave the original labels untouched.
    """
    if not rfl_tool_dir.is_dir():
        return RflConversionResult(
            success=False,
            target="",
            error=f"RFL tool directory does not exist: {rfl_tool_dir}",
        )
    try:
        import numpy

        if not hasattr(numpy, "float"):
            numpy.float = float  # type: ignore[attr-defined]

        with _prepend_sys_path(rfl_tool_dir):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                from RFL import cs_main  # type: ignore

                result = cs_main(source, is_show=False)
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
    return RflConversionResult(success=True, target=" ".join(cs_string))
