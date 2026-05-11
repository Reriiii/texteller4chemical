from __future__ import annotations

import re
from dataclasses import dataclass


SSML_GRAPH_NORM_FIELD = "ssml_graph_norm"
SSML_GRAPH_NORM_SOURCE_FIELD = "ssml_normed"

BOND_SPEC_RE = re.compile(
    r"(?P<prefix>[<>=~:\-]*\[:)"
    r"(?P<angle>-?\d+(?:\.\d+)?)"
    r","
    r"(?P<length>\d+(?:\.\d+)?)"
    r"(?P<suffix>\])"
)


@dataclass(frozen=True)
class GraphNormalizationStats:
    changed: bool
    bond_specs_seen: int
    bond_specs_changed: int


def is_graph_norm_field(target_field: str) -> bool:
    return target_field == SSML_GRAPH_NORM_FIELD


def normalize_ssml_graph(
    text: str,
    *,
    angle_step: float = 1.0,
    angle_decimals: int = 1,
    length_decimals: int = 1,
) -> str:
    """Normalize visually noisy SSML bond geometry without changing graph semantics.

    The GraphMatchingTool ignores small numeric formatting differences in bond
    geometry. Rounding these values reduces target vocabulary sparsity while
    preserving chemical graph comparison.
    """
    return normalize_ssml_graph_with_stats(
        text,
        angle_step=angle_step,
        angle_decimals=angle_decimals,
        length_decimals=length_decimals,
    )[0]


def normalize_ssml_graph_with_stats(
    text: str,
    *,
    angle_step: float = 1.0,
    angle_decimals: int = 1,
    length_decimals: int = 1,
) -> tuple[str, GraphNormalizationStats]:
    if angle_step <= 0:
        raise ValueError("angle_step must be positive.")
    if angle_decimals < 0 or length_decimals < 0:
        raise ValueError("angle_decimals and length_decimals must be non-negative.")

    seen = 0
    changed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal seen, changed
        seen += 1

        angle = float(match.group("angle"))
        length = float(match.group("length"))

        rounded_angle = (round(angle / angle_step) * angle_step) % 360.0
        rounded_length = round(length, length_decimals)

        angle_text = f"{rounded_angle:.{angle_decimals}f}"
        length_text = f"{rounded_length:.{length_decimals}f}"
        replacement = (
            f"{match.group('prefix')}{angle_text},{length_text}{match.group('suffix')}"
        )
        if replacement != match.group(0):
            changed += 1
        return replacement

    normalized = BOND_SPEC_RE.sub(repl, text)
    return normalized, GraphNormalizationStats(
        changed=normalized != text,
        bond_specs_seen=seen,
        bond_specs_changed=changed,
    )


def normalize_target_for_field(value: str, target_field: str) -> str:
    if is_graph_norm_field(target_field):
        return normalize_ssml_graph(value)
    return value
