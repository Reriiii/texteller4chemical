from __future__ import annotations

import re
from dataclasses import dataclass


SSML_GRAPH_NORM_FIELD = "ssml_graph_norm"
SSML_GRAPH_SD_FIELD = "ssml_graph_sd"
SSML_GRAPH_COMPACT_FIELD = "ssml_graph_compact"
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
    lengths_seen: int = 0
    lengths_dropped: int = 0
    lengths_preserved_nonzero: int = 0


def is_graph_norm_field(target_field: str) -> bool:
    return target_field in {
        SSML_GRAPH_NORM_FIELD,
        SSML_GRAPH_SD_FIELD,
        SSML_GRAPH_COMPACT_FIELD,
    }


def normalize_ssml_graph(
    text: str,
    *,
    angle_step: float = 1.0,
    angle_decimals: int = 1,
    length_decimals: int = 1,
    drop_default_length: bool = False,
    drop_all_lengths: bool = False,
    default_length: float = 1.0,
    length_default_tolerance: float = 0.05,
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
        drop_default_length=drop_default_length,
        drop_all_lengths=drop_all_lengths,
        default_length=default_length,
        length_default_tolerance=length_default_tolerance,
    )[0]


def normalize_ssml_graph_with_stats(
    text: str,
    *,
    angle_step: float = 1.0,
    angle_decimals: int = 1,
    length_decimals: int = 1,
    drop_default_length: bool = False,
    drop_all_lengths: bool = False,
    default_length: float = 1.0,
    length_default_tolerance: float = 0.05,
) -> tuple[str, GraphNormalizationStats]:
    if angle_step <= 0:
        raise ValueError("angle_step must be positive.")
    if angle_decimals < 0 or length_decimals < 0:
        raise ValueError("angle_decimals and length_decimals must be non-negative.")
    if length_default_tolerance < 0:
        raise ValueError("length_default_tolerance must be non-negative.")

    seen = 0
    changed = 0
    lengths_seen = 0
    lengths_dropped = 0
    lengths_preserved_nonzero = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal seen, changed, lengths_seen, lengths_dropped, lengths_preserved_nonzero
        seen += 1

        angle = float(match.group("angle"))
        length = float(match.group("length"))
        raw_length_text = match.group("length")
        lengths_seen += 1

        rounded_angle = (round(angle / angle_step) * angle_step) % 360.0
        rounded_length = round(length, length_decimals)

        angle_text = f"{rounded_angle:.{angle_decimals}f}"
        length_text = f"{rounded_length:.{length_decimals}f}"
        if length > 0.0 and rounded_length == 0.0:
            # A positive bond length rounded to zero can become a degenerate
            # edge for GraphMatchingTool, so preserve the original nonzero
            # geometry instead of turning it into 0.0.
            length_text = raw_length_text
            lengths_preserved_nonzero += 1
        should_drop_length = (
            drop_all_lengths
            or (
                drop_default_length
                and abs(length - default_length) <= length_default_tolerance + 1e-9
            )
        )
        if should_drop_length:
            lengths_dropped += 1
            replacement = f"{match.group('prefix')}{angle_text}{match.group('suffix')}"
        else:
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
        lengths_seen=lengths_seen,
        lengths_dropped=lengths_dropped,
        lengths_preserved_nonzero=lengths_preserved_nonzero,
    )


def normalize_ssml_graph_compact(text: str) -> str:
    """CROCS-style compact SSML target with conservative length dropping.

    CROCS' sequence-decoder baseline uses quantized bond angles and omits bond
    lengths. This variant follows that idea, but only drops lengths that are
    close to the SSML default of 1.0 so visibly unusual bond lengths are kept.
    """
    return normalize_ssml_graph(
        text,
        angle_step=15.0,
        angle_decimals=1,
        length_decimals=1,
        drop_default_length=True,
        default_length=1.0,
        length_default_tolerance=0.05,
    )


def normalize_ssml_graph_sd(text: str) -> str:
    """CROCS SSSL-SD style target: 15-degree bonds and omitted lengths."""
    return normalize_ssml_graph(
        text,
        angle_step=15.0,
        angle_decimals=0,
        length_decimals=1,
        drop_all_lengths=True,
    )


def normalize_ssml_graph_compact_with_stats(
    text: str,
) -> tuple[str, GraphNormalizationStats]:
    return normalize_ssml_graph_with_stats(
        text,
        angle_step=15.0,
        angle_decimals=1,
        length_decimals=1,
        drop_default_length=True,
        default_length=1.0,
        length_default_tolerance=0.05,
    )


def normalize_ssml_graph_sd_with_stats(
    text: str,
) -> tuple[str, GraphNormalizationStats]:
    return normalize_ssml_graph_with_stats(
        text,
        angle_step=15.0,
        angle_decimals=0,
        length_decimals=1,
        drop_all_lengths=True,
    )


def normalize_target_for_field_with_stats(
    value: str,
    target_field: str,
) -> tuple[str, GraphNormalizationStats]:
    if target_field == SSML_GRAPH_NORM_FIELD:
        return normalize_ssml_graph_with_stats(value)
    if target_field == SSML_GRAPH_SD_FIELD:
        return normalize_ssml_graph_sd_with_stats(value)
    if target_field == SSML_GRAPH_COMPACT_FIELD:
        return normalize_ssml_graph_compact_with_stats(value)
    return value, GraphNormalizationStats(
        changed=False,
        bond_specs_seen=0,
        bond_specs_changed=0,
    )


def normalize_target_for_field(value: str, target_field: str) -> str:
    if target_field == SSML_GRAPH_NORM_FIELD:
        return normalize_ssml_graph(value)
    if target_field == SSML_GRAPH_SD_FIELD:
        return normalize_ssml_graph_sd(value)
    if target_field == SSML_GRAPH_COMPACT_FIELD:
        return normalize_ssml_graph_compact(value)
    return value
