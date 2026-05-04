from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .utils import ensure_dir


METRIC_HEADER_RE = re.compile(r"-+\s*metric\s+([A-Za-z0-9_.-]+)\s*-+")
SENT_ACC_RE = re.compile(
    r"sent\s+acc\s*=\s*([+-]?\d+(?:\.\d+)?)\s*%\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)"
)


@dataclass(frozen=True)
class GraphMatchingResult:
    metrics: dict[str, Any]
    stdout: str
    stderr: str
    output_path: Path


def clean_graph_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\t", " ").splitlines()).strip()


def lookup_target(targets: Mapping[str, Any], key: str) -> str:
    if key.startswith("targets."):
        key = key.split(".", 1)[1]
    direct_value = targets.get(key)
    if isinstance(direct_value, str) and direct_value.strip():
        return direct_value.strip()

    value: Any = targets
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(key)
        value = value[part]
    if not isinstance(value, str) or not value.strip():
        raise KeyError(key)
    return value.strip()


def validate_graph_matching_tool(tool_dir: Path) -> Path:
    tool_dir = tool_dir.resolve()
    eval_script = tool_dir / "eval.py"
    if not tool_dir.exists():
        raise FileNotFoundError(f"GraphMatchingTool directory does not exist: {tool_dir}")
    if not eval_script.exists():
        raise FileNotFoundError(
            f"GraphMatchingTool eval.py was not found: {eval_script}"
        )
    return eval_script


def write_graph_matching_files(
    rows: list[Mapping[str, Any]],
    rec_path: Path,
    lab_path: Path,
) -> None:
    ensure_dir(rec_path.parent)
    ensure_dir(lab_path.parent)
    with rec_path.open("w", encoding="utf-8", newline="\n") as rec_file, lab_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as lab_file:
        for idx, row in enumerate(rows):
            image_name = clean_graph_text(row.get("image_name"))
            prediction = clean_graph_text(row.get("prediction"))
            label = clean_graph_text(row.get("graph_label"))
            if not image_name:
                raise ValueError(f"Missing image_name for graph evaluation row {idx}")
            if not label:
                raise ValueError(
                    f"Missing graph_label for graph evaluation row {idx} ({image_name})"
                )
            rec_file.write(f"{image_name}\t{prediction}\n")
            lab_file.write(f"{image_name}\t{label}\n")


def parse_graph_matching_stdout(stdout: str) -> dict[str, Any]:
    current_metric: str | None = None
    parsed: dict[str, tuple[float, int, int]] = {}
    for line in stdout.splitlines():
        header = METRIC_HEADER_RE.search(line)
        if header:
            current_metric = header.group(1)
            continue
        sent_acc = SENT_ACC_RE.search(line)
        if sent_acc and current_metric:
            percent = float(sent_acc.group(1))
            correct = int(sent_acc.group(2))
            total = int(sent_acc.group(3))
            parsed[current_metric] = (percent / 100.0, correct, total)

    metrics: dict[str, Any] = {}
    metric_mapping = {
        "base": "graph_base_sent_acc",
        "struct": "graph_structure_em",
        "struct.line": "graph_em",
    }
    for source_name, output_name in metric_mapping.items():
        if source_name not in parsed:
            continue
        value, correct, total = parsed[source_name]
        metrics[output_name] = value
        metrics[f"{output_name}_correct"] = correct
        metrics[f"{output_name}_total"] = total

    required = ("graph_structure_em", "graph_em")
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(
            "Could not parse required GraphMatchingTool metrics "
            f"{missing} from stdout:\n{stdout[-2000:]}"
        )
    return metrics


def run_graph_matching_tool(
    tool_dir: Path,
    rec_path: Path,
    lab_path: Path,
    output_path: Path,
    num_workers: int,
) -> GraphMatchingResult:
    eval_script = validate_graph_matching_tool(tool_dir)
    tool_dir = eval_script.parent
    rec_path = rec_path.resolve()
    lab_path = lab_path.resolve()
    output_path = output_path.resolve()
    ensure_dir(output_path.parent)
    cmd = [
        sys.executable,
        str(eval_script),
        "-rec",
        str(rec_path),
        "-lab",
        str(lab_path),
        "-output",
        str(output_path),
        "-num_workers",
        str(num_workers),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(
        cmd,
        cwd=str(tool_dir),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "GraphMatchingTool failed with exit code "
            f"{completed.returncode}.\nSTDOUT:\n{completed.stdout[-2000:]}\n"
            f"STDERR:\n{completed.stderr[-2000:]}"
        )
    metrics = parse_graph_matching_stdout(completed.stdout)
    return GraphMatchingResult(
        metrics=metrics,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_path=output_path,
    )
