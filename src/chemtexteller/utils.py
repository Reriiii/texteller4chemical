from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


LOGGER_NAME = "chemtexteller"


def is_main_process() -> bool:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) in {"", "-1", "0"}


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    effective_level = level if is_main_process() else logging.WARNING
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None and is_main_process():
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=effective_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy_logger in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "urllib3",
        "absl",
        "tensorflow",
        "tensorboard",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
    return logging.getLogger(LOGGER_NAME)


def get_logger(name: str | None = None) -> logging.Logger:
    base = LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(base)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_or_symlink(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode != "symlink":
        raise ValueError(f"Unsupported copy mode: {mode}")
    try:
        os.symlink(src, dst)
    except OSError:
        logging.getLogger(LOGGER_NAME).warning(
            "Symlink failed for %s; falling back to copy.", src
        )
        shutil.copy2(src, dst)


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
