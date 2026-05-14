from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_DATASET_ID = "ConstantHao/EDU-CHEMC_MM23"
DEFAULT_TARGET_FIELD = "ssml_graph_norm"
DEFAULT_DATASET_DIR = Path("data/processed/edu_chemc_graph_norm")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/runs/edu_chemc_texteller_graph_norm_full_model_bf16_30ep"
)
DEFAULT_EVAL_CSV = Path("outputs/eval_graph_norm_full_model_bf16_30ep_test_greedy.csv")
EXPECTED_HF_FIELDS = {
    "image",
    "chemfig",
    "ssml_sd",
    "ssml_normed",
    "ssml_rcgd",
    "image_path",
}
REPORT_ORDER = ("fail", "warn", "ok", "info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check input/output contracts for the EDU-CHEMC TexTeller pipeline "
            "without running long train/evaluate jobs."
        )
    )
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target_field", type=str, default=DEFAULT_TARGET_FIELD)
    parser.add_argument("--config", type=Path, default=Path("configs/train_edu_chemc.yaml"))
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="OleehyO/TexTeller")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval_output_csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=Path("external/GraphMatchingTool"))
    parser.add_argument("--graph_num_workers", type=int, default=8)
    parser.add_argument(
        "--graph_label_key",
        type=str,
        default=None,
        help="Graph label key for final evaluate; defaults to --target_field like the launcher.",
    )
    parser.add_argument(
        "--prediction_normalizer",
        type=str,
        default=None,
        help="Optional prediction normalizer to mirror evaluate.py.",
    )
    parser.add_argument(
        "--max_rows_per_split",
        type=int,
        default=1000,
        help="Rows to scan per metadata split. Use --scan_all for a full scan.",
    )
    parser.add_argument(
        "--sample_images",
        type=int,
        default=25,
        help="Maximum images per split to open with Pillow. Existence is checked for scanned rows.",
    )
    parser.add_argument("--scan_all", action="store_true", help="Scan every metadata row.")
    parser.add_argument(
        "--hf_smoke",
        action="store_true",
        help="Use datasets streaming to inspect one Hugging Face row. This may need network access.",
    )
    parser.add_argument("--hf_smoke_split", type=str, default="train")
    parser.add_argument("--hf_smoke_samples", type=int, default=1)
    parser.add_argument(
        "--check_tokenizer_load",
        action="store_true",
        help=(
            "Try loading the tokenizer/processor path with transformers. "
            "This may touch the Hugging Face cache or network."
        ),
    )
    parser.add_argument(
        "--out_report",
        type=Path,
        default=Path("outputs/reports/pipeline_io_check.json"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when blocking failures are found.",
    )
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")


def whitespace_tokenize(text: str) -> list[str]:
    return text.strip().split()


def qjoin(parts: list[str]) -> str:
    return shlex.join(parts)


def status_item(status: str, message: str, **details: Any) -> dict[str, Any]:
    return {"status": status, "message": message, **details}


def add(report: dict[str, Any], section: str, status: str, message: str, **details: Any) -> None:
    report.setdefault("checks", {}).setdefault(section, []).append(
        status_item(status, message, **details)
    )


def lookup_nested(row: dict[str, Any], key: str) -> Any:
    if not key:
        return None
    if key in row:
        return row[key]
    if key.startswith("targets."):
        targets = row.get("targets")
        if not isinstance(targets, dict):
            return None
        return targets.get(key.split(".", 1)[1])
    if "." not in key:
        targets = row.get("targets")
        if isinstance(targets, dict) and key in targets:
            return targets[key]
        return row.get(key)
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def target_key_name(key: str) -> str:
    return key.split(".", 1)[1] if key.startswith("targets.") else key


def resolve_data_target_key(config: dict[str, Any], split: str) -> str:
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        return "target"
    if split == "train":
        return str(data_cfg.get("train_target_key", data_cfg.get("target_key", "target")))
    if split in {"validation", "eval"}:
        return str(
            data_cfg.get(
                "eval_target_key",
                data_cfg.get("validation_target_key", data_cfg.get("target_key", "target")),
            )
        )
    return str(data_cfg.get("target_key", "target"))


def resolve_eval_target_key(config: dict[str, Any]) -> str:
    data_cfg = config.get("data", {})
    if isinstance(data_cfg, dict):
        return str(
            data_cfg.get(
                "eval_target_key",
                data_cfg.get("validation_target_key", data_cfg.get("target_key", "target")),
            )
        )
    return "target"


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return float(ordered[low])
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def describe_lengths(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def iter_jsonl_limited(path: Path, limit: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if limit is not None and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not a JSON object")
            yield line_no, row


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def split_dir_for(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / split


def resolve_image_path(split_dir: Path, row: dict[str, Any]) -> Path | None:
    file_name = row.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    path = Path(file_name)
    if not path.is_absolute():
        path = split_dir / path
    return path


def check_imports(report: dict[str, Any]) -> None:
    imports = ["yaml", "PIL", "datasets", "transformers", "accelerate", "torch"]
    for module in imports:
        try:
            imported = __import__(module)
        except Exception as exc:
            add(report, "environment", "fail", f"Cannot import {module}.", error=str(exc))
            continue
        version = getattr(imported, "__version__", None)
        add(report, "environment", "ok", f"Import available: {module}.", version=version)


def check_torch(report: dict[str, Any], args: argparse.Namespace, config: dict[str, Any]) -> None:
    try:
        import torch
    except Exception:
        return

    cuda_ok = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    devices: list[str] = []
    if cuda_ok:
        for idx in range(torch.cuda.device_count()):
            try:
                devices.append(torch.cuda.get_device_name(idx))
            except Exception:
                devices.append(f"cuda:{idx}")
    add(
        report,
        "environment",
        "ok" if cuda_ok else "warn",
        "CUDA status checked.",
        cuda_available=cuda_ok,
        cuda_version=cuda_version,
        devices=devices,
    )

    training_cfg = config.get("training", {})
    needs_bf16 = bool(isinstance(training_cfg, dict) and training_cfg.get("bf16"))
    if args.mixed_precision == "bf16" or args.dtype == "bf16" or needs_bf16:
        bf16_supported = False
        if cuda_ok:
            try:
                bf16_supported = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16_supported = False
        add(
            report,
            "environment",
            "ok" if bf16_supported else "warn",
            "bf16 capability checked.",
            bf16_requested=True,
            bf16_supported=bf16_supported,
        )


def check_config(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config
    if not config_path.is_file():
        add(report, "config", "fail", "Training config is missing.", path=rel(config_path))
        return {}

    config = load_yaml(config_path)
    add(report, "config", "ok", "Training config loaded.", path=rel(config_path))

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    eval_cfg = config.get("eval_metrics", {})
    if not isinstance(data_cfg, dict):
        add(report, "config", "fail", "Config data section must be a mapping.")
        data_cfg = {}
    if not isinstance(training_cfg, dict):
        add(report, "config", "fail", "Config training section must be a mapping.")
        training_cfg = {}
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}

    expected_key = f"targets.{args.target_field}"
    target_keys = {
        "target_key": data_cfg.get("target_key"),
        "train_target_key": data_cfg.get("train_target_key"),
        "eval_target_key": data_cfg.get("eval_target_key"),
    }
    mismatched = {
        key: value
        for key, value in target_keys.items()
        if value is not None and str(value) != expected_key
    }
    add(
        report,
        "config",
        "warn" if mismatched else "ok",
        "Config target keys checked.",
        expected=expected_key,
        actual=target_keys,
        mismatched=mismatched,
    )

    max_target_length = int(config.get("max_target_length", 0) or 0)
    add(
        report,
        "config",
        "ok" if max_target_length >= args.max_new_tokens else "warn",
        "Target length settings checked.",
        max_target_length=max_target_length,
        max_new_tokens=args.max_new_tokens,
        target_length_policy=data_cfg.get("target_length_policy"),
    )

    metric_for_best = training_cfg.get("metric_for_best_model")
    graph_eval_enabled = bool(eval_cfg.get("graph_eval", False))
    graph_label_key = eval_cfg.get("graph_label_key")
    add(
        report,
        "config",
        "ok",
        "Training/eval metric settings recorded.",
        metric_for_best_model=metric_for_best,
        graph_eval=graph_eval_enabled,
        eval_metrics_graph_label_key=graph_label_key,
        final_evaluate_graph_label_key=args.graph_label_key or args.target_field,
    )
    if graph_eval_enabled and metric_for_best == "eval_graph_em" and not graph_label_key:
        add(
            report,
            "config",
            "fail",
            "Config selects eval_graph_em as best metric but has no eval_metrics.graph_label_key.",
        )
    return config


def check_graph_tool(report: dict[str, Any], args: argparse.Namespace, config: dict[str, Any]) -> None:
    eval_cfg = config.get("eval_metrics", {})
    graph_needed = bool(isinstance(eval_cfg, dict) and eval_cfg.get("graph_eval", True))
    eval_py = args.graph_matching_tool_dir / "eval.py"
    status = "ok" if eval_py.is_file() else ("fail" if graph_needed else "warn")
    add(
        report,
        "graph_matching_tool",
        status,
        "GraphMatchingTool eval.py checked.",
        path=rel(eval_py),
        graph_needed=graph_needed,
    )


def check_hf_smoke(report: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.hf_smoke:
        add(
            report,
            "download",
            "info",
            "HF smoke check skipped. Pass --hf_smoke to inspect one streamed row.",
            dataset_id=args.dataset_id,
        )
        return
    try:
        from datasets import load_dataset
    except Exception as exc:
        add(report, "download", "fail", "Cannot import datasets for HF smoke check.", error=str(exc))
        return

    kwargs: dict[str, Any] = {"streaming": True}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    if args.revision:
        kwargs["revision"] = args.revision
    if args.cache_dir:
        kwargs["cache_dir"] = str(args.cache_dir)
    try:
        dataset = load_dataset(args.dataset_id, split=args.hf_smoke_split, **kwargs)
        rows = []
        iterator = iter(dataset)
        for _ in range(max(1, args.hf_smoke_samples)):
            rows.append(next(iterator))
    except Exception as exc:
        add(report, "download", "fail", "HF smoke check failed.", error=str(exc))
        return

    keys = set(rows[0].keys()) if rows else set()
    missing = sorted(EXPECTED_HF_FIELDS - keys)
    add(
        report,
        "download",
        "ok" if not missing else "fail",
        "HF streamed row schema checked.",
        dataset_id=args.dataset_id,
        split=args.hf_smoke_split,
        samples=len(rows),
        keys=sorted(keys),
        missing_expected_fields=missing,
    )


def check_dataset_split(
    report: dict[str, Any],
    args: argparse.Namespace,
    config: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    split_dir = split_dir_for(args.dataset_dir, split)
    metadata_path = split_dir / "metadata.jsonl"
    target_key = resolve_data_target_key(config, split)
    graph_eval_key = (
        config.get("eval_metrics", {}).get("graph_label_key")
        if isinstance(config.get("eval_metrics", {}), dict)
        else None
    )
    final_graph_key = args.graph_label_key or args.target_field
    keys_to_check = [target_key, final_graph_key]
    if isinstance(graph_eval_key, str) and graph_eval_key:
        keys_to_check.append(graph_eval_key)
    keys_to_check = list(dict.fromkeys(keys_to_check))

    split_report: dict[str, Any] = {
        "split": split,
        "split_dir": rel(split_dir),
        "metadata_path": rel(metadata_path),
        "target_key": target_key,
        "keys_checked": keys_to_check,
    }
    if not metadata_path.is_file():
        add(
            report,
            "prepare",
            "fail",
            f"{split} metadata is missing.",
            metadata_path=rel(metadata_path),
            expected_after="scripts/materialize_hf_edu_chemc.py",
        )
        split_report["exists"] = False
        return split_report

    add(report, "prepare", "ok", f"{split} metadata exists.", metadata_path=rel(metadata_path))
    scan_limit = None if args.scan_all else args.max_rows_per_split
    total_rows = count_jsonl(metadata_path) if args.scan_all else None
    if total_rows is None:
        try:
            total_rows = count_jsonl(metadata_path)
        except Exception:
            total_rows = None

    row_count = 0
    missing_by_key: Counter[str] = Counter()
    empty_by_key: Counter[str] = Counter()
    target_field_counts: Counter[str] = Counter()
    missing_file_name = 0
    missing_images = 0
    unreadable_images = 0
    images_checked = 0
    image_modes: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    target_lengths: list[int] = []
    first_row_keys: list[str] = []
    examples: dict[str, list[dict[str, Any]]] = {
        "missing_keys": [],
        "missing_images": [],
        "unreadable_images": [],
    }

    try:
        rows_iter = iter_jsonl_limited(metadata_path, scan_limit)
        for line_no, row in rows_iter:
            row_count += 1
            if row_count == 1:
                first_row_keys = sorted(str(key) for key in row.keys())
            target_field = row.get("target_field")
            if isinstance(target_field, str) and target_field:
                target_field_counts[target_field] += 1
            else:
                target_field_counts["<missing>"] += 1

            for key in keys_to_check:
                value = lookup_nested(row, key)
                if value is None:
                    missing_by_key[key] += 1
                    if len(examples["missing_keys"]) < 5:
                        examples["missing_keys"].append(
                            {"line": line_no, "key": key, "image_name": row.get("image_name")}
                        )
                    continue
                if not isinstance(value, str) or not value.strip():
                    empty_by_key[key] += 1
                    continue
                if key == target_key:
                    target_lengths.append(len(whitespace_tokenize(value)))

            image_path = resolve_image_path(split_dir, row)
            if image_path is None:
                missing_file_name += 1
                continue
            if not image_path.is_file():
                missing_images += 1
                if len(examples["missing_images"]) < 5:
                    examples["missing_images"].append(
                        {"line": line_no, "path": str(image_path), "image_name": row.get("image_name")}
                    )
                continue
            if images_checked < args.sample_images:
                images_checked += 1
                try:
                    from PIL import Image

                    with Image.open(image_path) as image:
                        image_modes[str(image.mode)] += 1
                        image_sizes[f"{image.width}x{image.height}"] += 1
                except Exception as exc:
                    unreadable_images += 1
                    if len(examples["unreadable_images"]) < 5:
                        examples["unreadable_images"].append(
                            {
                                "line": line_no,
                                "path": str(image_path),
                                "error": str(exc),
                            }
                        )
    except Exception as exc:
        add(report, "prepare", "fail", f"{split} metadata scan failed.", error=str(exc))
        split_report["scan_error"] = str(exc)
        return split_report

    length_stats = describe_lengths(target_lengths)
    max_target_length = int(config.get("max_target_length", args.max_new_tokens) or args.max_new_tokens)
    too_long_approx = sum(length > max_target_length for length in target_lengths)
    split_report.update(
        {
            "exists": True,
            "rows_scanned": row_count,
            "total_rows": total_rows,
            "scan_limit": scan_limit,
            "first_row_keys": first_row_keys,
            "target_field_counts": dict(target_field_counts),
            "missing_by_key": dict(missing_by_key),
            "empty_by_key": dict(empty_by_key),
            "missing_file_name": missing_file_name,
            "missing_images": missing_images,
            "images_checked": images_checked,
            "unreadable_images": unreadable_images,
            "image_modes_sample": dict(image_modes),
            "image_sizes_sample": dict(image_sizes),
            "target_length_whitespace": length_stats,
            "too_long_by_whitespace": too_long_approx,
            "examples": examples,
        }
    )

    if row_count == 0:
        add(report, "prepare", "fail", f"{split} metadata has no rows.", metadata_path=rel(metadata_path))
    if missing_by_key or empty_by_key:
        add(
            report,
            "prepare",
            "fail",
            f"{split} metadata is missing required target keys.",
            missing_by_key=dict(missing_by_key),
            empty_by_key=dict(empty_by_key),
            examples=examples["missing_keys"],
        )
    else:
        add(
            report,
            "prepare",
            "ok",
            f"{split} required target keys exist in scanned rows.",
            keys=keys_to_check,
            rows_scanned=row_count,
        )
    if missing_images or missing_file_name or unreadable_images:
        add(
            report,
            "prepare",
            "fail",
            f"{split} image references are not ready.",
            missing_file_name=missing_file_name,
            missing_images=missing_images,
            unreadable_images=unreadable_images,
            examples=examples,
        )
    else:
        add(
            report,
            "prepare",
            "ok",
            f"{split} image references look usable.",
            rows_scanned=row_count,
            images_opened=images_checked,
            image_modes_sample=dict(image_modes),
            image_sizes_sample=dict(image_sizes),
        )
    if too_long_approx:
        add(
            report,
            "prepare",
            "warn",
            f"{split} has labels above max_target_length by whitespace approximation.",
            count=too_long_approx,
            max_target_length=max_target_length,
            stats=length_stats,
        )
    else:
        add(
            report,
            "prepare",
            "ok",
            f"{split} target lengths are within max_target_length by whitespace approximation.",
            max_target_length=max_target_length,
            stats=length_stats,
        )
    return split_report


def check_dataset(report: dict[str, Any], args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not args.dataset_dir.exists():
        add(
            report,
            "prepare",
            "fail",
            "Prepared dataset directory is missing.",
            dataset_dir=rel(args.dataset_dir),
        )
        report["dataset"] = {"dataset_dir": rel(args.dataset_dir), "exists": False, "splits": []}
        return
    add(report, "prepare", "ok", "Prepared dataset directory exists.", dataset_dir=rel(args.dataset_dir))
    split_reports = [
        check_dataset_split(report, args, config, split)
        for split in args.splits
    ]
    report["dataset"] = {
        "dataset_dir": rel(args.dataset_dir),
        "exists": True,
        "splits": split_reports,
    }


def check_tokenizer_load(report: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.check_tokenizer_load:
        add(
            report,
            "model",
            "info",
            "Tokenizer/model smoke check skipped. Pass --check_tokenizer_load to try it.",
        )
        return
    source = args.tokenizer_path or args.pretrained_model_name_or_path
    try:
        from transformers import AutoProcessor, AutoTokenizer
    except Exception as exc:
        add(report, "model", "fail", "Cannot import transformers tokenizer classes.", error=str(exc))
        return
    tokenizer_ok = False
    processor_ok = False
    errors: dict[str, str] = {}
    try:
        AutoTokenizer.from_pretrained(source)
        tokenizer_ok = True
    except Exception as exc:
        errors["AutoTokenizer"] = str(exc)
    try:
        AutoProcessor.from_pretrained(source)
        processor_ok = True
    except Exception as exc:
        errors["AutoProcessor"] = str(exc)
    add(
        report,
        "model",
        "ok" if tokenizer_ok or processor_ok else "fail",
        "Tokenizer/processor load smoke checked.",
        source=source,
        tokenizer_ok=tokenizer_ok,
        processor_ok=processor_ok,
        errors=errors,
    )


def check_train_outputs(report: dict[str, Any], args: argparse.Namespace) -> None:
    best_dir = args.output_dir / "best"
    checkpoint_dirs = sorted(args.output_dir.glob("checkpoint-*")) if args.output_dir.exists() else []
    status = "ok" if best_dir.is_dir() else "info"
    add(
        report,
        "train",
        status,
        "Train output checkpoint state checked.",
        output_dir=rel(args.output_dir),
        best_dir=rel(best_dir),
        best_exists=best_dir.is_dir(),
        checkpoint_dirs=[rel(path) for path in checkpoint_dirs[-5:]],
        evaluate_ready=best_dir.is_dir(),
    )


def check_eval_outputs(report: dict[str, Any], args: argparse.Namespace) -> None:
    metrics_path = args.eval_output_csv.with_suffix(".metrics.json")
    graph_result = args.eval_output_csv.with_suffix(".graph_result.txt")
    add(
        report,
        "evaluate",
        "info" if not args.eval_output_csv.exists() else "ok",
        "Evaluation output state checked.",
        output_csv=rel(args.eval_output_csv),
        output_csv_exists=args.eval_output_csv.exists(),
        metrics_json=rel(metrics_path),
        metrics_json_exists=metrics_path.exists(),
        graph_result=rel(graph_result),
        graph_result_exists=graph_result.exists(),
    )


def command_map(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, str]:
    final_graph_key = args.graph_label_key or args.target_field
    prepare_cmd = [
        "uv",
        "run",
        "python",
        "scripts/materialize_hf_edu_chemc.py",
        "--dataset_id",
        args.dataset_id,
        "--out_dir",
        str(args.dataset_dir),
        "--target_field",
        args.target_field,
    ]
    if args.dataset_config:
        prepare_cmd.extend(["--dataset_config", args.dataset_config])
    if args.revision:
        prepare_cmd.extend(["--revision", args.revision])
    if args.cache_dir:
        prepare_cmd.extend(["--cache_dir", str(args.cache_dir)])

    validate_cmd = [
        "uv",
        "run",
        "python",
        "scripts/validate_graph_norm.py",
        "--dataset_dir",
        str(args.dataset_dir),
        "--splits",
        *args.splits,
        "--source_key",
        "ssml_normed",
        "--normalized_key",
        args.target_field,
        "--graph_matching_tool_dir",
        str(args.graph_matching_tool_dir),
        "--graph_num_workers",
        str(args.graph_num_workers),
    ]
    train_cmd = [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--num_machines",
        str(args.num_machines),
        "--mixed_precision",
        args.mixed_precision,
        "--dynamo_backend",
        "no",
    ]
    if args.num_processes is not None:
        train_cmd.extend(["--num_processes", str(args.num_processes)])
    train_cmd.extend(
        [
            "scripts/train.py",
            "--config",
            str(args.config),
            "--dataset_dir",
            str(args.dataset_dir),
            "--pretrained_model_name_or_path",
            args.pretrained_model_name_or_path,
            "--output_dir",
            str(args.output_dir),
        ]
    )
    if args.tokenizer_path:
        train_cmd.extend(["--tokenizer_path", args.tokenizer_path])
    evaluate_cmd = [
        "uv",
        "run",
        "python",
        "scripts/evaluate.py",
        "--model_ckpt",
        str(args.output_dir / "best"),
        "--dataset_dir",
        str(args.dataset_dir),
        "--split",
        args.eval_split,
        "--batch_size",
        str(args.eval_batch_size),
        "--num_beams",
        str(args.num_beams),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--output_csv",
        str(args.eval_output_csv),
        "--graph_label_key",
        final_graph_key,
        "--graph_eval",
        "--graph_matching_tool_dir",
        str(args.graph_matching_tool_dir),
        "--graph_num_workers",
        str(args.graph_num_workers),
    ]
    eval_target_key = resolve_eval_target_key(config)
    if eval_target_key:
        evaluate_cmd.extend(["--target_key", eval_target_key])
    if args.prediction_normalizer:
        evaluate_cmd.extend(["--prediction_normalizer", args.prediction_normalizer])
    if args.tokenizer_path:
        evaluate_cmd.extend(["--tokenizer_path", args.tokenizer_path])

    launcher_cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_edu_chemc_pipeline.py",
        "--stages",
        "train_eval",
        "--config",
        str(args.config),
        "--dataset_dir",
        str(args.dataset_dir),
        "--target_field",
        args.target_field,
        "--graph_matching_tool_dir",
        str(args.graph_matching_tool_dir),
        "--output_dir",
        str(args.output_dir),
        "--eval_output_csv",
        str(args.eval_output_csv),
    ]
    if args.cuda_visible_devices:
        launcher_cmd.extend(["--cuda_visible_devices", args.cuda_visible_devices])
    if args.num_processes is not None:
        launcher_cmd.extend(["--num_processes", str(args.num_processes)])
    if args.graph_label_key:
        launcher_cmd.extend(["--graph_label_key", args.graph_label_key])

    prefix = ""
    if args.cuda_visible_devices:
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)} "
    return {
        "prepare": qjoin(prepare_cmd),
        "validate_graph_norm": qjoin(validate_cmd),
        "analyze_targets": qjoin(
            [
                "uv",
                "run",
                "python",
                "scripts/analyze_targets.py",
                "--metadata",
                str(args.dataset_dir / "train" / "metadata.jsonl"),
            ]
        ),
        "analyze_tokenizer_coverage": qjoin(
            [
                "uv",
                "run",
                "python",
                "scripts/analyze_tokenizer_coverage.py",
                "--metadata",
                str(args.dataset_dir / "train" / "metadata.jsonl"),
                "--pretrained_model_name_or_path",
                args.pretrained_model_name_or_path,
                "--max_decoder_length",
                str(args.max_new_tokens),
            ]
        ),
        "train": prefix + qjoin(train_cmd),
        "evaluate": prefix + qjoin(evaluate_cmd),
        "launcher_train_eval": qjoin(launcher_cmd),
    }


def build_io_contract(
    args: argparse.Namespace,
    config: dict[str, Any],
    commands: dict[str, str],
) -> dict[str, dict[str, Any]]:
    train_target_key = resolve_data_target_key(config, "train")
    eval_target_key = resolve_eval_target_key(config)
    final_graph_key = args.graph_label_key or args.target_field
    train_metadata = args.dataset_dir / "train" / "metadata.jsonl"
    eval_metadata = args.dataset_dir / args.eval_split / "metadata.jsonl"
    best_dir = args.output_dir / "best"
    return {
        "download": {
            "inputs": [
                f"HF dataset id: {args.dataset_id}",
                "Network/cache access unless the dataset is already cached",
            ],
            "outputs": [
                "Hugging Face dataset cache entries",
                "No repo metadata.jsonl is created by download alone",
            ],
            "ready_when": "HF smoke can read expected fields or prepare can load the dataset.",
            "check_command": "uv run python scripts/check_pipeline_io.py --hf_smoke",
        },
        "prepare": {
            "inputs": [
                f"HF dataset id/cache: {args.dataset_id}",
                f"target_field: {args.target_field}",
            ],
            "outputs": [
                str(args.dataset_dir / "train" / "metadata.jsonl"),
                str(args.dataset_dir / "validation" / "metadata.jsonl"),
                str(args.dataset_dir / "test" / "metadata.jsonl"),
                f"Each row contains {train_target_key} and image file references",
            ],
            "ready_when": "All required split metadata files, target keys, and referenced images pass this check.",
            "command": commands["prepare"],
        },
        "validate_graph_norm": {
            "inputs": [
                str(args.dataset_dir),
                "targets.ssml_normed",
                f"targets.{args.target_field}",
                str(args.graph_matching_tool_dir / "eval.py"),
            ],
            "outputs": [
                "outputs/reports/graph_norm_validation/summary.json",
                "Per-split GraphMatchingTool result text files",
            ],
            "ready_when": "graph_em and graph_structure_em are 1.0 on train/validation/test.",
            "command": commands["validate_graph_norm"],
        },
        "analyze": {
            "inputs": [
                str(train_metadata),
                f"pretrained_model_name_or_path: {args.pretrained_model_name_or_path}",
            ],
            "outputs": [
                "Console/tokenizer coverage statistics",
                "No training checkpoint is created",
            ],
            "ready_when": "No missing tokenizer/model errors and no unacceptable length/coverage issue.",
            "commands": [commands["analyze_targets"], commands["analyze_tokenizer_coverage"]],
        },
        "train": {
            "inputs": [
                str(args.config),
                str(args.dataset_dir),
                f"train target key: {train_target_key}",
                f"base model: {args.pretrained_model_name_or_path}",
                "CUDA device with requested precision",
            ],
            "outputs": [
                str(args.output_dir),
                str(best_dir),
                "logs/<run_name>_<timestamp>.log",
                "logs/<run_name>_<timestamp>.trainer_events.jsonl",
            ],
            "ready_when": f"{best_dir} exists and contains a loadable model/tokenizer checkpoint.",
            "command": commands["train"],
        },
        "evaluate": {
            "inputs": [
                str(best_dir),
                str(eval_metadata),
                f"sequence target key: {eval_target_key}",
                f"graph label key: {final_graph_key}",
                str(args.graph_matching_tool_dir / "eval.py"),
            ],
            "outputs": [
                str(args.eval_output_csv),
                str(args.eval_output_csv.with_suffix(".metrics.json")),
                str(args.eval_output_csv.with_suffix(".graph_result.txt")),
            ],
            "ready_when": "CSV, metrics JSON, and graph result exist; graph_total matches evaluated sample count.",
            "command": commands["evaluate"],
        },
    }


def summarize_counts(report: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in REPORT_ORDER}
    for items in report.get("checks", {}).values():
        for item in items:
            status = str(item.get("status", "info"))
            counts[status] = counts.get(status, 0) + 1
    return counts


def print_report(report: dict[str, Any]) -> None:
    counts = report.get("summary", {})
    print("Pipeline IO check")
    print(f"  fails={counts.get('fail', 0)} warns={counts.get('warn', 0)} ok={counts.get('ok', 0)}")
    print()
    for section, items in report.get("checks", {}).items():
        print(f"[{section}]")
        for item in items:
            status = str(item.get("status", "info")).upper()
            print(f"  {status}: {item.get('message')}")
            detail_keys = [
                key
                for key in item.keys()
                if key not in {"status", "message"} and item.get(key) not in ({}, [], None)
            ]
            for key in detail_keys[:4]:
                value = item[key]
                if isinstance(value, (dict, list)):
                    value_text = json.dumps(value, ensure_ascii=False, default=str)
                else:
                    value_text = str(value)
                if len(value_text) > 220:
                    value_text = value_text[:217] + "..."
                print(f"    {key}: {value_text}")
        print()
    print("[io_contract]")
    for stage, contract in report.get("io_contract", {}).items():
        print(f"  {stage}:")
        for key in ("inputs", "outputs", "ready_when"):
            value = contract.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                value_text = "; ".join(str(item) for item in value)
            else:
                value_text = str(value)
            if len(value_text) > 300:
                value_text = value_text[:297] + "..."
            print(f"    {key}: {value_text}")
    print()
    print("[commands]")
    for name, cmd in report.get("commands", {}).items():
        print(f"  {name}:")
        print(f"    {cmd}")


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "project_root": str(PROJECT_ROOT),
        "checks": {},
    }

    config = check_config(report, args)
    check_imports(report)
    check_torch(report, args, config)
    check_graph_tool(report, args, config)
    check_hf_smoke(report, args)
    check_dataset(report, args, config)
    check_tokenizer_load(report, args)
    check_train_outputs(report, args)
    check_eval_outputs(report, args)
    commands = command_map(args, config)
    report["commands"] = commands
    report["io_contract"] = build_io_contract(args, config, commands)
    report["summary"] = summarize_counts(report)

    save_json(report, args.out_report)
    print_report(report)
    print()
    print(f"Wrote JSON report: {args.out_report}")

    if args.strict and report["summary"].get("fail", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
