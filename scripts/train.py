from __future__ import annotations

import argparse
import contextlib
import gc
import inspect
import json
import math
import os
import random
import shutil
import sys
import warnings
from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

warnings.filterwarnings("ignore", message=r".*warmup_ratio is deprecated.*")
warnings.filterwarnings("ignore", message=r".*find_unused_parameters=True.*")

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.utils import logging as transformers_logging
from transformers.trainer_callback import PrinterCallback, ProgressCallback, TrainerCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.data import EduChemcDataset, VisionSeq2SeqCollator
from chemtexteller.graph_matching_eval import (
    lookup_target,
    run_graph_matching_tool,
    validate_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.inference import (
    autocast_context,
    generate_from_pixel_values,
    generation_kwargs,
    move_pixel_values,
    set_generation_cache,
)
from chemtexteller.metrics import sequence_metrics
from chemtexteller.model_loader import (
    AddedTokenInit,
    add_texteller_repo_to_path,
    enable_gradient_checkpointing_if_available,
    freeze_encoder_if_available,
    initialize_new_token_embeddings_from_subtokens,
    load_pretrained_model_and_tokenizer,
    resize_token_embeddings_if_needed,
)
from chemtexteller.rfl_msd_loss import RflMsdBranchClassifier, RflMsdLoss, sequence_cross_entropy
from chemtexteller.tokenizer_utils import (
    add_chemical_tokens,
    extract_chemical_tokens_from_sources,
)
from chemtexteller.transforms import build_transform
from chemtexteller.utils import ensure_dir, load_yaml, save_json, save_yaml, set_seed, setup_logging


logger = setup_logging()
transformers_logging.set_verbosity_error()


def cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


DEFAULT_LORA_TARGET_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "query",
    "key",
    "value",
    "dense",
    "fc1",
    "fc2",
)

EXCLUDED_LORA_TARGET_LEAVES = (
    "embed_tokens",
    "lm_head",
    "output_projection",
    "classifier",
)


class StableTqdmProgressCallback(TrainerCallback):
    """Progress bars tuned for terminals/log viewers that duplicate fast tqdm redraws."""

    def __init__(
        self,
        mininterval: float = 5.0,
        miniters: int = 10,
        ncols: int = 100,
        ascii_bar: bool = False,
        max_str_len: int = 100,
        by_epoch: bool = True,
        write_logs: bool = False,
    ) -> None:
        self.mininterval = mininterval
        self.miniters = miniters
        self.ncols = ncols
        self.ascii_bar = ascii_bar
        self.max_str_len = max_str_len
        self.by_epoch = by_epoch
        self.write_logs = write_logs
        self.training_bar = None
        self.prediction_bar = None
        self.current_step = 0
        self.steps_per_epoch = 0

    def _bar(self, total: int, leave: bool, desc: str | None = None) -> tqdm:
        return tqdm(
            total=total,
            desc=desc,
            leave=leave,
            dynamic_ncols=False,
            ncols=self.ncols,
            mininterval=self.mininterval,
            miniters=self.miniters,
            ascii=self.ascii_bar,
            file=sys.stdout,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        self.steps_per_epoch = max(1, math.ceil(state.max_steps / float(args.num_train_epochs)))
        if state.is_world_process_zero and not self.by_epoch:
            self.training_bar = self._bar(total=state.max_steps, leave=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.by_epoch:
            return
        self._close_training_bar(state)
        self.current_step = state.global_step
        remaining_steps = max(1, state.max_steps - state.global_step)
        total = min(self.steps_per_epoch or remaining_steps, remaining_steps)
        current_epoch = min(
            int(state.global_step // max(1, self.steps_per_epoch)) + 1,
            int(math.ceil(args.num_train_epochs)),
        )
        desc = f"Epoch {current_epoch}/{int(math.ceil(args.num_train_epochs))}"
        self.training_bar = self._bar(total=total, leave=True, desc=desc)

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.by_epoch and self.training_bar is None:
            self.on_epoch_begin(args, state, control, **kwargs)
        if self.training_bar is None:
            return
        self.training_bar.update(state.global_step - self.current_step)
        self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not state.is_world_process_zero or eval_dataloader is None:
            return
        try:
            total = len(eval_dataloader)
        except TypeError:
            return
        if self.prediction_bar is None:
            self.prediction_bar = self._bar(total=total, leave=False, desc="Eval")
        self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        self._close_prediction_bar(state)

    def on_predict(self, args, state, control, **kwargs):
        self._close_prediction_bar(state)

    def _close_prediction_bar(self, state) -> None:
        if state.is_world_process_zero and self.prediction_bar is not None:
            self.prediction_bar.close()
            self.prediction_bar = None
            sys.stdout.write("\n")
            sys.stdout.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        if any(key.startswith("eval_") for key in logs):
            self._close_prediction_bar(state)
        if not self.write_logs:
            return
        shallow_logs = {}
        for key, value in logs.items():
            if key == "total_flos":
                continue
            if isinstance(value, str) and len(value) > self.max_str_len:
                shallow_logs[key] = (
                    f"[String too long to display, length: {len(value)} > {self.max_str_len}]"
                )
            elif isinstance(value, float):
                shallow_logs[key] = f"{value:.4g}"
            else:
                shallow_logs[key] = value
        if self.training_bar is not None:
            self.training_bar.write(str(shallow_logs))
        else:
            tqdm.write(str(shallow_logs), file=sys.stdout)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.by_epoch:
            self._close_training_bar(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._close_training_bar(state)

    def _close_training_bar(self, state) -> None:
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None


class TrainingFileLogCallback(TrainerCallback):
    """Append train/eval events to a JSONL file as soon as Trainer emits them."""

    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path
        ensure_dir(jsonl_path.parent)
        self.epoch_started_at: float | None = None

    def _append_event(self, event: str, state, payload: dict[str, Any] | None = None) -> None:
        if not state.is_world_process_zero:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "step": state.global_step,
            "epoch": state.epoch,
        }
        if payload:
            record.update(payload)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, default=str)
            f.write("\n")
            f.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        self._append_event(
            "train_begin",
            state,
            {
                "max_steps": state.max_steps,
                "num_train_epochs": args.num_train_epochs,
                "output_dir": args.output_dir,
            },
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_started_at = perf_counter()
        self._append_event("epoch_begin", state)

    def on_epoch_end(self, args, state, control, **kwargs):
        payload: dict[str, Any] = {}
        if self.epoch_started_at is not None:
            payload["train_epoch_runtime"] = perf_counter() - self.epoch_started_at
        logger.info(
            "Train epoch finished | epoch=%.3f step=%s train_epoch_runtime=%s",
            state.epoch or 0.0,
            state.global_step,
            (
                f"{payload['train_epoch_runtime']:.1f}s"
                if "train_epoch_runtime" in payload
                else "unknown"
            ),
        )
        self._append_event("epoch_end", state, payload)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        clean_logs = {key: value for key, value in logs.items() if key != "total_flos"}
        self._append_event("log", state, clean_logs)
        if not state.is_world_process_zero:
            return
        if "loss" in clean_logs:
            logger.info(
                "Train log | epoch=%.3f step=%s loss=%.6g grad_norm=%s lr=%s",
                state.epoch or 0.0,
                state.global_step,
                float(clean_logs["loss"]),
                clean_logs.get("grad_norm"),
                clean_logs.get("learning_rate"),
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        clean_metrics = metrics or {}
        self._append_event("evaluate", state, clean_metrics)
        if not state.is_world_process_zero:
            return
        logger.info(
            "Eval finished | epoch=%.3f step=%s eval_loss=%s eval_runtime=%s",
            state.epoch or 0.0,
            state.global_step,
            (
                f"{float(clean_metrics['eval_loss']):.6g}"
                if "eval_loss" in clean_metrics
                else "unknown"
            ),
            (
                f"{float(clean_metrics['eval_runtime']):.1f}s"
                if "eval_runtime" in clean_metrics
                else "unknown"
            ),
        )

    def on_save(self, args, state, control, **kwargs):
        self._append_event("save", state, {"output_dir": args.output_dir})

    def on_train_end(self, args, state, control, **kwargs):
        self._append_event("train_end", state, {"best_metric": state.best_metric})


def _format_count(value: int) -> str:
    return f"{value:,}"


def log_trainable_parameter_summary(model: torch.nn.Module) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    logger.info(
        "Parameter count | trainable=%s total=%s trainable_pct=%.4f",
        _format_count(trainable),
        _format_count(total),
        pct,
    )


def _config_list(value: Any, default: Sequence[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise TypeError(f"Expected string/list config value, got {type(value).__name__}")


def _config_set(value: Any) -> set[str] | None:
    items = _config_list(value)
    return set(items) if items else None


def _config_int_map(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping config value, got {type(value).__name__}")
    return {str(key): int(item) for key, item in value.items()}


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _tokenizer_chemical_cfg(config: dict[str, Any]) -> dict[str, Any] | None:
    tokenizer_cfg = config.get("tokenizer", {})
    if not isinstance(tokenizer_cfg, dict):
        return None
    chemical_cfg = tokenizer_cfg.get("chemical_tokens", {})
    if not isinstance(chemical_cfg, dict) or not bool(chemical_cfg.get("enabled", False)):
        return None
    return chemical_cfg


def maybe_extend_tokenizer_with_chemical_tokens(
    tokenizer: Any,
    config: dict[str, Any],
    *,
    dataset_dir: Path,
    output_dir: Path,
) -> list[AddedTokenInit]:
    chemical_cfg = _tokenizer_chemical_cfg(config)
    if chemical_cfg is None:
        return []

    sources = [_resolve_project_path(path) for path in _config_list(chemical_cfg.get("sources"))]
    split_to_file = {
        "train": dataset_dir / "train" / "metadata.jsonl",
        "validation": dataset_dir / "validation" / "metadata.jsonl",
        "test": dataset_dir / "test" / "metadata.jsonl",
    }
    include_splits = _config_list(chemical_cfg.get("include_metadata_splits"))
    for split in include_splits:
        if split not in split_to_file:
            raise ValueError(
                "tokenizer.chemical_tokens.include_metadata_splits contains "
                f"unsupported split '{split}'. Expected one of {sorted(split_to_file)}."
            )
        sources.append(split_to_file[split])
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError(
            "tokenizer.chemical_tokens.enabled=true requires sources or include_metadata_splits."
        )

    csv_fields = _config_list(
        chemical_cfg.get("csv_fields"),
        default=("ground_truth", "graph_label", "prediction", "raw_prediction"),
    )
    metadata_target_keys = _config_list(
        chemical_cfg.get("metadata_target_keys"),
        default=("targets.ssml_graph_norm", "targets.ssml_normed"),
    )
    tokens, counts = extract_chemical_tokens_from_sources(
        sources,
        csv_fields=csv_fields,
        metadata_target_keys=metadata_target_keys,
        min_frequency=int(chemical_cfg.get("min_frequency", 10)),
        max_tokens=(
            None
            if chemical_cfg.get("max_tokens") in {None, 0, "0"}
            else int(chemical_cfg.get("max_tokens"))
        ),
        include_categories=_config_set(chemical_cfg.get("include_categories")),
        exclude_categories=_config_set(chemical_cfg.get("exclude_categories")),
        max_tokens_per_category=_config_int_map(
            chemical_cfg.get("max_tokens_per_category")
        ),
        max_rows_per_source=(
            None
            if chemical_cfg.get("max_rows_per_source") in {None, 0, "0"}
            else int(chemical_cfg.get("max_rows_per_source"))
        ),
        include_default_tokens=bool(chemical_cfg.get("include_default_tokens", True)),
    )
    explicit_tokens = _config_list(chemical_cfg.get("tokens"))
    selected_tokens = list(dict.fromkeys(explicit_tokens + tokens))
    old_vocab = tokenizer.get_vocab()
    source_ids_by_token: dict[str, tuple[int, ...]] = {}
    for token in selected_tokens:
        if token in old_vocab:
            continue
        encoded = tokenizer(token, add_special_tokens=False)
        ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if isinstance(ids, list):
            source_ids_by_token[token] = tuple(int(token_id) for token_id in ids)
    added = add_chemical_tokens(tokenizer, selected_tokens)
    new_vocab = tokenizer.get_vocab()
    init_specs = [
        AddedTokenInit(
            token=token,
            new_id=int(new_vocab[token]),
            source_ids=source_ids,
        )
        for token, source_ids in source_ids_by_token.items()
        if token in new_vocab and int(new_vocab[token]) >= len(old_vocab)
    ]

    ensure_dir(output_dir)
    token_path = output_dir / "chemical_tokens.selected.txt"
    count_path = output_dir / "chemical_tokens.counts.json"
    token_path.write_text("\n".join(selected_tokens) + "\n", encoding="utf-8")
    save_json(
        {
            "added": added,
            "selected": len(selected_tokens),
            "sources": [str(path) for path in sources],
            "min_frequency": int(chemical_cfg.get("min_frequency", 10)),
            "max_tokens": chemical_cfg.get("max_tokens"),
            "include_categories": chemical_cfg.get("include_categories"),
            "exclude_categories": chemical_cfg.get("exclude_categories"),
            "max_tokens_per_category": chemical_cfg.get("max_tokens_per_category"),
            "initialization": (
                "subtoken_average"
                if bool(chemical_cfg.get("initialize_from_subtokens", True))
                else "default_resize_init"
            ),
            "init_specs": [
                {
                    "token": spec.token,
                    "new_id": spec.new_id,
                    "source_ids": list(spec.source_ids),
                }
                for spec in init_specs
            ],
            "counts": {token: counts[token] for token in selected_tokens},
        },
        count_path,
    )
    logger.info(
        "Chemical tokenizer extension | selected=%s added=%s sources=%s token_file=%s",
        len(selected_tokens),
        added,
        [str(path) for path in sources],
        token_path,
    )
    if not bool(chemical_cfg.get("initialize_from_subtokens", True)):
        return []
    return init_specs


def _length_balanced_cfg(config: dict[str, Any]) -> dict[str, Any] | None:
    sampling_cfg = config.get("sampling", {})
    if not isinstance(sampling_cfg, dict):
        return None
    length_cfg = sampling_cfg.get("length_balanced", {})
    if not isinstance(length_cfg, dict) or not bool(length_cfg.get("enabled", False)):
        return None
    return length_cfg


def _target_texts(dataset: Any) -> list[str]:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError(
            "Length-balanced sampling requires a dataset with a 'samples' attribute. "
            "EduChemcDataset provides this by default."
        )
    return [sample.target for sample in samples]


def _target_token_lengths(dataset: Any, batch_size: int = 512) -> list[int]:
    if hasattr(dataset, "target_token_lengths"):
        return list(dataset.target_token_lengths(batch_size=batch_size))
    tokenizer = getattr(dataset, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("Length-balanced sampling requires the dataset to expose its tokenizer.")
    texts = _target_texts(dataset)
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )
        lengths.extend(len(input_ids) for input_ids in tokenized["input_ids"])
    return lengths


def _target_whitespace_lengths(dataset: Any) -> list[int]:
    return [len(text.split()) for text in _target_texts(dataset)]


def _target_lengths(dataset: Any, cfg: dict[str, Any]) -> list[int]:
    metric = str(cfg.get("length_metric", cfg.get("metric", "tokenized"))).lower()
    if metric in {"tokenized", "tokenizer", "subtoken"}:
        batch_size = int(cfg.get("length_batch_size", 512))
        return _target_token_lengths(dataset, batch_size=batch_size)
    if metric in {"whitespace", "target_tokens", "raw"}:
        return _target_whitespace_lengths(dataset)
    raise ValueError(
        "Unsupported sampling.length_balanced.length_metric="
        f"{metric!r}. Use 'tokenized' or 'whitespace'."
    )


def _as_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _length_bin_weight(length: int, bins: Sequence[dict[str, Any]], default_weight: float) -> float:
    for bin_cfg in bins:
        max_length = bin_cfg.get("max_length")
        if max_length in {None, "inf", "infinity", "*"}:
            return _as_float(bin_cfg.get("weight"), default=default_weight)
        if length <= int(max_length):
            return _as_float(bin_cfg.get("weight"), default=default_weight)
    return default_weight


def _build_length_balanced_weights(dataset: Any, cfg: dict[str, Any]) -> torch.DoubleTensor:
    bins = cfg.get("bins", cfg.get("length_bins"))
    if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes)) or not bins:
        bins = [
            {"max_length": 256, "weight": 1.0},
            {"max_length": 512, "weight": 1.5},
            {"max_length": 768, "weight": 3.0},
            {"max_length": None, "weight": 2.0},
        ]
    normalized_bins: list[dict[str, Any]] = []
    for idx, bin_cfg in enumerate(bins):
        if not isinstance(bin_cfg, dict):
            raise TypeError(f"sampling.length_balanced.bins[{idx}] must be a mapping.")
        normalized_bins.append(dict(bin_cfg))

    default_weight = _as_float(cfg.get("default_weight"), default=1.0)
    lengths = _target_lengths(dataset, cfg)
    weights = [
        _length_bin_weight(length, normalized_bins, default_weight=default_weight)
        for length in lengths
    ]
    counts: dict[str, int] = {}
    for length, weight in zip(lengths, weights, strict=True):
        key = f"w={weight:g}"
        counts[key] = counts.get(key, 0) + 1
    logger.info(
        "Length-balanced sampling enabled | metric=%s samples=%s min_len=%s p50_len=%s p95_len=%s max_len=%s weight_counts=%s",
        str(cfg.get("length_metric", cfg.get("metric", "tokenized"))),
        len(lengths),
        min(lengths) if lengths else 0,
        int(torch.tensor(lengths, dtype=torch.float32).quantile(0.50).item()) if lengths else 0,
        int(torch.tensor(lengths, dtype=torch.float32).quantile(0.95).item()) if lengths else 0,
        max(lengths) if lengths else 0,
        counts,
    )
    return torch.as_tensor(weights, dtype=torch.double)


def _resolve_inference_dtype_from_args(args: Seq2SeqTrainingArguments) -> torch.dtype | None:
    if not torch.cuda.is_available():
        return None
    if bool(getattr(args, "bf16", False)):
        return torch.bfloat16
    if bool(getattr(args, "fp16", False)):
        return torch.float16
    return None


class EvalGenerationMetricRunner:
    def __init__(
        self,
        *,
        dataset: EduChemcDataset,
        tokenizer: Any,
        output_dir: Path,
        cfg: dict[str, Any],
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.cfg = cfg
        self._graph_warning_emitted = False
        self._distributed_warning_emitted = False

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def compute(
        self,
        trainer: Seq2SeqTrainer,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float | int | str]:
        if not self.enabled:
            return {}
        if int(getattr(trainer.args, "world_size", 1)) != 1:
            if not self._distributed_warning_emitted and trainer.is_world_process_zero():
                logger.warning(
                    "Skipping eval generation metrics because world_size=%s. "
                    "Run with --num_processes 1 for EM/graph EM during training.",
                    trainer.args.world_size,
                )
                self._distributed_warning_emitted = True
            return {}
        if not trainer.is_world_process_zero():
            return {}

        max_samples = self.cfg.get("max_samples")
        indices = list(range(len(self.dataset)))
        sample_seed = self.cfg.get("sample_seed")
        if sample_seed is not None:
            rng = random.Random(int(sample_seed))
            rng.shuffle(indices)
        if max_samples is not None:
            indices = indices[: int(max_samples)]
        if not indices:
            return {}

        batch_size = int(
            self.cfg.get(
                "batch_size",
                getattr(trainer.args, "per_device_eval_batch_size", 4),
            )
        )
        num_workers = int(self.cfg.get("dataloader_num_workers", 0))
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "collate_fn": VisionSeq2SeqCollator(self.tokenizer, include_metadata=True),
            "num_workers": num_workers,
            "pin_memory": bool(self.cfg.get("pin_memory", trainer.args.dataloader_pin_memory)),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(
                self.cfg.get("dataloader_persistent_workers", False)
            )
        loader = DataLoader(Subset(self.dataset, indices), **loader_kwargs)

        model = trainer.model
        if hasattr(trainer, "accelerator"):
            with contextlib.suppress(Exception):
                model = trainer.accelerator.unwrap_model(model)
        was_training = model.training
        model.eval()
        set_generation_cache(model, enabled=True)

        device = trainer.args.device
        dtype = _resolve_inference_dtype_from_args(trainer.args)
        gen_kwargs = generation_kwargs(
            model=model,
            tokenizer=self.tokenizer,
            num_beams=int(self.cfg.get("num_beams", 1)),
            max_new_tokens=int(self.cfg.get("max_new_tokens", 1024)),
            use_cache=bool(self.cfg.get("use_cache", True)),
            length_penalty=self.cfg.get("length_penalty"),
            early_stopping=self.cfg.get("early_stopping"),
            min_new_tokens=self.cfg.get("min_new_tokens"),
            no_repeat_ngram_size=self.cfg.get("no_repeat_ngram_size"),
            repetition_penalty=self.cfg.get("repetition_penalty"),
        )

        predictions: list[str] = []
        references: list[str] = []
        graph_rows: list[dict[str, str]] = []
        graph_label_key = str(
            self.cfg.get(
                "graph_label_key",
                self.cfg.get("target_key", "target"),
            )
        )
        progress = tqdm(
            loader,
            desc="Eval generation",
            leave=False,
            dynamic_ncols=False,
            ncols=int(self.cfg.get("tqdm_ncols", 100)),
            disable=not bool(self.cfg.get("tqdm", True)),
            file=sys.stdout,
        )
        started = perf_counter()
        try:
            with torch.inference_mode(), autocast_context(device, dtype):
                for batch in progress:
                    pixel_values = move_pixel_values(batch["pixel_values"], device, dtype)
                    generated = generate_from_pixel_values(model, pixel_values, gen_kwargs)
                    decoded = self.tokenizer.batch_decode(
                        generated,
                        skip_special_tokens=True,
                    )
                    del generated
                    del pixel_values
                    for image_name, target, metadata_targets, prediction in zip(
                        batch["image_names"],
                        batch["targets"],
                        batch["metadata_targets"],
                        decoded,
                        strict=True,
                    ):
                        prediction = " ".join(str(prediction).split())
                        reference = " ".join(str(target).split())
                        predictions.append(prediction)
                        references.append(reference)
                        if bool(self.cfg.get("graph_eval", False)):
                            try:
                                graph_label = lookup_target(metadata_targets, graph_label_key)
                            except KeyError:
                                continue
                            graph_rows.append(
                                {
                                    "image_name": str(image_name),
                                    "prediction": prediction,
                                    "graph_label": graph_label,
                                }
                            )
        finally:
            set_generation_cache(model, enabled=False)
            if was_training:
                model.train()
            if bool(self.cfg.get("empty_cache", True)):
                cleanup_cuda_memory()

        metrics = sequence_metrics(predictions, references)
        output_metrics: dict[str, float | int | str] = {
            f"{metric_key_prefix}_generation_samples": len(predictions),
            f"{metric_key_prefix}_generation_runtime": perf_counter() - started,
            f"{metric_key_prefix}_exact_match": float(metrics["exact_match"]),
            f"{metric_key_prefix}_normalized_exact_match": float(
                metrics["normalized_exact_match"]
            ),
            f"{metric_key_prefix}_mean_token_edit_distance": float(
                metrics["mean_token_edit_distance"]
            ),
            f"{metric_key_prefix}_mean_normalized_token_edit_distance": float(
                metrics["mean_normalized_token_edit_distance"]
            ),
        }

        graph_metrics = self._graph_metrics(
            graph_rows=graph_rows,
            trainer=trainer,
            metric_key_prefix=metric_key_prefix,
        )
        output_metrics.update(graph_metrics)
        logger.info(
            "Eval generation metrics | epoch=%.3f step=%s samples=%s EM=%.6f graph_EM=%s structure_EM=%s",
            trainer.state.epoch or 0.0,
            trainer.state.global_step,
            len(predictions),
            output_metrics[f"{metric_key_prefix}_exact_match"],
            output_metrics.get(f"{metric_key_prefix}_graph_em", "skipped"),
            output_metrics.get(f"{metric_key_prefix}_graph_structure_em", "skipped"),
        )
        return output_metrics

    def _graph_metrics(
        self,
        *,
        graph_rows: list[dict[str, str]],
        trainer: Seq2SeqTrainer,
        metric_key_prefix: str,
    ) -> dict[str, float | int | str]:
        if not bool(self.cfg.get("graph_eval", False)):
            return {}
        if not graph_rows:
            return {f"{metric_key_prefix}_graph_samples": 0}

        tool_value = self.cfg.get("graph_matching_tool_dir")
        if not tool_value:
            self._warn_graph_skip("eval_metrics.graph_matching_tool_dir is not set.")
            return {f"{metric_key_prefix}_graph_samples": 0}
        tool_dir = Path(str(tool_value))
        if not tool_dir.is_absolute():
            tool_dir = PROJECT_ROOT / tool_dir
        try:
            validate_graph_matching_tool(tool_dir)
        except FileNotFoundError as exc:
            if bool(self.cfg.get("require_graph_eval", False)):
                raise
            self._warn_graph_skip(str(exc))
            return {f"{metric_key_prefix}_graph_samples": 0}

        stem = f"step_{trainer.state.global_step:08d}"
        if trainer.state.epoch is not None:
            stem += f"_epoch_{trainer.state.epoch:.3f}".replace(".", "_")
        out_dir = ensure_dir(self.output_dir / "eval_metrics")
        rec_path = out_dir / f"{stem}.rec.txt"
        lab_path = out_dir / f"{stem}.lab.txt"
        graph_output_path = out_dir / f"{stem}.graph_result.txt"
        write_graph_matching_files(graph_rows, rec_path, lab_path)
        result = run_graph_matching_tool(
            tool_dir=tool_dir,
            rec_path=rec_path,
            lab_path=lab_path,
            output_path=graph_output_path,
            num_workers=int(self.cfg.get("graph_num_workers", 8)),
        )
        return {
            f"{metric_key_prefix}_graph_samples": len(graph_rows),
            f"{metric_key_prefix}_graph_em": float(result.metrics["graph_em"]),
            f"{metric_key_prefix}_graph_structure_em": float(
                result.metrics["graph_structure_em"]
            ),
            f"{metric_key_prefix}_graph_output_txt": str(graph_output_path),
        }

    def _warn_graph_skip(self, message: str) -> None:
        if self._graph_warning_emitted:
            return
        logger.warning("Skipping eval graph metrics: %s", message)
        self._graph_warning_emitted = True


RFL_MSD_BATCH_KEYS = {
    "rfl_branch_token_positions",
    "rfl_bond_token_positions",
    "rfl_branch_labels",
    "rfl_branch_mask",
    "rfl_token_alignment_ok",
    "rfl_split_token_count",
}


def _nested_config_value(config: Any, *names: str) -> Any:
    current = config
    for name in names:
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def infer_decoder_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    candidates = (
        _nested_config_value(config, "decoder", "hidden_size"),
        _nested_config_value(config, "decoder", "d_model"),
        _nested_config_value(config, "text_config", "hidden_size"),
        _nested_config_value(config, "text_config", "d_model"),
        _nested_config_value(config, "decoder_hidden_size"),
        _nested_config_value(config, "hidden_size"),
        _nested_config_value(config, "d_model"),
    )
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    raise SystemExit(
        "RFL-MSD loss is enabled but decoder hidden size could not be inferred from model.config. "
        "Set a model config with decoder.hidden_size/d_model or disable loss.type=rfl_msd."
    )


def _output_value(outputs: Any, name: str) -> Any:
    if hasattr(outputs, name):
        return getattr(outputs, name)
    if isinstance(outputs, dict):
        return outputs.get(name)
    return None


def decoder_last_hidden_state(outputs: Any) -> torch.Tensor:
    hidden_states = _output_value(outputs, "decoder_hidden_states")
    if hidden_states is None:
        hidden_states = _output_value(outputs, "hidden_states")
    if not hidden_states:
        raise RuntimeError(
            "RFL-MSD loss requires decoder hidden states, but the model did not return them. "
            "The TexTeller model must support output_hidden_states=True for this mode."
        )
    last_hidden = hidden_states[-1]
    if not isinstance(last_hidden, torch.Tensor) or last_hidden.ndim != 3:
        raise RuntimeError(
            "RFL-MSD loss expected decoder hidden state [B,L,D], "
            f"got {type(last_hidden).__name__} with shape {getattr(last_hidden, 'shape', None)}."
        )
    return last_hidden


def gather_token_states(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim != 2:
        raise ValueError(f"Expected positions [B,N], got {tuple(positions.shape)}")
    if positions.shape[1] == 0:
        return hidden.new_zeros((hidden.shape[0], 0, hidden.shape[-1]))
    valid = positions >= 0
    clamped = positions.clamp(min=0, max=max(0, hidden.shape[1] - 1))
    gathered = hidden.gather(1, clamped.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
    return gathered * valid.unsqueeze(-1).to(dtype=hidden.dtype)


class ChemSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args: Any,
        length_balanced_sampling: dict[str, Any] | None = None,
        eval_generation_metrics: EvalGenerationMetricRunner | None = None,
        rfl_msd_loss: RflMsdLoss | None = None,
        cuda_memory_log_steps: int = 0,
        cuda_empty_cache_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.length_balanced_sampling = length_balanced_sampling
        self._length_balanced_weights: torch.DoubleTensor | None = None
        self.eval_generation_metrics = eval_generation_metrics
        self.rfl_msd_loss = rfl_msd_loss
        self._rfl_split_warning_emitted = False
        self._rfl_alignment_warning_emitted = False
        self.cuda_memory_log_steps = max(0, int(cuda_memory_log_steps))
        self.cuda_empty_cache_steps = max(0, int(cuda_empty_cache_steps))
        self._raw_training_step_count = 0

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        if self.rfl_msd_loss is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        rfl_inputs = {key: inputs.pop(key, None) for key in RFL_MSD_BATCH_KEYS}
        labels = inputs.get("labels")
        if not isinstance(labels, torch.Tensor):
            raise RuntimeError("RFL-MSD loss requires labels in the training batch.")

        inputs["output_hidden_states"] = True
        inputs["return_dict"] = True
        outputs = model(**inputs)
        token_logits = _output_value(outputs, "logits")
        if not isinstance(token_logits, torch.Tensor):
            raise RuntimeError("RFL-MSD loss requires token logits from the model output.")

        branch_positions = rfl_inputs["rfl_branch_token_positions"]
        bond_positions = rfl_inputs["rfl_bond_token_positions"]
        branch_labels = rfl_inputs["rfl_branch_labels"]
        branch_mask = rfl_inputs["rfl_branch_mask"]
        alignment_ok = rfl_inputs["rfl_token_alignment_ok"]
        split_token_count = rfl_inputs["rfl_split_token_count"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (branch_positions, bond_positions, branch_labels, branch_mask, alignment_ok)
        ):
            raise RuntimeError(
                "RFL-MSD loss is enabled, but the dataset batch does not contain RFL-MSD tensors. "
                "Build the dataset with scripts/create_rfl_target_dataset.py and set data.rfl_aux_field."
            )
        if not bool(alignment_ok.all().item()):
            bad_items = int((~alignment_ok.bool()).sum().item())
            if not self._rfl_alignment_warning_emitted and self.is_world_process_zero():
                logger.warning(
                    "RFL-MSD loss found %s batch item(s) whose RFL word tokens could not be aligned "
                    "to decoder label positions. Sequence loss will still train them, but branch loss "
                    "is masked for those items. Inspect rfl_token_alignment_ok if this warning persists.",
                    bad_items,
                )
                self._rfl_alignment_warning_emitted = True
            valid_alignment = alignment_ok.bool().view(-1, 1, 1).to(device=branch_mask.device)
            branch_mask = branch_mask * valid_alignment.to(dtype=branch_mask.dtype)
        if (
            isinstance(split_token_count, torch.Tensor)
            and int(split_token_count.sum().item()) > 0
            and not self._rfl_split_warning_emitted
            and self.is_world_process_zero()
        ):
            logger.warning(
                "RFL-MSD loss is using the first subtoken hidden state for RFL words that the tokenizer "
                "split into multiple pieces. For paper-faithful MSD training, extend/use a tokenizer that "
                "keeps RFL tokens atomic."
            )
            self._rfl_split_warning_emitted = True

        hidden = decoder_last_hidden_state(outputs)
        branch_hidden = gather_token_states(hidden, branch_positions)
        bond_hidden = gather_token_states(hidden, bond_positions)
        branch_head = getattr(model, "rfl_msd_branch_classifier", None)
        if not isinstance(branch_head, RflMsdBranchClassifier):
            raise RuntimeError("Model is missing rfl_msd_branch_classifier.")
        branch_logits = branch_head(branch_hidden, bond_hidden)
        loss_output = self.rfl_msd_loss(
            token_logits,
            labels,
            branch_logits=branch_logits,
            branch_labels=branch_labels,
            branch_mask=branch_mask,
        )
        if return_outputs:
            return loss_output.loss, outputs
        return loss_output.loss

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, Any]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        cleanup_cuda_memory()
        if self.eval_generation_metrics is not None:
            generation_metrics = self.eval_generation_metrics.compute(
                self,
                metric_key_prefix=metric_key_prefix,
            )
            if generation_metrics:
                metrics.update(generation_metrics)
                self.log(generation_metrics)
        cleanup_cuda_memory()
        return metrics

    def training_step(self, model: torch.nn.Module, inputs: dict[str, Any], *args: Any, **kwargs: Any):
        label_max_len = None
        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor) and labels.ndim >= 2:
            label_max_len = int(labels.shape[-1])

        loss = super().training_step(model, inputs, *args, **kwargs)
        self._raw_training_step_count += 1

        if torch.cuda.is_available():
            if (
                self.cuda_memory_log_steps > 0
                and self._raw_training_step_count % self.cuda_memory_log_steps == 0
                and self.is_world_process_zero()
            ):
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                logger.info(
                    "CUDA memory | raw_step=%s global_step=%s label_max_len=%s "
                    "allocated=%.2fGiB reserved=%.2fGiB peak_allocated=%.2fGiB",
                    self._raw_training_step_count,
                    self.state.global_step,
                    label_max_len,
                    allocated,
                    reserved,
                    peak,
                )
            if (
                self.cuda_empty_cache_steps > 0
                and self._raw_training_step_count % self.cuda_empty_cache_steps == 0
            ):
                cleanup_cuda_memory()
        return loss

    def _get_train_sampler(self):
        if self.length_balanced_sampling is None:
            return super()._get_train_sampler()
        if self.train_dataset is None:
            return None
        if self.args.world_size != 1:
            logger.warning(
                "Length-balanced sampling is enabled but world_size=%s. Falling back to the default sampler.",
                self.args.world_size,
            )
            return super()._get_train_sampler()
        if self._length_balanced_weights is None:
            self._length_balanced_weights = _build_length_balanced_weights(
                self.train_dataset,
                self.length_balanced_sampling,
            )
        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed))
        return WeightedRandomSampler(
            weights=self._length_balanced_weights,
            num_samples=len(self._length_balanced_weights),
            replacement=True,
            generator=generator,
        )


def training_log_paths(output_dir: Path) -> tuple[Path, Path]:
    log_dir = ensure_dir(PROJECT_ROOT / "logs")
    run_name = output_dir.name or "train"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{run_name}_{timestamp}"
    return log_dir / f"{stem}.log", log_dir / f"{stem}.trainer_events.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained TexTeller on EDU-CHEMC.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_edu_chemc.yaml"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/processed/edu_chemc"))
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="OleehyO/TexTeller")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/runs/edu_chemc_texteller"))
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--texteller_repo_path", type=str, default=None)
    return parser.parse_args()


def training_args_kwargs(output_dir: Path, training_cfg: dict[str, Any]) -> dict[str, Any]:
    stable_tqdm = bool(training_cfg.get("stable_tqdm", True))
    eval_strategy = training_cfg.get(
        "eval_strategy",
        training_cfg.get("evaluation_strategy", "epoch"),
    )
    save_strategy = training_cfg.get("save_strategy", eval_strategy)
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": training_cfg.get("num_train_epochs", 20),
        "learning_rate": training_cfg.get("learning_rate", 1.0e-5),
        "weight_decay": training_cfg.get("weight_decay", 0.01),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": training_cfg.get("per_device_eval_batch_size", 2),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 16),
        "max_grad_norm": training_cfg.get("max_grad_norm", 1.0),
        "fp16": training_cfg.get("fp16", False),
        "bf16": training_cfg.get("bf16", False),
        "dataloader_num_workers": training_cfg.get("dataloader_num_workers", 4),
        "dataloader_pin_memory": training_cfg.get("dataloader_pin_memory", True),
        "dataloader_persistent_workers": training_cfg.get("dataloader_persistent_workers", False),
        "dataloader_prefetch_factor": training_cfg.get("dataloader_prefetch_factor"),
        "logging_steps": training_cfg.get("logging_steps", 50),
        "logging_strategy": training_cfg.get("logging_strategy", "steps"),
        "logging_first_step": training_cfg.get("logging_first_step", False),
        "log_level": training_cfg.get("log_level", "info"),
        "log_level_replica": training_cfg.get("log_level_replica", "error"),
        "log_on_each_node": training_cfg.get("log_on_each_node", False),
        "disable_tqdm": training_cfg.get("disable_tqdm", stable_tqdm),
        "save_total_limit": training_cfg.get("save_total_limit", 5),
        "save_strategy": save_strategy,
        "load_best_model_at_end": training_cfg.get("load_best_model_at_end", True),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": training_cfg.get("greater_is_better", False),
        "report_to": training_cfg.get("report_to", ["tensorboard"]),
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "predict_with_generate": False,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments)
    if "warmup_steps" in training_cfg:
        kwargs["warmup_steps"] = training_cfg["warmup_steps"]
    else:
        kwargs["warmup_ratio"] = training_cfg.get("warmup_ratio", 0.05)
    if str(eval_strategy).lower() == "steps":
        kwargs["eval_steps"] = training_cfg.get("eval_steps", 500)
    if str(save_strategy).lower() == "steps":
        kwargs["save_steps"] = training_cfg.get("save_steps", 500)
    if "ddp_find_unused_parameters" in signature.parameters:
        kwargs["ddp_find_unused_parameters"] = training_cfg.get(
            "ddp_find_unused_parameters",
            False,
        )
    strategy_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[strategy_key] = eval_strategy
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def validate_precision_config(training_cfg: dict[str, Any]) -> None:
    if bool(training_cfg.get("fp16", False)) and bool(training_cfg.get("bf16", False)):
        raise SystemExit("training.fp16 and training.bf16 cannot both be true.")
    if not bool(training_cfg.get("bf16", False)):
        return
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise SystemExit(
            "training.bf16 is true, but this CUDA device does not support bf16. "
            "Use an Ampere-or-newer GPU, set training.fp16=true, or set training.bf16=false."
        )


def build_rfl_msd_loss(config: dict[str, Any], model: torch.nn.Module) -> RflMsdLoss | None:
    loss_cfg = config.get("loss", {})
    if not isinstance(loss_cfg, dict) or str(loss_cfg.get("type", "seq2seq")).lower() != "rfl_msd":
        return None
    hidden_size = infer_decoder_hidden_size(model)
    match_size = loss_cfg.get("branch_match_size")
    branch_head = RflMsdBranchClassifier(
        hidden_size=hidden_size,
        match_size=int(match_size) if match_size is not None else None,
    )
    model.add_module("rfl_msd_branch_classifier", branch_head)
    logger.info(
        "Enabled RFL-MSD loss | hidden_size=%s lambda_sequence=%s lambda_branch=%s.",
        hidden_size,
        loss_cfg.get("lambda_sequence", 1.0),
        loss_cfg.get("lambda_branch", 1.0),
    )
    return RflMsdLoss(
        lambda_sequence=float(loss_cfg.get("lambda_sequence", 1.0)),
        lambda_branch=float(loss_cfg.get("lambda_branch", 1.0)),
        label_ignore_index=-100,
        branch_ignore_index=-1,
    )


def save_model_with_assets(
    trainer: Seq2SeqTrainer,
    bundle: Any,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    trainer.save_model(str(output_dir))
    bundle.tokenizer.save_pretrained(output_dir)
    if bundle.processor is not None and hasattr(bundle.processor, "save_pretrained"):
        bundle.processor.save_pretrained(output_dir)


def target_length_policy(config: dict[str, Any]) -> str:
    data_cfg = config.get("data", {})
    if isinstance(data_cfg, dict) and "target_length_policy" in data_cfg:
        return str(data_cfg["target_length_policy"])
    return str(config.get("target_length_policy", "error"))


def data_target_key(config: dict[str, Any], split: str) -> str:
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


def configure_progress_callback(trainer: Seq2SeqTrainer, training_cfg: dict[str, Any]) -> None:
    if not bool(training_cfg.get("stable_tqdm", True)):
        return

    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(
        StableTqdmProgressCallback(
            mininterval=float(training_cfg.get("tqdm_mininterval", 5.0)),
            miniters=int(training_cfg.get("tqdm_miniters", 10)),
            ncols=int(training_cfg.get("tqdm_ncols", 100)),
            ascii_bar=bool(training_cfg.get("tqdm_ascii", False)),
            max_str_len=int(training_cfg.get("tqdm_max_str_len", 100)),
            by_epoch=bool(training_cfg.get("tqdm_by_epoch", True)),
            write_logs=bool(training_cfg.get("tqdm_write_logs", False)),
        )
    )


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value in {"auto", "auto_decoder", "all-linear"}:
            return [value]
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise TypeError(f"Expected target_modules to be null, string, or list; got {type(value).__name__}")


def _in_lora_scope(module_name: str, scope: str) -> bool:
    if scope in {"all", "all_linear"}:
        return True
    if scope in {"decoder", "decoder_only"}:
        return module_name.startswith("decoder") or ".decoder" in module_name
    if scope in {"encoder", "encoder_only"}:
        return module_name.startswith("encoder") or ".encoder" in module_name
    raise ValueError(f"Unsupported LoRA target_scope: {scope}")


def infer_lora_target_modules(
    model: torch.nn.Module,
    target_modules: Any,
    target_scope: str = "decoder",
) -> str | list[str]:
    requested = _as_list(target_modules)
    if requested and requested != ["auto"] and requested != ["auto_decoder"]:
        if requested == ["all-linear"]:
            return "all-linear"
        return requested

    scope = "decoder" if requested == ["auto_decoder"] else target_scope
    linear_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    scoped_leaves = {
        name.rsplit(".", 1)[-1]
        for name in linear_modules
        if _in_lora_scope(name, scope)
    }
    selected = [
        leaf
        for leaf in DEFAULT_LORA_TARGET_LEAVES
        if leaf in scoped_leaves and leaf not in EXCLUDED_LORA_TARGET_LEAVES
    ]
    if not selected:
        selected = sorted(
            leaf
            for leaf in scoped_leaves
            if leaf not in EXCLUDED_LORA_TARGET_LEAVES
        )
    if not selected:
        raise RuntimeError(
            f"Could not infer LoRA target modules for scope '{scope}'. "
            "Set lora.target_modules explicitly in the config."
        )

    selected_set = set(selected)
    has_same_leaf_outside_scope = any(
        name.rsplit(".", 1)[-1] in selected_set and not _in_lora_scope(name, scope)
        for name in linear_modules
    )
    if scope not in {"all", "all_linear"} and has_same_leaf_outside_scope:
        leaf_pattern = "|".join(selected)
        return rf".*{scope}.*\.({leaf_pattern})$"
    return selected


def maybe_apply_lora(model: torch.nn.Module, cfg: dict[str, Any], cli_enabled: bool) -> torch.nn.Module:
    lora_cfg = cfg.get("lora", {})
    enabled = cli_enabled or bool(lora_cfg.get("enabled", False))
    if not enabled:
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA requested but peft is not installed.") from exc

    target_modules = infer_lora_target_modules(
        model,
        lora_cfg.get("target_modules", "auto"),
        target_scope=str(lora_cfg.get("target_scope", "decoder")),
    )
    logger.info("Using LoRA target modules: %s", target_modules)
    kwargs = {
        "r": int(lora_cfg.get("r", 16)),
        "lora_alpha": int(lora_cfg.get("alpha", 32)),
        "lora_dropout": float(lora_cfg.get("dropout", 0.05)),
        "bias": str(lora_cfg.get("bias", "none")),
        "target_modules": target_modules,
    }
    modules_to_save = _as_list(lora_cfg.get("modules_to_save"))
    if modules_to_save:
        kwargs["modules_to_save"] = modules_to_save
    peft_config = LoraConfig(**kwargs)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def trainer_kwargs_for_processing_class(tokenizer: Any) -> dict[str, Any]:
    signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    return {}


def main() -> None:
    global logger
    args = parse_args()
    log_file, event_log_file = training_log_paths(args.output_dir)
    logger = setup_logging(log_file=log_file)
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    ensure_dir(args.output_dir)
    save_yaml(config, args.output_dir / "train_config.yaml")
    shutil.copy2(args.config, args.output_dir / "source_config.yaml")
    save_json(vars(args), args.output_dir / "training_args.json")
    logger.info("Writing run log to %s", log_file)
    logger.info("Writing trainer event log to %s", event_log_file)
    logger.info("Saved run config metadata to %s", args.output_dir)

    if args.from_scratch:
        raise SystemExit(
            "--from_scratch is intentionally not implemented for the default pipeline. "
            "This project is for fine-tuning/domain adaptation from pretrained TexTeller."
        )
    if not args.pretrained_model_name_or_path:
        raise SystemExit(
            "--pretrained_model_name_or_path is required. Pass a HuggingFace model id or "
            "local TexTeller checkpoint path."
        )

    add_texteller_repo_to_path(args.texteller_repo_path)
    training_cfg = config.get("training", {})
    validate_precision_config(training_cfg)
    bundle = load_pretrained_model_and_tokenizer(
        model_name_or_path=args.pretrained_model_name_or_path,
        tokenizer_path=args.tokenizer_path,
        device=None,
        trust_remote_code=args.trust_remote_code,
    )
    added_token_init_specs = maybe_extend_tokenizer_with_chemical_tokens(
        bundle.tokenizer,
        config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )
    if added_token_init_specs and args.resume_from_checkpoint:
        logger.warning(
            "Chemical tokens were added while --resume_from_checkpoint is set. "
            "Prefer loading the previous best checkpoint via --pretrained_model_name_or_path "
            "and writing to a fresh output_dir, because optimizer states from the old vocab "
            "usually do not match resized embeddings."
        )
    resize_token_embeddings_if_needed(bundle.model, bundle.tokenizer)
    initialize_new_token_embeddings_from_subtokens(bundle.model, added_token_init_specs)

    freeze_cfg = config.get("freeze", {})
    if args.freeze_encoder or bool(freeze_cfg.get("encoder", False)):
        frozen = freeze_encoder_if_available(bundle.model)
        logger.info("Froze encoder parameters: %s", frozen)

    if args.gradient_checkpointing or bool(training_cfg.get("gradient_checkpointing", False)):
        enable_gradient_checkpointing_if_available(bundle.model)
        logger.info("Enabled gradient checkpointing.")
    set_generation_cache(bundle.model, enabled=False)

    bundle.model = maybe_apply_lora(bundle.model, config, cli_enabled=args.use_lora)
    set_generation_cache(bundle.model, enabled=False)
    rfl_msd_loss = build_rfl_msd_loss(config, bundle.model)
    log_trainable_parameter_summary(bundle.model)

    max_target_length = int(config.get("max_target_length", 512))
    length_policy = target_length_policy(config)
    train_target_key = data_target_key(config, "train")
    eval_target_key = data_target_key(config, "validation")
    data_cfg = config.get("data", {})
    rfl_aux_field = (
        str(data_cfg.get("rfl_aux_field"))
        if isinstance(data_cfg, dict) and data_cfg.get("rfl_aux_field")
        else None
    )
    train_transform = build_transform(config, train=True, processor=bundle.processor)
    eval_transform = build_transform(config, train=False, processor=bundle.processor)
    train_dataset = EduChemcDataset(
        split_dir=args.dataset_dir / "train",
        tokenizer=bundle.tokenizer,
        transform=train_transform,
        max_target_length=max_target_length,
        target_key=train_target_key,
        target_length_policy=length_policy,
        rfl_aux_field=rfl_aux_field,
    )
    eval_dataset = EduChemcDataset(
        split_dir=args.dataset_dir / "validation",
        tokenizer=bundle.tokenizer,
        transform=eval_transform,
        max_target_length=max_target_length,
        target_key=eval_target_key,
        target_length_policy=length_policy,
        rfl_aux_field=rfl_aux_field,
    )
    collator = VisionSeq2SeqCollator(bundle.tokenizer)

    train_args = Seq2SeqTrainingArguments(
        **training_args_kwargs(args.output_dir, training_cfg)
    )
    trainer = ChemSeq2SeqTrainer(
        model=bundle.model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        length_balanced_sampling=_length_balanced_cfg(config),
        eval_generation_metrics=EvalGenerationMetricRunner(
            dataset=eval_dataset,
            tokenizer=bundle.tokenizer,
            output_dir=args.output_dir,
            cfg=config.get("eval_metrics", {}),
        ),
        rfl_msd_loss=rfl_msd_loss,
        cuda_memory_log_steps=int(training_cfg.get("cuda_memory_log_steps", 0)),
        cuda_empty_cache_steps=int(training_cfg.get("cuda_empty_cache_steps", 0)),
        **trainer_kwargs_for_processing_class(bundle.tokenizer),
    )
    configure_progress_callback(trainer, training_cfg)
    trainer.add_callback(TrainingFileLogCallback(event_log_file))

    logger.info(
        "Starting fine-tuning with %s train and %s validation samples | train_target_key=%s eval_target_key=%s.",
        len(train_dataset),
        len(eval_dataset),
        train_target_key,
        eval_target_key,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if bool(training_cfg.get("save_last_model", False)):
        if bool(training_cfg.get("load_best_model_at_end", True)):
            logger.warning(
                "training.save_last_model=true with load_best_model_at_end=true saves "
                "the loaded best model, not final-step weights."
            )
        save_model_with_assets(trainer, bundle, args.output_dir / "last")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    best_dir = ensure_dir(args.output_dir / "best")
    save_model_with_assets(trainer, bundle, best_dir)

    save_yaml(config, args.output_dir / "train_config.yaml")
    save_yaml(config, best_dir / "train_config.yaml")
    save_json(vars(args), args.output_dir / "training_args.json")
    save_json({"model_type": bundle.model_type, "source": bundle.source}, args.output_dir / "model_loader.json")
    (args.output_dir / "trainer_state_summary.json").write_text(
        json.dumps({"train": metrics, "eval": eval_metrics}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Finished. Best/loaded model saved to %s", best_dir)


if __name__ == "__main__":
    main()
