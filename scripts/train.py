from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback, TrainerCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.data import EduChemcDataset, VisionSeq2SeqCollator
from chemtexteller.model_loader import (
    add_texteller_repo_to_path,
    enable_gradient_checkpointing_if_available,
    freeze_encoder_if_available,
    load_pretrained_model_and_tokenizer,
    resize_token_embeddings_if_needed,
)
from chemtexteller.transforms import build_transform
from chemtexteller.utils import ensure_dir, load_yaml, save_json, save_yaml, set_seed, setup_logging


logger = setup_logging()


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
    ) -> None:
        self.mininterval = mininterval
        self.miniters = miniters
        self.ncols = ncols
        self.ascii_bar = ascii_bar
        self.max_str_len = max_str_len
        self.training_bar = None
        self.prediction_bar = None
        self.current_step = 0

    def _bar(self, total: int, leave: bool) -> tqdm:
        return tqdm(
            total=total,
            leave=leave,
            dynamic_ncols=False,
            ncols=self.ncols,
            mininterval=self.mininterval,
            miniters=self.miniters,
            ascii=self.ascii_bar,
            file=sys.stdout,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = self._bar(total=state.max_steps, leave=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
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
            self.prediction_bar = self._bar(total=total, leave=self.training_bar is None)
        self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        self._close_prediction_bar(state)

    def on_predict(self, args, state, control, **kwargs):
        self._close_prediction_bar(state)

    def _close_prediction_bar(self, state) -> None:
        if state.is_world_process_zero and self.prediction_bar is not None:
            self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or self.training_bar is None or logs is None:
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
        self.training_bar.write(str(shallow_logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None


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
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": training_cfg.get("num_train_epochs", 20),
        "learning_rate": training_cfg.get("learning_rate", 1.0e-5),
        "weight_decay": training_cfg.get("weight_decay", 0.01),
        "warmup_ratio": training_cfg.get("warmup_ratio", 0.05),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": training_cfg.get("per_device_eval_batch_size", 2),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 16),
        "max_grad_norm": training_cfg.get("max_grad_norm", 1.0),
        "fp16": training_cfg.get("fp16", False),
        "bf16": training_cfg.get("bf16", False),
        "dataloader_num_workers": training_cfg.get("dataloader_num_workers", 4),
        "logging_steps": training_cfg.get("logging_steps", 50),
        "logging_strategy": training_cfg.get("logging_strategy", "steps"),
        "logging_first_step": training_cfg.get("logging_first_step", True),
        "log_level": training_cfg.get("log_level", "info"),
        "disable_tqdm": training_cfg.get("disable_tqdm", stable_tqdm),
        "eval_steps": training_cfg.get("eval_steps", 500),
        "save_steps": training_cfg.get("save_steps", 500),
        "save_total_limit": training_cfg.get("save_total_limit", 5),
        "save_strategy": training_cfg.get("save_strategy", "steps"),
        "load_best_model_at_end": training_cfg.get("load_best_model_at_end", True),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": training_cfg.get("greater_is_better", False),
        "report_to": training_cfg.get("report_to", ["tensorboard"]),
        "remove_unused_columns": False,
        "predict_with_generate": False,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments)
    strategy_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[strategy_key] = training_cfg.get("eval_strategy", training_cfg.get("evaluation_strategy", "steps"))
    return kwargs


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
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA requested but peft is not installed.") from exc

    target_modules = infer_lora_target_modules(
        model,
        lora_cfg.get("target_modules", "auto"),
        target_scope=str(lora_cfg.get("target_scope", "decoder")),
    )
    logger.info("Using LoRA target modules: %s", target_modules)
    kwargs = {
        "task_type": TaskType.SEQ_2_SEQ_LM,
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
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    ensure_dir(args.output_dir)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_pretrained_model_and_tokenizer(
        model_name_or_path=args.pretrained_model_name_or_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    resize_token_embeddings_if_needed(bundle.model, bundle.tokenizer)

    freeze_cfg = config.get("freeze", {})
    if args.freeze_encoder or bool(freeze_cfg.get("encoder", False)):
        frozen = freeze_encoder_if_available(bundle.model)
        logger.info("Froze encoder parameters: %s", frozen)

    if args.gradient_checkpointing:
        enable_gradient_checkpointing_if_available(bundle.model)
        logger.info("Enabled gradient checkpointing.")

    bundle.model = maybe_apply_lora(bundle.model, config, cli_enabled=args.use_lora)

    max_target_length = int(config.get("max_target_length", 512))
    train_transform = build_transform(config, train=True, processor=bundle.processor)
    eval_transform = build_transform(config, train=False, processor=bundle.processor)
    train_dataset = EduChemcDataset(
        split_dir=args.dataset_dir / "train",
        tokenizer=bundle.tokenizer,
        transform=train_transform,
        max_target_length=max_target_length,
    )
    eval_dataset = EduChemcDataset(
        split_dir=args.dataset_dir / "validation",
        tokenizer=bundle.tokenizer,
        transform=eval_transform,
        max_target_length=max_target_length,
    )
    collator = VisionSeq2SeqCollator(bundle.tokenizer)

    train_args = Seq2SeqTrainingArguments(
        **training_args_kwargs(args.output_dir, config.get("training", {}))
    )
    trainer = Seq2SeqTrainer(
        model=bundle.model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        **trainer_kwargs_for_processing_class(bundle.tokenizer),
    )
    configure_progress_callback(trainer, config.get("training", {}))

    logger.info(
        "Starting fine-tuning with %s train and %s validation samples.",
        len(train_dataset),
        len(eval_dataset),
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "last"))
    bundle.tokenizer.save_pretrained(args.output_dir / "last")
    if bundle.processor is not None and hasattr(bundle.processor, "save_pretrained"):
        bundle.processor.save_pretrained(args.output_dir / "last")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    best_dir = ensure_dir(args.output_dir / "best")
    trainer.save_model(str(best_dir))
    bundle.tokenizer.save_pretrained(best_dir)
    if bundle.processor is not None and hasattr(bundle.processor, "save_pretrained"):
        bundle.processor.save_pretrained(best_dir)

    save_yaml(config, args.output_dir / "train_config.yaml")
    save_yaml(config, best_dir / "train_config.yaml")
    shutil.copy2(args.config, args.output_dir / "source_config.yaml")
    save_json(vars(args), args.output_dir / "training_args.json")
    save_json({"model_type": bundle.model_type, "source": bundle.source}, args.output_dir / "model_loader.json")
    (args.output_dir / "trainer_state_summary.json").write_text(
        json.dumps({"train": metrics, "eval": eval_metrics}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Finished. Best/loaded model saved to %s", best_dir)


if __name__ == "__main__":
    main()
