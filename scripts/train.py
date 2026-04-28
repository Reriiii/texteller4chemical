from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

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


def maybe_apply_lora(model: torch.nn.Module, cfg: dict[str, Any], cli_enabled: bool) -> torch.nn.Module:
    lora_cfg = cfg.get("lora", {})
    enabled = cli_enabled or bool(lora_cfg.get("enabled", False))
    if not enabled:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA requested but peft is not installed.") from exc

    target_modules = lora_cfg.get("target_modules")
    kwargs = {
        "task_type": TaskType.SEQ_2_SEQ_LM,
        "r": int(lora_cfg.get("r", 16)),
        "lora_alpha": int(lora_cfg.get("alpha", 32)),
        "lora_dropout": float(lora_cfg.get("dropout", 0.05)),
    }
    if target_modules:
        kwargs["target_modules"] = target_modules
    peft_config = LoraConfig(**kwargs)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


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
        tokenizer=bundle.tokenizer,
    )

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
