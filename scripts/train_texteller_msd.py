from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoProcessor, get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.graph_matching_eval import (  # noqa: E402
    lookup_target,
    run_graph_matching_tool,
    write_graph_matching_files,
)
from chemtexteller.metrics import sequence_metrics  # noqa: E402
from chemtexteller.rfl_adapter import restore_rfl_tokens_to_chemfig  # noqa: E402
from chemtexteller.rfl_vocab import RflVocab  # noqa: E402
from chemtexteller.texteller_msd_data import TextellerMsdCollator, TextellerMsdDataset  # noqa: E402
from chemtexteller.texteller_msd_model import MsdDecoderConfig, TexTellerMsdModel  # noqa: E402
from chemtexteller.transforms import build_transform  # noqa: E402
from chemtexteller.utils import ensure_dir, load_yaml, save_json, save_yaml, set_seed, setup_logging  # noqa: E402


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TexTeller encoder + full MSD decoder/head.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_edu_chemc_texteller_msd.yaml"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/processed/edu_chemc_rfl_msd"))
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="OleehyO/TexTeller")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/runs/edu_chemc_texteller_msd"))
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


def load_processor(model_name_or_path: str, trust_remote_code: bool) -> Any | None:
    for cls in (AutoProcessor, AutoImageProcessor):
        try:
            return cls.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        except Exception:
            continue
    return None


def msd_config_from_yaml(config: dict[str, Any]) -> MsdDecoderConfig:
    model_cfg = dict(config.get("msd_model") or {})
    loss_cfg = dict(config.get("loss") or {})
    kwargs: dict[str, Any] = {}
    for field_name in MsdDecoderConfig.__dataclass_fields__:
        if field_name in model_cfg:
            kwargs[field_name] = model_cfg[field_name]
    if "lambda_sequence" in loss_cfg:
        kwargs["lambda_sequence"] = float(loss_cfg["lambda_sequence"])
    if "lambda_memory" in loss_cfg:
        kwargs["lambda_memory"] = float(loss_cfg["lambda_memory"])
    if "lambda_branch" in loss_cfg:
        kwargs["lambda_memory"] = float(loss_cfg["lambda_branch"])
    for key in ("decoder_cover_kernel", "decoder_cover_padding"):
        if key in kwargs and isinstance(kwargs[key], list):
            kwargs[key] = tuple(int(item) for item in kwargs[key])
    return MsdDecoderConfig(**kwargs)


def lookup_metadata_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is not None:
        return value
    targets = row.get("targets")
    if isinstance(targets, dict):
        nested_key = key.split(".", 1)[1] if key.startswith("targets.") else key
        return targets.get(nested_key)
    return None


def iter_metadata_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def iter_rfl_tokens_from_split(
    split_dir: Path,
    *,
    target_key: str,
    rfl_aux_field: str,
):
    metadata_path = split_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        return
    for row in iter_metadata_rows(metadata_path):
        aux = row.get(rfl_aux_field)
        tokens = aux.get("tokens") if isinstance(aux, dict) else None
        if isinstance(tokens, list) and all(isinstance(item, str) for item in tokens):
            yield list(tokens)
            continue
        target = lookup_metadata_value(row, target_key)
        if isinstance(target, str) and target.strip():
            yield target.split()


def load_rfl_vocab(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[RflVocab, dict[str, Any]]:
    vocab_cfg = dict(config.get("vocab") or {})
    data_cfg = dict(config.get("data") or {})
    vocab_file = Path(vocab_cfg.get("file", "external/RFL-MSD/dict/vocab.txt"))
    vocab = RflVocab.from_file(vocab_file)
    report: dict[str, Any] = {
        "vocab_file": str(vocab_file),
        "base_vocab_size": vocab.vocab_size,
        "extended_from_dataset": False,
        "added_tokens": 0,
        "unknown_token_count_before_extension": 0,
        "unknown_token_top_before_extension": [],
    }
    if not bool(vocab_cfg.get("extend_from_dataset", False)):
        return vocab, report

    split_names = vocab_cfg.get("extend_splits", ["train"])
    if isinstance(split_names, str):
        split_names = [item.strip() for item in split_names.split(",") if item.strip()]
    split_names = [str(item) for item in split_names]
    min_frequency = int(vocab_cfg.get("min_frequency", 1))
    max_added = vocab_cfg.get("max_added_tokens")
    max_added = None if max_added in {None, 0, "0"} else int(max_added)
    target_key = str(data_cfg.get("target_key", data_cfg.get("train_target_key", "targets.ssml_rfl")))
    rfl_aux_field = str(data_cfg.get("rfl_aux_field", "rfl"))

    counts: Counter[str] = Counter()
    ordered_missing: list[str] = []
    seen_missing: set[str] = set()
    for split in split_names:
        for tokens in iter_rfl_tokens_from_split(
            args.dataset_dir / split,
            target_key=target_key,
            rfl_aux_field=rfl_aux_field,
        ):
            for token in tokens:
                if token in vocab.word2id:
                    continue
                counts[token] += 1
                if token not in seen_missing:
                    seen_missing.add(token)
                    ordered_missing.append(token)

    word2id = dict(vocab.word2id)
    id2word = dict(vocab.id2word)
    next_id = max(id2word) + 1 if id2word else 0
    added_tokens: list[str] = []
    for token in ordered_missing:
        if counts[token] < min_frequency:
            continue
        if max_added is not None and len(added_tokens) >= max_added:
            break
        word2id[token] = next_id
        id2word[next_id] = token
        next_id += 1
        added_tokens.append(token)

    report.update(
        {
            "extended_from_dataset": True,
            "extend_splits": split_names,
            "min_frequency": min_frequency,
            "max_added_tokens": max_added,
            "unknown_token_count_before_extension": int(sum(counts.values())),
            "unknown_token_top_before_extension": counts.most_common(50),
            "added_tokens": len(added_tokens),
            "added_token_list": added_tokens,
            "final_vocab_size": max(id2word) + 1 if id2word else 0,
        }
    )
    if added_tokens:
        logger.info(
            "Extended RFL vocab from dataset | added=%s splits=%s examples=%s",
            len(added_tokens),
            split_names,
            added_tokens[:10],
        )
    return RflVocab(word2id=word2id, id2word=id2word, unk_token=vocab.unk_token), report


def dataloader_kwargs(config: dict[str, Any], *, train: bool) -> dict[str, Any]:
    training = dict(config.get("training") or {})
    workers = int(training.get("dataloader_num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(training.get("dataloader_pin_memory", True)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(training.get("dataloader_persistent_workers", False))
        prefetch = training.get("dataloader_prefetch_factor")
        if prefetch is not None:
            kwargs["prefetch_factor"] = int(prefetch)
    if train:
        kwargs["shuffle"] = True
    return kwargs


def build_datasets(
    args: argparse.Namespace,
    config: dict[str, Any],
    vocab: RflVocab,
    processor: Any | None,
) -> tuple[TextellerMsdDataset, TextellerMsdDataset]:
    data_cfg = dict(config.get("data") or {})
    target_key = str(data_cfg.get("target_key", data_cfg.get("train_target_key", "targets.ssml_rfl")))
    eval_target_key = str(data_cfg.get("eval_target_key", target_key))
    rfl_aux_field = str(data_cfg.get("rfl_aux_field", "rfl"))
    max_target_length = int(config.get("max_target_length", 1024))
    target_length_policy = str(data_cfg.get("target_length_policy", "filter"))
    train_transform = build_transform(config, train=True, processor=processor)
    eval_transform = build_transform(config, train=False, processor=processor)
    train_dataset = TextellerMsdDataset(
        args.dataset_dir / "train",
        vocab,
        train_transform,
        target_key=target_key,
        rfl_aux_field=rfl_aux_field,
        max_target_length=max_target_length,
        target_length_policy=target_length_policy,
        max_samples=args.max_train_samples,
    )
    eval_dataset = TextellerMsdDataset(
        args.dataset_dir / "validation",
        vocab,
        eval_transform,
        target_key=eval_target_key,
        rfl_aux_field=rfl_aux_field,
        max_target_length=max_target_length,
        target_length_policy=target_length_policy,
        max_samples=args.max_eval_samples,
    )
    return train_dataset, eval_dataset


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, accelerator: Accelerator) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    seq_losses: list[float] = []
    mem_losses: list[float] = []
    for batch in loader:
        outputs = model(**model_inputs(batch))
        losses.append(float(accelerator.gather(outputs["loss"].detach()).mean().cpu()))
        seq_losses.append(float(accelerator.gather(outputs["sequence_loss"].detach()).mean().cpu()))
        mem_losses.append(float(accelerator.gather(outputs["memory_loss"].detach()).mean().cpu()))
    model.train()
    return {
        "eval_loss": sum(losses) / max(1, len(losses)),
        "eval_sequence_loss": sum(seq_losses) / max(1, len(seq_losses)),
        "eval_memory_loss": sum(mem_losses) / max(1, len(mem_losses)),
    }


@torch.no_grad()
def evaluate_generation(
    model: TexTellerMsdModel,
    loader: DataLoader,
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    epoch: int,
    device: torch.device,
) -> dict[str, Any]:
    eval_cfg = dict(cfg.get("eval_metrics") or {})
    if not bool(eval_cfg.get("enabled", False)):
        return {}
    model.eval()
    max_samples = eval_cfg.get("max_samples")
    max_samples = None if max_samples is None else int(max_samples)
    num_beams = int(eval_cfg.get("num_beams", 5))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", cfg.get("max_target_length", 1024)))
    rfl_tool_dir = Path(eval_cfg.get("rfl_tool_dir", "external/RFL-MSD"))
    graph_label_key = str(eval_cfg.get("graph_label_key", "ssml_rfl_graph_norm"))
    graph_eval = bool(eval_cfg.get("graph_eval", False))
    graph_tool_dir = Path(eval_cfg.get("graph_matching_tool_dir", "external/GraphMatchingTool"))
    graph_workers = int(eval_cfg.get("graph_num_workers", 8))

    rows: list[dict[str, str]] = []
    predictions: list[str] = []
    references: list[str] = []
    graph_rows: list[dict[str, str]] = []
    seen = 0
    for batch in tqdm(loader, desc="Eval MSD generation", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        beam_outputs = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        for idx, beams in enumerate(beam_outputs):
            if max_samples is not None and seen >= max_samples:
                break
            best = beams[0] if beams else None
            raw_prediction = best.text if best is not None else ""
            restore = (
                restore_rfl_tokens_to_chemfig(
                    best.tokens,
                    rfl_tool_dir,
                    branch_pairs=best.branch_pairs,
                    cond_data=best.cond_data,
                )
                if best is not None
                else None
            )
            restored_prediction = restore.chemfig if restore is not None and restore.success else raw_prediction
            reference = str(batch["target_texts"][idx])
            image_name = str(batch["image_names"][idx])
            metadata_targets = batch["metadata_targets"][idx]
            row = {
                "image_name": image_name,
                "ground_truth": reference,
                "prediction": raw_prediction,
                "restored_prediction": restored_prediction,
                "restore_status": "ok" if restore is not None and restore.success else "failed",
                "restore_error": "" if restore is None or restore.success else str(restore.error or ""),
            }
            rows.append(row)
            predictions.append(raw_prediction)
            references.append(reference)
            if graph_eval:
                graph_rows.append(
                    {
                        "image_name": image_name,
                        "prediction": restored_prediction,
                        "graph_label": lookup_target(metadata_targets, graph_label_key),
                    }
                )
            seen += 1
        if max_samples is not None and seen >= max_samples:
            break

    metrics = sequence_metrics(predictions, references)
    out_dir = ensure_dir(output_dir / "eval_metrics")
    csv_path = out_dir / f"epoch_{epoch:03d}_msd_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_name",
                "ground_truth",
                "prediction",
                "restored_prediction",
                "restore_status",
                "restore_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    metrics["eval_generation_csv"] = str(csv_path)
    if graph_eval and graph_rows:
        rec_path = out_dir / f"epoch_{epoch:03d}.rec.txt"
        lab_path = out_dir / f"epoch_{epoch:03d}.lab.txt"
        graph_output = out_dir / f"epoch_{epoch:03d}.graph_result.txt"
        write_graph_matching_files(graph_rows, rec_path, lab_path)
        graph_result = run_graph_matching_tool(
            graph_tool_dir,
            rec_path,
            lab_path,
            graph_output,
            graph_workers,
        )
        metrics.update(graph_result.metrics)
        metrics["graph_output_txt"] = str(graph_result.output_path)
    model.train()
    return {f"eval_{key}": value for key, value in metrics.items()}


def save_checkpoint(
    model: torch.nn.Module,
    accelerator: Accelerator,
    output_dir: Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    if not isinstance(unwrapped, TexTellerMsdModel):
        raise TypeError(f"Unexpected model type: {type(unwrapped).__name__}")
    ensure_dir(output_dir)
    unwrapped.save_pretrained(output_dir)
    save_yaml(config, output_dir / "train_config.yaml")
    save_json(metrics, output_dir / "metrics.json")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training = dict(config.get("training") or {})
    seed = int(config.get("seed", 42))
    set_seed(seed)
    ensure_dir(args.output_dir)
    save_yaml(config, args.output_dir / "train_config.yaml")
    shutil.copy2(args.config, args.output_dir / "source_config.yaml")
    save_json(vars(args), args.output_dir / "training_args.json")

    mixed_precision = "no"
    if bool(training.get("bf16", False)):
        mixed_precision = "bf16"
    elif bool(training.get("fp16", False)):
        mixed_precision = "fp16"
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=mixed_precision,
    )
    logger.info("Accelerator initialized | processes=%s mixed_precision=%s", accelerator.num_processes, mixed_precision)

    vocab, vocab_report = load_rfl_vocab(args, config)
    if accelerator.is_main_process:
        save_json(vocab_report, args.output_dir / "rfl_vocab_report.json")
    processor = load_processor(args.pretrained_model_name_or_path, args.trust_remote_code)
    train_dataset, eval_dataset = build_datasets(args, config, vocab, processor)
    collator = TextellerMsdCollator(vocab, include_metadata=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("per_device_train_batch_size", 4)),
        collate_fn=collator,
        **dataloader_kwargs(config, train=True),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(training.get("per_device_eval_batch_size", 4)),
        collate_fn=collator,
        **dataloader_kwargs(config, train=False),
    )

    model = TexTellerMsdModel.from_pretrained(
        args.pretrained_model_name_or_path,
        vocab,
        msd_config_from_yaml(config),
        trust_remote_code=args.trust_remote_code,
    )
    if args.resume_from_checkpoint is not None:
        payload = torch.load(args.resume_from_checkpoint / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        logger.info("Loaded TexTeller+MSD weights from %s", args.resume_from_checkpoint)
    if bool(training.get("gradient_checkpointing", True)):
        model.enable_encoder_gradient_checkpointing()
    freeze_encoder_epochs = int(training.get("freeze_encoder_epochs", 0))
    if freeze_encoder_epochs > 0:
        model.freeze_encoder(True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    num_epochs = int(training.get("num_train_epochs", 10))
    updates_per_epoch = math.ceil(len(train_loader) / int(training.get("gradient_accumulation_steps", 1)))
    total_steps = max(1, updates_per_epoch * num_epochs)
    scheduler = get_scheduler(
        str(training.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(training.get("warmup_steps", 0)),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        eval_loader,
        scheduler,
    )

    best_metric_name = str(training.get("metric_for_best_model", "eval_loss"))
    greater_is_better = bool(training.get("greater_is_better", False))
    best_metric = -float("inf") if greater_is_better else float("inf")
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        if epoch == freeze_encoder_epochs + 1 and freeze_encoder_epochs > 0:
            accelerator.unwrap_model(model).freeze_encoder(False)
            logger.info("Unfroze TexTeller encoder at epoch %s.", epoch)
        start_time = time.time()
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=not accelerator.is_local_main_process)
        running_loss = 0.0
        running_seq = 0.0
        running_mem = 0.0
        for step, batch in enumerate(progress, start=1):
            with accelerator.accumulate(model):
                outputs = model(**model_inputs(batch))
                loss = outputs["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(training.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            global_step += 1
            running_loss += float(outputs["loss"].detach().cpu())
            running_seq += float(outputs["sequence_loss"].detach().cpu())
            running_mem += float(outputs["memory_loss"].detach().cpu())
            if accelerator.is_local_main_process:
                progress.set_postfix(loss=running_loss / step, seq=running_seq / step, mem=running_mem / step)

        train_metrics = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": running_loss / max(1, len(train_loader)),
            "train_sequence_loss": running_seq / max(1, len(train_loader)),
            "train_memory_loss": running_mem / max(1, len(train_loader)),
            "train_epoch_runtime": time.time() - start_time,
        }
        eval_metrics = evaluate_loss(model, eval_loader, accelerator)
        metrics = {**train_metrics, **eval_metrics}
        if accelerator.is_main_process and accelerator.num_processes == 1:
            generation_metrics = evaluate_generation(
                accelerator.unwrap_model(model),
                eval_loader,
                cfg=config,
                output_dir=args.output_dir,
                epoch=epoch,
                device=accelerator.device,
            )
            metrics.update(generation_metrics)
        if accelerator.is_main_process:
            logger.info("Epoch metrics | %s", metrics)
            save_json(metrics, args.output_dir / f"epoch_{epoch:03d}_metrics.json")
        metric_value = metrics.get(best_metric_name)
        if isinstance(metric_value, (int, float)):
            improved = metric_value > best_metric if greater_is_better else metric_value < best_metric
            if improved:
                best_metric = float(metric_value)
                save_checkpoint(
                    model,
                    accelerator,
                    args.output_dir / "best",
                    config=config,
                    metrics=metrics,
                )
        if bool(training.get("save_each_epoch", False)):
            save_checkpoint(
                model,
                accelerator,
                args.output_dir / f"checkpoint-epoch-{epoch:03d}",
                config=config,
                metrics=metrics,
            )

    save_checkpoint(model, accelerator, args.output_dir / "last", config=config, metrics={"step": global_step})
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Finished TexTeller+MSD training. Best %s=%s", best_metric_name, best_metric)


if __name__ == "__main__":
    main()
