from __future__ import annotations

import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .utils import get_logger, load_yaml


logger = get_logger("inference")

DEFAULT_INFERENCE_CONFIG: dict[str, Any] = {
    "max_target_length": 768,
    "image_size": {
        "height": 448,
        "width": 448,
        "channels": 1,
        "pad_value": 255,
    },
    "augmentation": {"enabled": False},
}


def load_inference_config(
    model_ckpt: Path,
    config_path: Path | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    if config_path is not None:
        return load_yaml(config_path)
    candidate = model_ckpt / "train_config.yaml"
    if candidate.exists():
        return load_yaml(candidate)
    config = deepcopy(DEFAULT_INFERENCE_CONFIG)
    config["max_target_length"] = max_new_tokens
    return config


def resolve_inference_dtype(dtype_name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or dtype_name == "fp32":
        return None
    if dtype_name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        raise ValueError("bf16 inference was requested, but this CUDA device does not support bf16.")
    if dtype_name == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_generation_cache(model: torch.nn.Module, enabled: bool) -> None:
    seen: set[int] = set()
    candidates: list[Any] = [model]
    if hasattr(model, "get_base_model"):
        with contextlib.suppress(Exception):
            candidates.append(model.get_base_model())
    for attr in ("base_model", "model"):
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)

    for obj in candidates:
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        for config_attr in ("config", "generation_config"):
            config_obj = getattr(obj, config_attr, None)
            if config_obj is None:
                continue
            if hasattr(config_obj, "use_cache"):
                config_obj.use_cache = enabled
            decoder_cfg = getattr(config_obj, "decoder", None)
            if decoder_cfg is not None and hasattr(decoder_cfg, "use_cache"):
                decoder_cfg.use_cache = enabled


def _configured_token_id(model: torch.nn.Module, name: str) -> Any | None:
    for config_attr in ("generation_config", "config"):
        config_obj = getattr(model, config_attr, None)
        if config_obj is None:
            continue
        value = getattr(config_obj, name, None)
        if value is not None:
            return value
        decoder_cfg = getattr(config_obj, "decoder", None)
        if decoder_cfg is not None:
            value = getattr(decoder_cfg, name, None)
            if value is not None:
                return value
    return None


def _tokenizer_token_id(tokenizer: Any, name: str) -> Any | None:
    return getattr(tokenizer, name, None)


def generation_kwargs(
    model: torch.nn.Module,
    tokenizer: Any,
    num_beams: int,
    max_new_tokens: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }

    for name in ("pad_token_id", "eos_token_id"):
        value = _configured_token_id(model, name)
        if value is None:
            value = _tokenizer_token_id(tokenizer, name)
        if value is not None:
            kwargs[name] = value

    decoder_start = _configured_token_id(model, "decoder_start_token_id")
    if decoder_start is None:
        decoder_start = _tokenizer_token_id(tokenizer, "bos_token_id")
    if decoder_start is None:
        decoder_start = _tokenizer_token_id(tokenizer, "cls_token_id")
    if decoder_start is not None:
        kwargs["decoder_start_token_id"] = decoder_start
    return kwargs


def merge_lora_for_inference(model: torch.nn.Module, enabled: bool = True) -> torch.nn.Module:
    if not enabled or not hasattr(model, "merge_and_unload"):
        return model
    try:
        merged = model.merge_and_unload()
    except Exception as exc:
        logger.warning("Could not merge LoRA adapter for inference; using adapter model: %s", exc)
        return model
    logger.info("Merged LoRA adapter into the base model for inference.")
    return merged


def move_pixel_values(
    pixel_values: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    return pixel_values.to(
        device=device,
        dtype=dtype if dtype is not None else pixel_values.dtype,
        non_blocking=device.type == "cuda",
    )


def autocast_context(
    device: torch.device,
    dtype: torch.dtype | None,
) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def generate_from_pixel_values(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    gen_kwargs: dict[str, object],
) -> torch.Tensor:
    try:
        return model.generate(pixel_values=pixel_values, **gen_kwargs)
    except TypeError:
        return model.generate(inputs=pixel_values, **gen_kwargs)
