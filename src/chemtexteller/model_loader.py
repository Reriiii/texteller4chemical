from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    VisionEncoderDecoderModel,
)

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForVision2Seq = None  # type: ignore[assignment]

from .tokenizer_utils import build_special_token_kwargs
from .utils import get_logger


logger = get_logger("model_loader")


@dataclass
class LoadedModelBundle:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    processor: Any | None
    source: str
    model_type: str


def _load_tokenizer(
    model_name_or_path: str,
    tokenizer_path: str | None,
    trust_remote_code: bool,
) -> PreTrainedTokenizerBase:
    source = tokenizer_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    special_kwargs = build_special_token_kwargs(tokenizer)
    if special_kwargs:
        tokenizer.add_special_tokens(special_kwargs)
    return tokenizer


def _try_texteller_tokenizer(
    model_name_or_path: str,
    tokenizer_path: str | None,
) -> PreTrainedTokenizerBase:
    try:
        texteller = importlib.import_module("texteller")
    except ImportError as exc:
        raise RuntimeError("Python package 'texteller' is not installed.") from exc
    if not hasattr(texteller, "load_tokenizer"):
        raise RuntimeError("Package 'texteller' does not expose load_tokenizer().")

    load_tokenizer = texteller.load_tokenizer
    candidates: list[tuple[str, tuple[Any, ...]]] = []
    if tokenizer_path:
        candidates.append((f"load_tokenizer({tokenizer_path})", (tokenizer_path,)))
    candidates.append((f"load_tokenizer({model_name_or_path})", (model_name_or_path,)))
    candidates.append(("load_tokenizer()", ()))

    errors: list[str] = []
    for label, args in candidates:
        try:
            tokenizer = load_tokenizer(*args)
            special_kwargs = build_special_token_kwargs(tokenizer)
            if special_kwargs and hasattr(tokenizer, "add_special_tokens"):
                tokenizer.add_special_tokens(special_kwargs)
            return tokenizer
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError("\n".join(errors))


def _load_processor(model_name_or_path: str, trust_remote_code: bool) -> Any | None:
    for cls in (AutoProcessor, AutoImageProcessor):
        try:
            return cls.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        except Exception as exc:
            logger.info("Could not load %s: %s", cls.__name__, exc)
    return None


def _try_texteller_package(
    model_name_or_path: str,
    device: str | None,
) -> torch.nn.Module:
    try:
        texteller = importlib.import_module("texteller")
    except ImportError as exc:
        raise RuntimeError("Python package 'texteller' is not installed.") from exc

    candidates: list[Any] = []
    if hasattr(texteller, "load_model"):
        candidates.append(texteller.load_model)
    if hasattr(texteller, "from_pretrained"):
        candidates.append(texteller.from_pretrained)
    for attr in ("TexTeller", "Texteller", "TexTellerModel", "Model"):
        obj = getattr(texteller, attr, None)
        if obj is not None and hasattr(obj, "from_pretrained"):
            candidates.append(obj.from_pretrained)

    errors: list[str] = []
    for factory in candidates:
        try:
            model = factory(model_name_or_path)
            if not isinstance(model, torch.nn.Module):
                raise TypeError(
                    f"texteller factory returned {type(model).__name__}, not torch.nn.Module"
                )
            if device:
                model.to(device)
            return model
        except Exception as exc:
            errors.append(f"{factory}: {exc}")
    if not candidates:
        errors.append("No from_pretrained-style API found in package 'texteller'.")
    raise RuntimeError("\n".join(errors))


def load_pretrained_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_path: str | None = None,
    device: str | None = None,
    trust_remote_code: bool = False,
) -> LoadedModelBundle:
    if not model_name_or_path:
        raise ValueError(
            "model_name_or_path is required. Pass a HuggingFace model id, a local "
            "checkpoint directory, or a local TexTeller checkpoint path."
        )

    errors: list[str] = []
    tokenizer: PreTrainedTokenizerBase | None = None
    try:
        tokenizer = _load_tokenizer(model_name_or_path, tokenizer_path, trust_remote_code)
    except Exception as exc:
        errors.append(f"AutoTokenizer: {exc}")
    processor = _load_processor(model_name_or_path, trust_remote_code)

    model_classes = [cls for cls in (AutoModelForVision2Seq, VisionEncoderDecoderModel) if cls is not None]
    for cls in model_classes:
        try:
            if tokenizer is None:
                raise RuntimeError("HF tokenizer could not be loaded for this checkpoint.")
            logger.info("Trying to load model with %s from %s", cls.__name__, model_name_or_path)
            model = cls.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            _configure_special_token_ids(model, tokenizer)
            if device:
                model.to(device)
            return LoadedModelBundle(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                source=model_name_or_path,
                model_type=cls.__name__,
            )
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")

    try:
        logger.info("Trying to load model through optional texteller package.")
        if tokenizer is None:
            tokenizer = _try_texteller_tokenizer(model_name_or_path, tokenizer_path)
        model = _try_texteller_package(model_name_or_path, device)
        _configure_special_token_ids(model, tokenizer)
        return LoadedModelBundle(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            source=model_name_or_path,
            model_type="texteller-package",
        )
    except Exception as exc:
        errors.append(f"texteller package: {exc}")

    detail = "\n".join(f"- {err}" for err in errors)
    raise RuntimeError(
        "Could not load pretrained TexTeller/model checkpoint.\n\n"
        f"Tried:\n{detail}\n\n"
        "What to check:\n"
        "1. Pass --pretrained_model_name_or_path with a real HF model id or local checkpoint.\n"
        "2. If TexTeller is only available through its package, run: uv add texteller\n"
        "3. If package APIs are not exposed, clone the repo:\n"
        "   git clone https://github.com/OleehyO/TexTeller external/TexTeller\n"
        "   then adapt model_loader.py to import that repo's model class/checkpoint loader.\n"
        "4. Do not fall back to training from scratch unless you intentionally pass --from_scratch."
    )


def _configure_special_token_ids(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    if getattr(config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    if getattr(config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    decoder_start = getattr(config, "decoder_start_token_id", None)
    if decoder_start is None:
        if tokenizer.bos_token_id is not None:
            config.decoder_start_token_id = tokenizer.bos_token_id
        elif tokenizer.cls_token_id is not None:
            config.decoder_start_token_id = tokenizer.cls_token_id
        elif tokenizer.eos_token_id is not None:
            config.decoder_start_token_id = tokenizer.eos_token_id

    decoder_cfg = getattr(config, "decoder", None)
    if decoder_cfg is not None:
        for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
            value = getattr(tokenizer, name, None)
            if value is not None and getattr(decoder_cfg, name, None) is None:
                setattr(decoder_cfg, name, value)


def resize_token_embeddings_if_needed(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    target_size = len(tokenizer)
    current_size = _current_embedding_size(model)
    if current_size == target_size:
        return

    logger.info("Resizing token embeddings from %s to %s", current_size, target_size)
    resize_errors: list[str] = []
    for obj_name, obj in (
        ("model", model),
        ("model.decoder", getattr(model, "decoder", None)),
        ("model.get_decoder()", _safe_get_decoder(model)),
    ):
        if obj is None or not hasattr(obj, "resize_token_embeddings"):
            continue
        try:
            obj.resize_token_embeddings(target_size)
            _configure_special_token_ids(model, tokenizer)
            return
        except Exception as exc:
            resize_errors.append(f"{obj_name}: {exc}")

    raise RuntimeError(
        "Tokenizer size differs from model vocabulary, but embeddings could not be resized.\n"
        f"Model embedding size: {current_size}; tokenizer size: {target_size}\n"
        f"Resize attempts: {resize_errors}\n"
        "If you extended the tokenizer, the TexTeller decoder embedding and output head must "
        "support resizing or be manually replaced."
    )


def _current_embedding_size(model: torch.nn.Module) -> int | None:
    for obj in (
        model,
        getattr(model, "decoder", None),
        _safe_get_decoder(model),
    ):
        if obj is None or not hasattr(obj, "get_input_embeddings"):
            continue
        embedding = obj.get_input_embeddings()
        if embedding is not None and hasattr(embedding, "num_embeddings"):
            return int(embedding.num_embeddings)
    return None


def _safe_get_decoder(model: torch.nn.Module) -> Any | None:
    if not hasattr(model, "get_decoder"):
        return None
    try:
        return model.get_decoder()
    except Exception:
        return None


def freeze_encoder_if_available(model: torch.nn.Module) -> int:
    encoder = None
    if hasattr(model, "get_encoder"):
        try:
            encoder = model.get_encoder()
        except Exception:
            encoder = None
    if encoder is None:
        encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("Could not find encoder to freeze on this model.")

    frozen = 0
    for parameter in encoder.parameters():
        parameter.requires_grad = False
        frozen += parameter.numel()
    return frozen


def enable_gradient_checkpointing_if_available(model: torch.nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return
    raise RuntimeError("This model does not expose gradient_checkpointing_enable().")


def add_texteller_repo_to_path(texteller_repo_path: str | None) -> None:
    if not texteller_repo_path:
        return
    import sys

    repo = Path(texteller_repo_path).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"TexTeller repo path does not exist: {repo}")
    sys.path.insert(0, str(repo))
