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


def _is_peft_checkpoint(model_name_or_path: str) -> bool:
    path = Path(model_name_or_path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def _hf_model_classes() -> list[Any]:
    return [cls for cls in (AutoModelForVision2Seq, VisionEncoderDecoderModel) if cls is not None]


def _load_hf_model(
    model_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
    device: str | None,
    trust_remote_code: bool,
) -> tuple[torch.nn.Module, str]:
    errors: list[str] = []
    for cls in _hf_model_classes():
        try:
            logger.info("Trying to load model with %s from %s", cls.__name__, model_name_or_path)
            model = cls.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            _configure_special_token_ids(model, tokenizer)
            if device:
                model.to(device)
            return model, cls.__name__
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("\n".join(errors))


def _load_peft_checkpoint(
    adapter_path: str,
    tokenizer_path: str | None,
    device: str | None,
    trust_remote_code: bool,
) -> LoadedModelBundle:
    try:
        from peft import PeftConfig, PeftModel
    except ImportError as exc:
        raise RuntimeError("LoRA/PEFT checkpoint requested but peft is not installed.") from exc

    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name_or_path = getattr(peft_config, "base_model_name_or_path", None)
    if not base_model_name_or_path:
        raise RuntimeError(
            f"PEFT adapter at {adapter_path} does not record base_model_name_or_path."
        )

    tokenizer_errors: list[str] = []
    tokenizer: PreTrainedTokenizerBase | None = None
    for source in (tokenizer_path, adapter_path, base_model_name_or_path):
        if source is None:
            continue
        try:
            tokenizer = _load_tokenizer(str(source), None, trust_remote_code)
            break
        except Exception as exc:
            tokenizer_errors.append(f"{source}: {exc}")
    if tokenizer is None:
        detail = "\n".join(f"- {err}" for err in tokenizer_errors)
        raise RuntimeError(f"Could not load tokenizer for PEFT checkpoint:\n{detail}")

    processor = _load_processor(adapter_path, trust_remote_code)
    if processor is None:
        processor = _load_processor(str(base_model_name_or_path), trust_remote_code)

    base_model, base_model_type = _load_hf_model(
        str(base_model_name_or_path),
        tokenizer=tokenizer,
        device=None,
        trust_remote_code=trust_remote_code,
    )
    resize_token_embeddings_if_needed(base_model, tokenizer)
    logger.info("Loading PEFT adapter from %s onto %s", adapter_path, base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    _configure_special_token_ids(model, tokenizer)
    if device:
        model.to(device)
    return LoadedModelBundle(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        source=adapter_path,
        model_type=f"{base_model_type}+peft",
    )


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

    if _is_peft_checkpoint(model_name_or_path):
        return _load_peft_checkpoint(
            adapter_path=model_name_or_path,
            tokenizer_path=tokenizer_path,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    errors: list[str] = []
    tokenizer: PreTrainedTokenizerBase | None = None
    try:
        tokenizer = _load_tokenizer(model_name_or_path, tokenizer_path, trust_remote_code)
    except Exception as exc:
        errors.append(f"AutoTokenizer: {exc}")
    processor = _load_processor(model_name_or_path, trust_remote_code)

    try:
        if tokenizer is None:
            raise RuntimeError("HF tokenizer could not be loaded for this checkpoint.")
        model, model_type = _load_hf_model(
            model_name_or_path,
            tokenizer=tokenizer,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        return LoadedModelBundle(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            source=model_name_or_path,
            model_type=model_type,
        )
    except Exception as exc:
        errors.append(str(exc))

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
