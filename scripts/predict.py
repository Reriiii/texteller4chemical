from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.model_loader import load_pretrained_model_and_tokenizer
from chemtexteller.transforms import build_transform
from chemtexteller.utils import ensure_dir, load_yaml, setup_logging


logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one EDU-CHEMC image.")
    parser.add_argument("--model_ckpt", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Inference dtype. auto uses bf16 on supported CUDA GPUs, otherwise fp16 on CUDA.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--save_txt", type=Path, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    if args.config is not None:
        return load_yaml(args.config)
    candidate = args.model_ckpt / "train_config.yaml"
    if candidate.exists():
        return load_yaml(candidate)
    return {
        "max_target_length": args.max_new_tokens,
        "image_size": {"height": 384, "width": 768, "channels": 3},
        "augmentation": {"enabled": False},
    }


def resolve_inference_dtype(dtype_name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or dtype_name == "fp32":
        return None
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def enable_generation_cache(model: torch.nn.Module) -> None:
    candidates = [model]
    if hasattr(model, "get_base_model"):
        with contextlib.suppress(Exception):
            candidates.append(model.get_base_model())
    for attr in ("base_model", "model"):
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)

    for obj in candidates:
        for config_attr in ("config", "generation_config"):
            config_obj = getattr(obj, config_attr, None)
            if config_obj is None:
                continue
            if hasattr(config_obj, "use_cache"):
                config_obj.use_cache = True
            decoder_cfg = getattr(config_obj, "decoder", None)
            if decoder_cfg is not None and hasattr(decoder_cfg, "use_cache"):
                decoder_cfg.use_cache = True


def generation_kwargs(
    tokenizer,
    num_beams: int,
    max_new_tokens: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        kwargs["decoder_start_token_id"] = tokenizer.bos_token_id
    elif tokenizer.cls_token_id is not None:
        kwargs["decoder_start_token_id"] = tokenizer.cls_token_id
    return kwargs


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args)
    bundle = load_pretrained_model_and_tokenizer(
        model_name_or_path=str(args.model_ckpt),
        tokenizer_path=args.tokenizer_path,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    bundle.model.eval()
    enable_generation_cache(bundle.model)
    device_obj = torch.device(device)
    inference_dtype = resolve_inference_dtype(args.dtype, device_obj)
    if inference_dtype is not None:
        bundle.model.to(dtype=inference_dtype)
        logger.info("Using %s inference for generation.", inference_dtype)
    transform = build_transform(config, train=False, processor=bundle.processor)
    with Image.open(args.image_path) as image:
        pixel_values = transform(image).unsqueeze(0).to(device)
    if inference_dtype is not None:
        pixel_values = pixel_values.to(dtype=inference_dtype)

    gen_kwargs = generation_kwargs(
        bundle.tokenizer,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=inference_dtype)
        if inference_dtype is not None
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        try:
            generated = bundle.model.generate(
                pixel_values=pixel_values,
                **gen_kwargs,
            )
        except TypeError:
            generated = bundle.model.generate(
                inputs=pixel_values,
                **gen_kwargs,
            )
    prediction = bundle.tokenizer.decode(generated[0], skip_special_tokens=True)
    print(prediction)

    if args.save_txt is not None:
        ensure_dir(args.save_txt.parent)
        args.save_txt.write_text(prediction + "\n", encoding="utf-8")
        logger.info("Saved prediction to %s", args.save_txt)


if __name__ == "__main__":
    main()
