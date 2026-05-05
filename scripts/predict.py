from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemtexteller.inference import (
    autocast_context,
    generate_from_pixel_values,
    generation_kwargs,
    load_inference_config,
    merge_lora_for_inference,
    move_pixel_values,
    resolve_inference_dtype,
    set_generation_cache,
)
from chemtexteller.model_loader import load_pretrained_model_and_tokenizer
from chemtexteller.transforms import build_transform
from chemtexteller.utils import ensure_dir, setup_logging


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
    parser.add_argument("--no_merge_lora", action="store_true")
    parser.add_argument("--save_txt", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        inference_dtype = resolve_inference_dtype(args.dtype, device)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = load_inference_config(args.model_ckpt, args.config, args.max_new_tokens)
    bundle = load_pretrained_model_and_tokenizer(
        model_name_or_path=str(args.model_ckpt),
        tokenizer_path=args.tokenizer_path,
        device=str(device),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=inference_dtype,
    )
    bundle.model = merge_lora_for_inference(bundle.model, enabled=not args.no_merge_lora)
    if inference_dtype is not None:
        bundle.model.to(device=device, dtype=inference_dtype)
        logger.info("Using %s inference for generation.", inference_dtype)
    else:
        bundle.model.to(device)
    bundle.model.eval()
    set_generation_cache(bundle.model, enabled=True)
    transform = build_transform(config, train=False, processor=bundle.processor)
    with Image.open(args.image_path) as image:
        pixel_values = move_pixel_values(transform(image).unsqueeze(0), device, inference_dtype)

    gen_kwargs = generation_kwargs(
        bundle.model,
        bundle.tokenizer,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    autocast_ctx = autocast_context(device, inference_dtype)
    with torch.inference_mode(), autocast_ctx:
        generated = generate_from_pixel_values(bundle.model, pixel_values, gen_kwargs)
    prediction = bundle.tokenizer.decode(generated[0], skip_special_tokens=True)
    print(prediction)

    if args.save_txt is not None:
        ensure_dir(args.save_txt.parent)
        args.save_txt.write_text(prediction + "\n", encoding="utf-8")
        logger.info("Saved prediction to %s", args.save_txt)


if __name__ == "__main__":
    main()
