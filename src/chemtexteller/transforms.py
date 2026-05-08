from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


logger = logging.getLogger("chemtexteller.transforms")


@dataclass
class ImagePreprocessConfig:
    height: int = 448
    width: int = 448
    channels: int = 1
    pad_value: int = 255
    trim_white_border: bool = False
    trim_threshold: int = 15
    resize_mode: str = "fit"
    pad_position: str = "center"
    normalize_mean: tuple[float, ...] | None = None
    normalize_std: tuple[float, ...] | None = None
    augmentation_enabled: bool = False
    brightness: float = 0.10
    contrast: float = 0.10
    gaussian_blur_prob: float = 0.05
    affine_degrees: float = 2.0
    affine_translate: float = 0.02
    random_erasing_prob: float = 0.0


def _as_tuple(value: Any, channels: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return tuple(float(value) for _ in range(channels))
    values = tuple(float(v) for v in value)
    if len(values) == channels:
        return values
    if len(values) == 1:
        return tuple(values[0] for _ in range(channels))
    raise ValueError(f"Expected {channels} normalization values, got {values}")


def image_config_from_dict(config: dict[str, Any]) -> ImagePreprocessConfig:
    image = config.get("image_size", {})
    aug = config.get("augmentation", {})
    channels = int(image.get("channels", 1))
    return ImagePreprocessConfig(
        height=int(image.get("height", 448)),
        width=int(image.get("width", 448)),
        channels=channels,
        pad_value=int(image.get("pad_value", 255)),
        trim_white_border=bool(image.get("trim_white_border", False)),
        trim_threshold=int(image.get("trim_threshold", 15)),
        resize_mode=str(image.get("resize_mode", "fit")),
        pad_position=str(image.get("pad_position", "center")),
        normalize_mean=_as_tuple(image.get("normalize_mean"), channels),
        normalize_std=_as_tuple(image.get("normalize_std"), channels),
        augmentation_enabled=bool(aug.get("enabled", False)),
        brightness=float(aug.get("brightness", 0.10)),
        contrast=float(aug.get("contrast", 0.10)),
        gaussian_blur_prob=float(aug.get("gaussian_blur_prob", 0.05)),
        affine_degrees=float(aug.get("affine_degrees", 2.0)),
        affine_translate=float(aug.get("affine_translate", 0.02)),
        random_erasing_prob=float(aug.get("random_erasing_prob", 0.0)),
    )


def apply_processor_stats(
    cfg: ImagePreprocessConfig,
    processor: Any | None,
) -> ImagePreprocessConfig:
    image_processor = getattr(processor, "image_processor", processor)
    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)
    if cfg.normalize_mean is None and mean is not None:
        try:
            cfg.normalize_mean = _as_tuple(mean, cfg.channels)
        except ValueError:
            logger.warning(
                "Ignoring processor image_mean=%s for %s-channel input.",
                mean,
                cfg.channels,
            )
    if cfg.normalize_std is None and std is not None:
        try:
            cfg.normalize_std = _as_tuple(std, cfg.channels)
        except ValueError:
            logger.warning(
                "Ignoring processor image_std=%s for %s-channel input.",
                std,
                cfg.channels,
            )
    return cfg


class ResizePadTransform:
    def __init__(self, cfg: ImagePreprocessConfig, train: bool = False) -> None:
        self.cfg = cfg
        self.train = train
        use_augmentation = train and cfg.augmentation_enabled
        self.color_jitter = (
            transforms.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
            )
            if use_augmentation and (cfg.brightness > 0 or cfg.contrast > 0)
            else None
        )
        self.random_affine = (
            transforms.RandomAffine(
                degrees=cfg.affine_degrees,
                translate=(cfg.affine_translate, cfg.affine_translate),
                interpolation=InterpolationMode.BILINEAR,
                fill=cfg.pad_value,
            )
            if use_augmentation and (cfg.affine_degrees > 0 or cfg.affine_translate > 0)
            else None
        )
        self.gaussian_blur = (
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))
            if use_augmentation and cfg.gaussian_blur_prob > 0
            else None
        )
        self.random_erasing = transforms.RandomErasing(
            p=cfg.random_erasing_prob,
            scale=(0.005, 0.02),
            ratio=(0.3, 3.3),
            value=1.0,
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        mode = "RGB" if self.cfg.channels == 3 else "L"
        image = image.convert("RGB")

        if self.cfg.trim_white_border:
            image = self._trim_white_border(image)

        image = image.convert(mode)

        if self.train and self.cfg.augmentation_enabled:
            image = self._augment_pil(image)
            if self.cfg.trim_white_border:
                image = self._trim_white_border(image).convert(mode)

        image = self._resize_and_pad(image)
        tensor = F.to_tensor(image)

        if self.train and self.cfg.augmentation_enabled and self.cfg.random_erasing_prob > 0:
            tensor = self.random_erasing(tensor)

        if self.cfg.normalize_mean is not None and self.cfg.normalize_std is not None:
            tensor = F.normalize(tensor, mean=self.cfg.normalize_mean, std=self.cfg.normalize_std)
        return tensor

    def _augment_pil(self, image: Image.Image) -> Image.Image:
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        if self.random_affine is not None:
            image = self.random_affine(image)
        if self.gaussian_blur is not None:
            if torch.rand(1).item() < self.cfg.gaussian_blur_prob:
                image = self.gaussian_blur(image)
        return image

    def _trim_white_border(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if arr.size == 0:
            return image

        corners = (
            tuple(arr[0, 0]),
            tuple(arr[0, -1]),
            tuple(arr[-1, 0]),
            tuple(arr[-1, -1]),
        )
        bg_color = np.asarray(Counter(corners).most_common(1)[0][0], dtype=np.int16)
        diff = np.abs(arr.astype(np.int16) - bg_color)
        mask = diff.max(axis=2) > self.cfg.trim_threshold
        if not mask.any():
            return image

        ys, xs = np.where(mask)
        left = int(xs.min())
        right = int(xs.max()) + 1
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        if right <= left or bottom <= top:
            return image
        return image.crop((left, top, right, bottom))

    def _resize_and_pad(self, image: Image.Image) -> Image.Image:
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"Invalid image size: {image.size}")
        scale = self._resize_scale(src_w, src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        mode = "RGB" if self.cfg.channels == 3 else "L"
        fill = (
            (self.cfg.pad_value,) * 3
            if mode == "RGB"
            else self.cfg.pad_value
        )
        canvas = Image.new(mode, (self.cfg.width, self.cfg.height), fill)
        left, top = self._paste_offset(new_w, new_h)
        canvas.paste(image, (left, top))
        return canvas

    def _resize_scale(self, src_w: int, src_h: int) -> float:
        resize_mode = self.cfg.resize_mode.lower()
        if resize_mode == "fit":
            return min(self.cfg.width / src_w, self.cfg.height / src_h)
        if resize_mode != "texteller":
            raise ValueError("resize_mode must be 'fit' or 'texteller'.")

        target_short = max(1, min(self.cfg.width, self.cfg.height) - 1)
        target_long = max(self.cfg.width, self.cfg.height)
        src_short = min(src_w, src_h)
        src_long = max(src_w, src_h)
        scale = target_short / src_short
        if round(src_long * scale) > target_long:
            scale = target_long / src_long
        return scale

    def _paste_offset(self, new_w: int, new_h: int) -> tuple[int, int]:
        pad_position = self.cfg.pad_position.lower()
        if pad_position in {"top_left", "topleft"}:
            return 0, 0
        if pad_position == "center":
            return (self.cfg.width - new_w) // 2, (self.cfg.height - new_h) // 2
        raise ValueError("pad_position must be 'center' or 'top_left'.")


def build_transform(
    config: dict[str, Any],
    train: bool,
    processor: Any | None = None,
) -> ResizePadTransform:
    cfg = image_config_from_dict(config)
    cfg = apply_processor_stats(cfg, processor)
    return ResizePadTransform(cfg, train=train)
