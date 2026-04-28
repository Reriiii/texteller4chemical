from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@dataclass
class ImagePreprocessConfig:
    height: int = 384
    width: int = 768
    channels: int = 3
    pad_value: int = 255
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
    channels = int(image.get("channels", 3))
    return ImagePreprocessConfig(
        height=int(image.get("height", 384)),
        width=int(image.get("width", 768)),
        channels=channels,
        pad_value=int(image.get("pad_value", 255)),
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
        cfg.normalize_mean = _as_tuple(mean, cfg.channels)
    if cfg.normalize_std is None and std is not None:
        cfg.normalize_std = _as_tuple(std, cfg.channels)
    return cfg


class ResizePadTransform:
    def __init__(self, cfg: ImagePreprocessConfig, train: bool = False) -> None:
        self.cfg = cfg
        self.train = train
        self.color_jitter = transforms.ColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
        )
        self.random_erasing = transforms.RandomErasing(
            p=cfg.random_erasing_prob,
            scale=(0.005, 0.02),
            ratio=(0.3, 3.3),
            value=1.0,
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        mode = "RGB" if self.cfg.channels == 3 else "L"
        image = image.convert(mode)

        if self.train and self.cfg.augmentation_enabled:
            image = self._augment_pil(image)

        image = self._resize_and_pad(image)
        tensor = F.to_tensor(image)

        if self.train and self.cfg.augmentation_enabled and self.cfg.random_erasing_prob > 0:
            tensor = self.random_erasing(tensor)

        if self.cfg.normalize_mean is not None and self.cfg.normalize_std is not None:
            tensor = F.normalize(tensor, mean=self.cfg.normalize_mean, std=self.cfg.normalize_std)
        return tensor

    def _augment_pil(self, image: Image.Image) -> Image.Image:
        if self.cfg.channels == 3 and (self.cfg.brightness > 0 or self.cfg.contrast > 0):
            image = self.color_jitter(image)
        if self.cfg.affine_degrees > 0 or self.cfg.affine_translate > 0:
            image = transforms.RandomAffine(
                degrees=self.cfg.affine_degrees,
                translate=(self.cfg.affine_translate, self.cfg.affine_translate),
                interpolation=InterpolationMode.BILINEAR,
                fill=self.cfg.pad_value,
            )(image)
        if self.cfg.gaussian_blur_prob > 0:
            if torch.rand(1).item() < self.cfg.gaussian_blur_prob:
                image = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))(image)
        return image

    def _resize_and_pad(self, image: Image.Image) -> Image.Image:
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"Invalid image size: {image.size}")
        scale = min(self.cfg.width / src_w, self.cfg.height / src_h)
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
        left = (self.cfg.width - new_w) // 2
        top = (self.cfg.height - new_h) // 2
        canvas.paste(image, (left, top))
        return canvas


def build_transform(
    config: dict[str, Any],
    train: bool,
    processor: Any | None = None,
) -> ResizePadTransform:
    cfg = image_config_from_dict(config)
    cfg = apply_processor_stats(cfg, processor)
    return ResizePadTransform(cfg, train=train)
