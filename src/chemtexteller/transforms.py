from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
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
    augmentation_profile: str = "light"
    brightness: float = 0.10
    contrast: float = 0.10
    gaussian_blur_prob: float = 0.05
    affine_degrees: float = 2.0
    affine_translate: float = 0.02
    random_erasing_prob: float = 0.0
    random_resize_min: float = 0.75
    random_resize_max: float = 1.15
    rotate_prob: float = 0.20
    rotate_degrees: float = 5.0
    random_border_max: int = 25
    min_augmented_size: int = 30
    augraphy_enabled: bool = False


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
        augmentation_profile=str(aug.get("profile", "light")),
        brightness=float(aug.get("brightness", 0.10)),
        contrast=float(aug.get("contrast", 0.10)),
        gaussian_blur_prob=float(aug.get("gaussian_blur_prob", 0.05)),
        affine_degrees=float(aug.get("affine_degrees", 2.0)),
        affine_translate=float(aug.get("affine_translate", 0.02)),
        random_erasing_prob=float(aug.get("random_erasing_prob", 0.0)),
        random_resize_min=float(aug.get("random_resize_min", 0.75)),
        random_resize_max=float(aug.get("random_resize_max", 1.15)),
        rotate_prob=float(aug.get("rotate_prob", 0.20)),
        rotate_degrees=float(aug.get("rotate_degrees", 5.0)),
        random_border_max=int(aug.get("random_border_max", 25)),
        min_augmented_size=int(aug.get("min_augmented_size", 30)),
        augraphy_enabled=bool(aug.get("augraphy_enabled", False)),
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
        self.augmentation_profile = cfg.augmentation_profile.lower()
        if self.augmentation_profile in {"texteller", "texteller_ocr", "ocr"}:
            self.augmentation_profile = "texteller_ocr"
        if self.augmentation_profile not in {"light", "texteller_ocr", "none"}:
            raise ValueError(
                "augmentation.profile must be 'light', 'texteller_ocr', or 'none'."
            )
        self._augraphy_pipeline: Any | None = None
        use_light_augmentation = (
            train
            and cfg.augmentation_enabled
            and self.augmentation_profile == "light"
        )
        self.color_jitter = (
            transforms.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
            )
            if use_light_augmentation and (cfg.brightness > 0 or cfg.contrast > 0)
            else None
        )
        self.random_affine = (
            transforms.RandomAffine(
                degrees=cfg.affine_degrees,
                translate=(cfg.affine_translate, cfg.affine_translate),
                interpolation=InterpolationMode.BILINEAR,
                fill=cfg.pad_value,
            )
            if use_light_augmentation and (cfg.affine_degrees > 0 or cfg.affine_translate > 0)
            else None
        )
        self.gaussian_blur = (
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))
            if use_light_augmentation and cfg.gaussian_blur_prob > 0
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

        if (
            self.train
            and self.cfg.augmentation_enabled
            and self.augmentation_profile == "texteller_ocr"
        ):
            image = self._augment_texteller_ocr(image)
        else:
            if self.cfg.trim_white_border:
                image = self._trim_white_border(image)

            image = image.convert(mode)

            if (
                self.train
                and self.cfg.augmentation_enabled
                and self.augmentation_profile == "light"
            ):
                image = self._augment_pil(image)
                if self.cfg.trim_white_border:
                    image = self._trim_white_border(image).convert(mode)

        image = image.convert(mode)

        image = self._resize_and_pad(image)
        tensor = F.to_tensor(image)

        if (
            self.train
            and self.cfg.augmentation_enabled
            and self.augmentation_profile == "light"
            and self.cfg.random_erasing_prob > 0
        ):
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

    def _augment_texteller_ocr(self, image: Image.Image) -> Image.Image:
        image = self._random_resize(image)
        if self.cfg.trim_white_border:
            image = self._trim_white_border(image)

        if self.cfg.rotate_prob > 0 and random.random() < self.cfg.rotate_prob:
            rotate_degrees = self.cfg.rotate_degrees
            if float(rotate_degrees).is_integer():
                angle = random.randint(-int(rotate_degrees), int(rotate_degrees))
            else:
                angle = random.uniform(-rotate_degrees, rotate_degrees)
            image = image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=(255, 255, 255),
            )

        image = self._add_white_border(image, self.cfg.random_border_max)

        if self.cfg.augraphy_enabled:
            image = self._apply_augraphy(image)

        # TexTeller applies the normal inference transform after OCR augmentation;
        # that transform trims the border again before grayscale/resize/pad.
        if self.cfg.trim_white_border:
            image = self._trim_white_border(image)
        return image

    def _random_resize(self, image: Image.Image) -> Image.Image:
        min_ratio = self.cfg.random_resize_min
        max_ratio = self.cfg.random_resize_max
        if min_ratio <= 0 or max_ratio <= 0:
            raise ValueError("random_resize_min/max must be positive.")
        if min_ratio > max_ratio:
            raise ValueError("random_resize_min must be <= random_resize_max.")
        ratio = random.uniform(min_ratio, max_ratio)
        if abs(ratio - 1.0) < 1e-6:
            return image
        new_w = max(1, int(image.width * ratio))
        new_h = max(1, int(image.height * ratio))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _add_white_border(self, image: Image.Image, max_size: int) -> Image.Image:
        max_size = max(0, int(max_size))
        left, top, right, bottom = [random.randint(0, max_size) for _ in range(4)]
        pad_h = top + bottom
        pad_w = left + right
        if pad_h + image.height < self.cfg.min_augmented_size:
            extra = int((self.cfg.min_augmented_size - (pad_h + image.height)) * 0.5) + 1
            top += extra
            bottom += extra
        if pad_w + image.width < self.cfg.min_augmented_size:
            extra = int((self.cfg.min_augmented_size - (pad_w + image.width)) * 0.5) + 1
            left += extra
            right += extra
        return ImageOps.expand(
            image.convert("RGB"),
            border=(left, top, right, bottom),
            fill=(255, 255, 255),
        )

    def _apply_augraphy(self, image: Image.Image) -> Image.Image:
        pipeline = self._get_augraphy_pipeline()
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        augmented = pipeline(arr)
        if isinstance(augmented, tuple):
            augmented = augmented[0]
        if isinstance(augmented, list):
            augmented = augmented[0]
        augmented = np.asarray(augmented, dtype=np.uint8)
        if augmented.ndim == 2:
            return Image.fromarray(augmented).convert("RGB")
        if augmented.ndim == 3 and augmented.shape[2] > 3:
            augmented = augmented[:, :, :3]
        return Image.fromarray(augmented).convert("RGB")

    def _get_augraphy_pipeline(self) -> Any:
        if self._augraphy_pipeline is None:
            self._augraphy_pipeline = _build_texteller_augraphy_pipeline()
        return self._augraphy_pipeline

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


def _build_texteller_augraphy_pipeline() -> Any:
    try:
        from augraphy import (
            AugraphyPipeline,
            Brightness,
            BrightnessTexturize,
            ColorShift,
            DirtyDrum,
            Dithering,
            Gamma,
            InkBleed,
            InkColorSwap,
            InkShifter,
            Jpeg,
            LightingGradient,
            LinesDegradation,
            NoiseTexturize,
            OneOf,
            SubtleNoise,
        )
    except ImportError as exc:
        raise RuntimeError(
            "augmentation.profile=texteller_ocr with augraphy_enabled=true requires "
            "the 'augraphy' package. Install it with `uv add augraphy>=8.2.6` "
            "or run `uv sync` after adding the dependency."
        ) from exc

    ink_phase = [
        InkColorSwap(
            ink_swap_color="random",
            ink_swap_sequence_number_range=(5, 10),
            ink_swap_min_width_range=(2, 3),
            ink_swap_max_width_range=(100, 120),
            ink_swap_min_height_range=(2, 3),
            ink_swap_max_height_range=(100, 120),
            ink_swap_min_area_range=(10, 20),
            ink_swap_max_area_range=(400, 500),
            p=0.2,
        ),
        LinesDegradation(
            line_roi=(0.0, 0.0, 1.0, 1.0),
            line_gradient_range=(32, 255),
            line_gradient_direction=(0, 2),
            line_split_probability=(0.2, 0.4),
            line_replacement_value=(250, 255),
            line_min_length=(30, 40),
            line_long_to_short_ratio=(5, 7),
            line_replacement_probability=(0.4, 0.5),
            line_replacement_thickness=(1, 3),
            p=0.2,
        ),
        OneOf(
            [
                Dithering(dither="floyd-steinberg", order=(3, 5)),
                InkBleed(
                    intensity_range=(0.1, 0.2),
                    kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
                    severity=(0.4, 0.6),
                ),
            ],
            p=0.2,
        ),
        InkShifter(
            text_shift_scale_range=(18, 27),
            text_shift_factor_range=(1, 4),
            text_fade_range=(0, 2),
            blur_kernel_size=(5, 5),
            blur_sigma=0,
            noise_type="perlin",
            p=0.2,
        ),
    ]

    paper_phase = [
        NoiseTexturize(
            sigma_range=(3, 10),
            turbulence_range=(2, 5),
            texture_width_range=(300, 500),
            texture_height_range=(300, 500),
            p=0.2,
        ),
        BrightnessTexturize(texturize_range=(0.9, 0.99), deviation=0.03, p=0.2),
    ]

    post_phase = [
        ColorShift(
            color_shift_offset_x_range=(3, 5),
            color_shift_offset_y_range=(3, 5),
            color_shift_iterations=(2, 3),
            color_shift_brightness_range=(0.9, 1.1),
            color_shift_gaussian_kernel_range=(3, 3),
            p=0.2,
        ),
        DirtyDrum(
            line_width_range=(1, 6),
            line_concentration=random.uniform(0.05, 0.15),
            direction=random.randint(0, 2),
            noise_intensity=random.uniform(0.6, 0.95),
            noise_value=(64, 224),
            ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
            sigmaX=0,
            p=0.2,
        ),
        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    linear_decay_rate=None,
                    transparency=None,
                ),
                Brightness(
                    brightness_range=(0.9, 1.1),
                    min_brightness=0,
                    min_brightness_value=(120, 150),
                ),
                Gamma(gamma_range=(0.9, 1.1)),
            ],
            p=0.2,
        ),
        OneOf(
            [
                SubtleNoise(subtle_range=random.randint(5, 10)),
                Jpeg(quality_range=(70, 95)),
            ],
            p=0.2,
        ),
    ]

    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=[],
        log=False,
    )


def build_transform(
    config: dict[str, Any],
    train: bool,
    processor: Any | None = None,
) -> ResizePadTransform:
    cfg = image_config_from_dict(config)
    cfg = apply_processor_stats(cfg, processor)
    return ResizePadTransform(cfg, train=train)
