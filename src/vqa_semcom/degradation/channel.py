from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from vqa_semcom.config import ensure_parent
from vqa_semcom.snr import degradation_config


def _deterministic_mask_boxes(width: int, height: int, rate: float) -> list[tuple[int, int, int, int]]:
    if rate <= 0:
        return []
    box_count = max(1, int(rate * 20))
    boxes = []
    cell_w = max(8, width // 12)
    cell_h = max(8, height // 12)
    for idx in range(box_count):
        x = (idx * 7919) % max(1, width - cell_w)
        y = (idx * 3571) % max(1, height - cell_h)
        boxes.append((x, y, min(width, x + cell_w), min(height, y + cell_h)))
    return boxes


def degrade_image(image_path: Path, out_dir: Path, channel_bin: str, cfg: dict) -> Path:
    deg_cfg = degradation_config("image", channel_bin, cfg)
    quality = int(deg_cfg.get("jpeg_quality", 90))
    resize_scale = float(deg_cfg.get("resize_scale", 1.0))
    blur_radius = float(deg_cfg.get("blur_radius", 0.0))
    mask_rate = float(deg_cfg.get("packet_loss_mask_rate", 0.0))
    out_path = out_dir / channel_bin / f"{image_path.stem}_{channel_bin}.jpg"
    ensure_parent(out_path)
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        if resize_scale < 1.0:
            small_size = (max(1, int(rgb.width * resize_scale)), max(1, int(rgb.height * resize_scale)))
            rgb = rgb.resize(small_size, Image.Resampling.BILINEAR).resize((img.width, img.height), Image.Resampling.BILINEAR)
        if blur_radius > 0:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        if mask_rate > 0:
            draw = ImageDraw.Draw(rgb)
            for box in _deterministic_mask_boxes(rgb.width, rgb.height, mask_rate):
                draw.rectangle(box, fill=(16, 16, 16))
        rgb.save(out_path, format="JPEG", quality=quality, optimize=True)
    return out_path


def degrade_pil_image(rgb: Image.Image, out_path: Path, channel_bin: str, cfg: dict) -> Path:
    deg_cfg = degradation_config("image", channel_bin, cfg)
    quality = int(deg_cfg.get("jpeg_quality", 90))
    resize_scale = float(deg_cfg.get("resize_scale", 1.0))
    blur_radius = float(deg_cfg.get("blur_radius", 0.0))
    mask_rate = float(deg_cfg.get("packet_loss_mask_rate", 0.0))
    ensure_parent(out_path)
    img = rgb.convert("RGB")
    original_size = img.size
    if resize_scale < 1.0:
        small_size = (max(1, int(img.width * resize_scale)), max(1, int(img.height * resize_scale)))
        img = img.resize(small_size, Image.Resampling.BILINEAR).resize(original_size, Image.Resampling.BILINEAR)
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if mask_rate > 0:
        draw = ImageDraw.Draw(img)
        for box in _deterministic_mask_boxes(img.width, img.height, mask_rate):
            draw.rectangle(box, fill=(16, 16, 16))
    img.save(out_path, format="JPEG", quality=quality, optimize=True)
    return out_path
