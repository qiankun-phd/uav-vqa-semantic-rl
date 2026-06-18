from __future__ import annotations

import math
from typing import Any


DEFAULT_SNR_BINS_DB = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]


def parse_snr_bins(values: str | list[int | float | str] | None) -> list[float]:
    if values is None or values == "":
        return []
    if isinstance(values, str):
        raw_values = [item.strip() for item in values.split(",") if item.strip()]
    else:
        raw_values = list(values)
    bins: list[float] = []
    for value in raw_values:
        text = str(value).strip().replace("snr_", "").replace("SNR_", "").replace("dB", "").replace("db", "")
        if text:
            bins.append(float(text))
    return sorted(dict.fromkeys(bins))


def snr_bins_from_config(cfg: dict[str, Any]) -> list[float]:
    bins_cfg = cfg.get("bins", {})
    vlm_cfg = cfg.get("vlm", {})
    return parse_snr_bins(
        bins_cfg.get("snr_db")
        or bins_cfg.get("snr_bins_db")
        or vlm_cfg.get("snr_bins_db")
        or []
    )


def snr_bin_label(snr_db: float | int | str) -> str:
    value = float(snr_db)
    if value.is_integer():
        return f"{int(value)}dB"
    return f"{value:g}dB"


def snr_db_from_label(label: str | int | float) -> float:
    if isinstance(label, (int, float)):
        return float(label)
    text = str(label).strip().replace("snr_", "").replace("SNR_", "").replace("dB", "").replace("db", "")
    return float(text)


def nearest_snr_bin(sensed_snr_db: float, snr_bins_db: list[float]) -> float:
    if not snr_bins_db:
        raise ValueError("snr_bins_db must not be empty")
    return min(snr_bins_db, key=lambda item: abs(float(item) - float(sensed_snr_db)))


def snr_db_to_bin_label(sensed_snr_db: float, snr_bins_db: list[float]) -> str:
    return snr_bin_label(nearest_snr_bin(sensed_snr_db, snr_bins_db))


def channel_bin_from_snr(snr_db: float) -> str:
    if snr_db <= 0:
        return "bad"
    if snr_db < 15:
        return "medium"
    return "good"


def legacy_channel_from_quality(link_quality: str) -> str:
    if link_quality in {"bad", "medium", "good"}:
        return link_quality
    try:
        return channel_bin_from_snr(snr_db_from_label(link_quality))
    except ValueError:
        return str(link_quality)


def is_snr_quality(link_quality: str) -> bool:
    try:
        snr_db_from_label(link_quality)
    except ValueError:
        return False
    return True


def snr_severity(link_quality: str, min_snr_db: float = -5.0, max_snr_db: float = 20.0) -> float:
    snr_db = snr_db_from_label(link_quality)
    if math.isclose(max_snr_db, min_snr_db):
        return 0.0
    normalized = (snr_db - min_snr_db) / (max_snr_db - min_snr_db)
    return max(0.0, min(1.0, 1.0 - normalized))


def snr_image_degradation(link_quality: str) -> dict[str, float | int]:
    severity = snr_severity(link_quality)
    return {
        "jpeg_quality": int(round(90 - 65 * severity)),
        "resize_scale": round(1.0 - 0.5 * severity, 3),
        "blur_radius": round(0.8 * severity, 3),
        "packet_loss_mask_rate": round(0.04 * severity, 4),
    }


def snr_light_degradation(link_quality: str) -> dict[str, float | int]:
    severity = snr_severity(link_quality)
    return {
        "drop_rate": round(0.35 * severity, 4),
        "class_corrupt_rate": round(0.12 * severity, 4),
        "bbox_quantization": max(1, int(round(1 + 15 * severity))),
        "confidence_threshold": round(0.45 * severity, 4),
    }


def degradation_config(kind: str, link_quality: str, cfg: dict[str, Any]) -> dict[str, Any]:
    vlm_cfg = cfg.get("vlm", {})
    snr_key = "snr_image_degradation" if kind == "image" else "snr_light_evidence_degradation"
    legacy_key = "image_degradation" if kind == "image" else "light_evidence_degradation"
    if link_quality in vlm_cfg.get(snr_key, {}):
        return dict(vlm_cfg[snr_key][link_quality])
    if link_quality in vlm_cfg.get(legacy_key, {}):
        return dict(vlm_cfg[legacy_key][link_quality])
    if is_snr_quality(link_quality):
        return snr_image_degradation(link_quality) if kind == "image" else snr_light_degradation(link_quality)
    legacy = legacy_channel_from_quality(link_quality)
    return dict(vlm_cfg.get(legacy_key, {}).get(legacy, {}))
