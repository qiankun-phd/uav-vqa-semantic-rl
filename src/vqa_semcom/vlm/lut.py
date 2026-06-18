from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

from vqa_semcom.config import ensure_parent
VLM_LUT_FIELDNAMES = [
    "question_type",
    "service_level",
    "channel_bin",
    "snr_bin",
    "view_quality_bin",
    "freshness_bin",
    "risk_level",
    "expected_accuracy",
    "sample_count",
    "std_or_ci",
    "avg_payload_bytes",
    "avg_payload_kb",
]


@dataclass(frozen=True)
class VlmLUTRow:
    question_type: str
    service_level: int
    channel_bin: str
    snr_bin: str
    view_quality_bin: str
    freshness_bin: str
    risk_level: str
    expected_accuracy: float
    sample_count: int
    std_or_ci: float
    avg_payload_bytes: float
    avg_payload_kb: float


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_correct(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def _payload_value(row: dict[str, str]) -> float:
    raw = row.get("payload_bytes", "")
    if raw in {"", None}:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def build_lut_from_predictions(predictions: list[dict[str, str]]) -> list[VlmLUTRow]:
    grouped: dict[tuple[str, int, str, str, str, str, str], list[float]] = defaultdict(list)
    payloads: dict[tuple[str, int, str, str, str, str, str], list[float]] = defaultdict(list)
    for row in predictions:
        snr_bin = row.get("snr_bin", "")
        key = (
            row["question_type"],
            int(row["service_level"]),
            row["channel_bin"],
            snr_bin,
            row["view_quality_bin"],
            row["freshness_bin"],
            row["risk_level"],
        )
        grouped[key].append(1.0 if _is_correct(row["correct"]) else 0.0)
        payloads[key].append(_payload_value(row))
    lut_rows: list[VlmLUTRow] = []
    for key, values in sorted(grouped.items()):
        mu = mean(values)
        std = pstdev(values) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(max(1, len(values)))
        avg_payload = mean(payloads[key]) if payloads[key] else 0.0
        lut_rows.append(
            VlmLUTRow(
                *key,
                round(mu, 6),
                len(values),
                round(ci95, 6),
                round(avg_payload, 3),
                round(avg_payload / 1024.0, 6),
            )
        )
    return lut_rows


def write_lut_csv(rows: list[VlmLUTRow], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VLM_LUT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
