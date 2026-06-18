from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from vqa_semcom.config import ensure_parent


LUT_FIELDNAMES = [
    "question_type",
    "service_level",
    "channel_bin",
    "view_quality_bin",
    "freshness_bin",
    "risk_level",
    "expected_accuracy",
    "sample_count",
    "std_or_ci",
]


@dataclass(frozen=True)
class LUTRow:
    question_type: str
    service_level: int
    channel_bin: str
    view_quality_bin: str
    freshness_bin: str
    risk_level: str
    expected_accuracy: float
    sample_count: int
    std_or_ci: float


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_accuracy(
    question_type: str,
    service_level: int,
    channel_bin: str,
    view_quality_bin: str,
    freshness_bin: str,
    risk_level: str,
    evaluator_cfg: dict[str, Any],
) -> float:
    base = evaluator_cfg["base_accuracy"][question_type]
    service_gain = evaluator_cfg["service_gain"][str(service_level)]
    channel = evaluator_cfg["channel_quality"][channel_bin]
    view = evaluator_cfg["view_quality"][view_quality_bin]
    freshness = evaluator_cfg["freshness_quality"][str(service_level)][freshness_bin]
    risk = evaluator_cfg["risk_penalty"][risk_level]
    # Service level 0 is cache-dominated, so channel/view should have only mild impact.
    if service_level == 0:
        channel = 0.96 + 0.04 * channel
        view = 0.94 + 0.06 * view
    elif service_level == 2:
        freshness = 0.98 + 0.02 * freshness
    acc = base * service_gain * channel * view * freshness * risk
    if service_level == 1:
        boost = evaluator_cfg.get("light_evidence_boost", {}).get(question_type, 1.0)
        if view_quality_bin in {"medium", "good"} and channel_bin in {"medium", "good"}:
            acc *= boost
        elif question_type in {"presence", "counting"} and view_quality_bin == "good":
            acc *= 1.04
    if service_level == 2:
        # High-fidelity evidence should usually dominate lightweight evidence.
        acc += 0.035
    return round(_clip(acc), 6)


def _deterministic_task_jitter(task: dict[str, str], key: tuple[str, int, str, str, str, str], evaluator_cfg: dict[str, Any]) -> float:
    scale = float(evaluator_cfg.get("ci_noise_scale", 0.0))
    if scale <= 0:
        return 1.0
    raw = "|".join([task.get("image_id", ""), task.get("target_class", ""), task.get("object_count", ""), *map(str, key)])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    centered = (unit - 0.5) * 2.0
    density = float(task.get("density_score", "0") or 0.0)
    density_factor = min(1.6, 1.0 + density / 80.0)
    return max(0.75, min(1.25, 1.0 + centered * scale * density_factor))


def read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_lut(tasks: list[dict[str, str]], cfg: dict[str, Any]) -> list[LUTRow]:
    grouped: dict[tuple[str, int, str, str, str, str], list[float]] = defaultdict(list)
    for task in tasks:
        question_type = task["question_type"]
        view_bin = task["view_quality_bin"]
        risk_level = task["risk_level"]
        for service_level in cfg["bins"]["service_levels"]:
            for channel_bin in cfg["bins"]["channel"]:
                for freshness_bin in cfg["bins"]["freshness"]:
                    acc = estimate_accuracy(
                        question_type=question_type,
                        service_level=int(service_level),
                        channel_bin=channel_bin,
                        view_quality_bin=view_bin,
                        freshness_bin=freshness_bin,
                        risk_level=risk_level,
                        evaluator_cfg=cfg["evaluator"],
                    )
                    key = (question_type, int(service_level), channel_bin, view_bin, freshness_bin, risk_level)
                    grouped[key].append(_clip(acc * _deterministic_task_jitter(task, key, cfg["evaluator"])))
    rows: list[LUTRow] = []
    for key, values in sorted(grouped.items()):
        mu = mean(values)
        std = pstdev(values) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(max(1, len(values)))
        rows.append(LUTRow(*key, round(mu, 6), len(values), round(ci95, 6)))
    return rows


def write_lut_csv(rows: list[LUTRow], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LUT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(rows: list[LUTRow], tasks: list[dict[str, str]], json_path: Path, md_path: Path) -> None:
    ensure_parent(json_path)
    ensure_parent(md_path)
    service_levels = sorted({row.service_level for row in rows})
    payload = {
        "status": "v0_semantic_quality_lut_ready",
        "task_count": len(tasks),
        "lut_rows": len(rows),
        "service_levels": service_levels,
        "quality_constraint": "A_k >= epsilon_k",
        "deadline_constraint": "T_k <= tau_k",
        "critical_is_risk_level": True,
        "schema": LUT_FIELDNAMES,
        "min_sample_count": min((row.sample_count for row in rows), default=0),
        "max_sample_count": max((row.sample_count for row in rows), default=0),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# V0 Semantic Quality LUT Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Tasks: `{payload['task_count']}`",
        f"- LUT rows: `{payload['lut_rows']}`",
        f"- Service levels: `{service_levels}`",
        "- Quality constraint: `A_k >= epsilon_k`",
        "- Deadline constraint: `T_k <= tau_k`",
        "- `critical` is a risk level, not a service level.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
