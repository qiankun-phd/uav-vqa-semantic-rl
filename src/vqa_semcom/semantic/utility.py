from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from vqa_semcom.config import ensure_parent, load_config, resolve_path
from vqa_semcom.snr import snr_bin_label, snr_db_from_label, snr_bins_from_config


UTILITY_FIELDNAMES = [
    "question_type",
    "service_level",
    "channel_bin",
    "snr_bin",
    "view_quality_bin",
    "freshness_bin",
    "risk_level",
    "sample_count",
    "accuracy_mean",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "accuracy_lcb",
    "payload_kb",
    "uncertainty",
    "raw_accuracy_mean",
    "raw_accuracy_ci_low",
    "raw_accuracy_ci_high",
    "raw_payload_kb",
    "payload_bytes",
    "calibration_note",
]


@dataclass(frozen=True)
class SemanticUtilityEstimate:
    accuracy_mean: float
    accuracy_lcb: float
    payload_kb: float
    uncertainty: float
    sample_count: int


@dataclass(frozen=True)
class CacheQualityMetrics:
    accuracy_mean: float
    accuracy_lcb: float
    uncertainty: float
    sample_count: int
    payload_kb: float
    quality_gap: float
    recommended: bool
    eligible: bool


@dataclass(frozen=True)
class SemanticPathUtility:
    semantic_path: str
    service_level: int
    service_name: str
    accuracy_mean: float
    accuracy_lcb: float
    uncertainty: float
    sample_count: int
    payload_kb: float
    semantic_quality_gap: float
    semantic_efficiency: float
    cache_accuracy_mean: float
    cache_accuracy_lcb: float
    cache_uncertainty: float
    cache_quality_gap: float
    cache_recommended: bool
    cache_eligible: bool
    cache_update: bool


@dataclass(frozen=True)
class ServiceCandidateUtility:
    service_level: int
    service_name: str
    semantic_path: str
    accuracy_mean: float
    accuracy_lcb: float
    uncertainty: float
    payload_kb: float
    sample_count: int
    semantic_quality_gap: float
    semantic_efficiency: float
    estimated_delay_s: float
    semantic_feasible: bool
    deadline_feasible: bool
    estimated_delay_feasible: bool
    joint_feasible: bool
    cache_accuracy_mean: float
    cache_accuracy_lcb: float
    cache_uncertainty: float
    cache_quality_gap: float
    cache_recommended: bool
    cache_eligible: bool
    candidate_path_metrics: dict[str, Any]
    is_snr_sensitive: bool
    recommended_for_low_snr: bool
    recommended_for_critical: bool


@dataclass(frozen=True)
class SemanticUtilityCell:
    question_type: str
    service_level: int
    channel_bin: str
    snr_bin: str
    view_quality_bin: str
    freshness_bin: str
    risk_level: str
    sample_count: int
    accuracy_mean: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    accuracy_lcb: float
    payload_kb: float
    uncertainty: float
    raw_accuracy_mean: float
    raw_accuracy_ci_low: float
    raw_accuracy_ci_high: float
    raw_payload_kb: float
    payload_bytes: float
    calibration_note: str

    def estimate(self) -> SemanticUtilityEstimate:
        return SemanticUtilityEstimate(
            accuracy_mean=self.accuracy_mean,
            accuracy_lcb=self.accuracy_lcb,
            payload_kb=self.payload_kb,
            uncertainty=self.uncertainty,
            sample_count=self.sample_count,
        )


def _truthy(value: str | bool | int | float | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_value(value: str | int | float | None, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _int_value(value: str | int | float | None, default: int = 0) -> int:
    if value in {"", None}:
        return default
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + ((z * z) / (4.0 * total * total)))
    return _clip01(center - margin), _clip01(center + margin)


def _uncertainty(ci_low: float, ci_high: float, sample_count: int, min_samples: int) -> float:
    ci_half_width = max(0.0, (ci_high - ci_low) / 2.0)
    count_penalty = 1.0 / math.sqrt(max(1, sample_count))
    sparse_penalty = max(0.0, (min_samples - sample_count) / max(1, min_samples)) * 0.25
    return round(min(1.0, ci_half_width + count_penalty + sparse_penalty), 6)


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _raw_cells_from_predictions(predictions: Iterable[dict[str, str]], min_samples: int) -> list[SemanticUtilityCell]:
    grouped: dict[tuple[str, int, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        service_level = _int_value(row.get("service_level"), -1)
        if service_level < 0:
            continue
        snr_bin = row.get("snr_bin", "") or row.get("channel_bin", "")
        key = (
            row.get("question_type", ""),
            service_level,
            row.get("channel_bin", ""),
            snr_bin,
            row.get("view_quality_bin", ""),
            row.get("freshness_bin", ""),
            row.get("risk_level", ""),
        )
        if all(str(item) != "" for item in key):
            grouped[key].append(row)

    cells: list[SemanticUtilityCell] = []
    for key, rows in sorted(grouped.items()):
        values = [1.0 if _truthy(row.get("correct")) else 0.0 for row in rows]
        payloads = [max(0.0, _float_value(row.get("payload_bytes"))) for row in rows]
        n = len(values)
        successes = sum(values)
        raw_mean = _clip01(successes / max(1, n))
        ci_low, ci_high = wilson_interval(successes, n)
        payload_bytes = mean(payloads) if payloads else 0.0
        note = "raw"
        if n < min_samples:
            note = "raw;sparse_cell"
        cells.append(
            SemanticUtilityCell(
                question_type=key[0],
                service_level=key[1],
                channel_bin=key[2],
                snr_bin=key[3],
                view_quality_bin=key[4],
                freshness_bin=key[5],
                risk_level=key[6],
                sample_count=n,
                accuracy_mean=round(raw_mean, 6),
                accuracy_ci_low=round(ci_low, 6),
                accuracy_ci_high=round(ci_high, 6),
                accuracy_lcb=round(ci_low, 6),
                payload_kb=round(payload_bytes / 1024.0, 6),
                uncertainty=_uncertainty(ci_low, ci_high, n, min_samples),
                raw_accuracy_mean=round(raw_mean, 6),
                raw_accuracy_ci_low=round(ci_low, 6),
                raw_accuracy_ci_high=round(ci_high, 6),
                raw_payload_kb=round(payload_bytes / 1024.0, 6),
                payload_bytes=round(payload_bytes, 3),
                calibration_note=note,
            )
        )
    return cells


def _non_snr_key(cell: SemanticUtilityCell) -> tuple[str, int, str, str, str]:
    return (
        cell.question_type,
        cell.service_level,
        cell.view_quality_bin,
        cell.freshness_bin,
        cell.risk_level,
    )


def _replace_cell(
    cell: SemanticUtilityCell,
    accuracy_mean: float,
    accuracy_ci_low: float,
    accuracy_ci_high: float,
    payload_kb: float,
    note: str,
    min_samples: int,
) -> SemanticUtilityCell:
    accuracy_mean = _clip01(accuracy_mean)
    accuracy_ci_low = min(accuracy_mean, _clip01(accuracy_ci_low))
    accuracy_ci_high = max(accuracy_mean, _clip01(accuracy_ci_high))
    return SemanticUtilityCell(
        question_type=cell.question_type,
        service_level=cell.service_level,
        channel_bin=cell.channel_bin,
        snr_bin=cell.snr_bin,
        view_quality_bin=cell.view_quality_bin,
        freshness_bin=cell.freshness_bin,
        risk_level=cell.risk_level,
        sample_count=cell.sample_count,
        accuracy_mean=round(accuracy_mean, 6),
        accuracy_ci_low=round(accuracy_ci_low, 6),
        accuracy_ci_high=round(accuracy_ci_high, 6),
        accuracy_lcb=round(accuracy_ci_low, 6),
        payload_kb=round(max(0.0, payload_kb), 6),
        uncertainty=_uncertainty(accuracy_ci_low, accuracy_ci_high, cell.sample_count, min_samples),
        raw_accuracy_mean=cell.raw_accuracy_mean,
        raw_accuracy_ci_low=cell.raw_accuracy_ci_low,
        raw_accuracy_ci_high=cell.raw_accuracy_ci_high,
        raw_payload_kb=cell.raw_payload_kb,
        payload_bytes=round(max(0.0, payload_kb) * 1024.0, 3),
        calibration_note=note,
    )


def calibrate_snr_monotonicity(cells: list[SemanticUtilityCell], min_samples: int = 5) -> list[SemanticUtilityCell]:
    groups: dict[tuple[str, int, str, str, str], list[SemanticUtilityCell]] = defaultdict(list)
    for cell in cells:
        groups[_non_snr_key(cell)].append(cell)

    calibrated: list[SemanticUtilityCell] = []
    for group_cells in groups.values():
        if group_cells[0].service_level == 0:
            total_n = sum(cell.sample_count for cell in group_cells)
            if total_n > 0:
                successes = sum(cell.raw_accuracy_mean * cell.sample_count for cell in group_cells)
                cache_mean = _clip01(successes / total_n)
                ci_low, ci_high = wilson_interval(successes, total_n)
                payload_kb = sum(cell.raw_payload_kb * cell.sample_count for cell in group_cells) / total_n
            else:
                cache_mean, ci_low, ci_high, payload_kb = 0.0, 0.0, 1.0, 0.0
            for cell in group_cells:
                note_parts = ["cache_snr_invariant"]
                if cell.sample_count < min_samples:
                    note_parts.append("sparse_cell")
                calibrated.append(_replace_cell(cell, cache_mean, ci_low, ci_high, payload_kb, ";".join(note_parts), min_samples))
            continue

        ordered = sorted(group_cells, key=lambda cell: snr_db_from_label(cell.snr_bin))
        running_mean = 0.0
        running_lcb = 0.0
        for cell in ordered:
            adjusted = False
            mean_value = cell.raw_accuracy_mean
            ci_low = cell.raw_accuracy_ci_low
            ci_high = cell.raw_accuracy_ci_high
            if mean_value < running_mean:
                mean_value = running_mean
                adjusted = True
            if ci_low < running_lcb:
                ci_low = running_lcb
                adjusted = True
            ci_low = min(ci_low, mean_value)
            ci_high = max(ci_high, mean_value)
            running_mean = max(running_mean, mean_value)
            running_lcb = max(running_lcb, ci_low)
            note_parts = ["snr_monotonic_adjusted" if adjusted else "raw"]
            if cell.sample_count < min_samples:
                note_parts.append("sparse_cell")
            calibrated.append(_replace_cell(cell, mean_value, ci_low, ci_high, cell.raw_payload_kb, ";".join(note_parts), min_samples))

    return sorted(
        calibrated,
        key=lambda cell: (
            cell.question_type,
            cell.service_level,
            cell.view_quality_bin,
            cell.freshness_bin,
            cell.risk_level,
            snr_db_from_label(cell.snr_bin),
        ),
    )


def build_semantic_utility_from_predictions(
    predictions: Iterable[dict[str, str]],
    min_samples: int = 5,
) -> list[SemanticUtilityCell]:
    raw_cells = _raw_cells_from_predictions(predictions, min_samples=min_samples)
    return calibrate_snr_monotonicity(raw_cells, min_samples=min_samples)


def write_semantic_utility_csv(cells: list[SemanticUtilityCell], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UTILITY_FIELDNAMES)
        writer.writeheader()
        for cell in cells:
            writer.writerow(asdict(cell))


class SemanticUtilityModel:
    def __init__(self, cells: Iterable[SemanticUtilityCell]):
        self.cells = list(cells)
        self.table: dict[tuple[str, int, str, str, str, str], SemanticUtilityCell] = {}
        self.by_service_payload: dict[int, float] = defaultdict(float)
        payload_values: dict[int, list[float]] = defaultdict(list)
        for cell in self.cells:
            key = (
                cell.question_type,
                cell.service_level,
                cell.snr_bin,
                cell.view_quality_bin,
                cell.freshness_bin,
                cell.risk_level,
            )
            self.table[key] = cell
            payload_values[cell.service_level].append(cell.payload_kb)
        for service_level, values in payload_values.items():
            self.by_service_payload[service_level] = mean(values) if values else 0.0

    @classmethod
    def from_csv(cls, path: Path) -> "SemanticUtilityModel":
        return cls(read_semantic_utility_csv(path))

    def U_sem(
        self,
        task_type: str,
        service_level: int,
        snr_bin: str | int | float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
    ) -> SemanticUtilityEstimate:
        label = snr_bin_label(snr_bin) if isinstance(snr_bin, (int, float)) else str(snr_bin)
        key = (task_type, int(service_level), label, view_quality_bin, freshness_bin, risk_level)
        cell = self.table.get(key)
        if cell is None:
            cell = self._nearest_snr_cell(task_type, int(service_level), label, view_quality_bin, freshness_bin, risk_level)
        if cell is not None:
            return cell.estimate()
        return SemanticUtilityEstimate(
            accuracy_mean=0.0,
            accuracy_lcb=0.0,
            payload_kb=round(self.by_service_payload.get(int(service_level), 0.0), 6),
            uncertainty=1.0,
            sample_count=0,
        )

    def cache_quality_lcb(
        self,
        task_type: str,
        snr_bin: str | int | float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
    ) -> float:
        """Return cache-answer LCB while keeping cache SNR-invariant.

        The stable LUT key still contains ``snr_bin`` for compatibility, but
        calibrated cache cells are treated as SNR-invariant.  This helper uses
        service level 0 and ignores SNR if an equivalent cache cell exists.
        """

        return self._cache_estimate(task_type, snr_bin, view_quality_bin, freshness_bin, risk_level).accuracy_lcb

    def cache_quality_metrics(
        self,
        task_type: str,
        snr_bin: str | int | float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
        epsilon_k: float = 0.0,
    ) -> CacheQualityMetrics:
        estimate = self._cache_estimate(task_type, snr_bin, view_quality_bin, freshness_bin, risk_level)
        quality_gap = max(0.0, float(epsilon_k) - estimate.accuracy_lcb)
        recommended = self._cache_recommended(estimate, quality_gap, freshness_bin, risk_level)
        return CacheQualityMetrics(
            accuracy_mean=estimate.accuracy_mean,
            accuracy_lcb=estimate.accuracy_lcb,
            uncertainty=estimate.uncertainty,
            sample_count=estimate.sample_count,
            payload_kb=estimate.payload_kb,
            quality_gap=round(quality_gap, 6),
            recommended=recommended,
            eligible=recommended,
        )

    def path_utility(
        self,
        semantic_path: str,
        task_type: str,
        snr_bin: str | int | float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
        epsilon_k: float = 0.0,
        cache_update_service_level: int | str = 1,
    ) -> SemanticPathUtility:
        """Return utility for cache/token/image/cache_update paths.

        ``cache_update`` is a control path that serves the current task with a
        fresh-evidence service and refreshes cache state for future tasks.  By
        default it uses semantic tokens to preserve the lightweight path, but
        callers can override it with service level 2 when image refresh is
        needed.
        """

        path = _normalize_semantic_path(semantic_path)
        service_level = _service_level_for_path(path, cache_update_service_level)
        estimate = self.U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
        quality_gap = max(0.0, float(epsilon_k) - estimate.accuracy_lcb)
        cache = self.cache_quality_metrics(task_type, snr_bin, view_quality_bin, freshness_bin, risk_level, epsilon_k)
        return SemanticPathUtility(
            semantic_path=path,
            service_level=service_level,
            service_name=service_level_name(service_level),
            accuracy_mean=estimate.accuracy_mean,
            accuracy_lcb=estimate.accuracy_lcb,
            uncertainty=estimate.uncertainty,
            sample_count=estimate.sample_count,
            payload_kb=estimate.payload_kb,
            semantic_quality_gap=round(quality_gap, 6),
            semantic_efficiency=self._semantic_efficiency(estimate, quality_gap),
            cache_accuracy_mean=cache.accuracy_mean,
            cache_accuracy_lcb=cache.accuracy_lcb,
            cache_uncertainty=cache.uncertainty,
            cache_quality_gap=cache.quality_gap,
            cache_recommended=cache.recommended,
            cache_eligible=cache.eligible,
            cache_update=path == "cache_update",
        )

    def get_service_candidates(
        self,
        obs: dict[str, Any],
        service_levels: Iterable[int] | None = None,
    ) -> list[ServiceCandidateUtility]:
        """Return control-facing utility for each candidate evidence service.

        This is a thin RL/env-facing layer on top of the stable LUT key:

        ``question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level``.

        The method does not add a new LUT dimension.  It only evaluates the
        same task condition across candidate service levels and derives routing
        hints that are convenient for mobility-aware controllers.
        """

        task_type = str(obs.get("question_type", obs.get("task_type", "")))
        if not task_type:
            raise ValueError("obs must contain question_type or task_type")
        snr_value = obs.get("snr_bin", obs.get("sensed_snr_db", ""))
        if snr_value == "":
            raise ValueError("obs must contain snr_bin or sensed_snr_db")
        view_quality = str(obs.get("view_quality_bin", "medium"))
        freshness = str(obs.get("freshness_bin", "fresh"))
        risk = str(obs.get("risk_level", "normal"))
        epsilon = _float_value(obs.get("epsilon_k", obs.get("epsilon", 0.0)), 0.0)
        deadline_s = _float_value(obs.get("deadline_s", obs.get("tau_k", obs.get("deadline", 0.0))), 0.0)
        levels = list(service_levels) if service_levels is not None else sorted({cell.service_level for cell in self.cells})
        cache = self.cache_quality_metrics(task_type, snr_value, view_quality, freshness, risk, epsilon)

        candidates: list[ServiceCandidateUtility] = []
        for level in levels:
            estimate = self.U_sem(task_type, int(level), snr_value, view_quality, freshness, risk)
            gap = max(0.0, epsilon - estimate.accuracy_lcb)
            estimated_delay_s = self._estimated_delay_for_service(obs, int(level), estimate)
            semantic_feasible = gap <= 0.0
            deadline_feasible = deadline_s <= 0.0 or estimated_delay_s <= deadline_s
            semantic_efficiency = self._semantic_efficiency(estimate, gap)
            snr_sensitive = int(level) != 0
            candidates.append(
                ServiceCandidateUtility(
                    service_level=int(level),
                    service_name=service_level_name(int(level)),
                    semantic_path=semantic_path_name(int(level)),
                    accuracy_mean=estimate.accuracy_mean,
                    accuracy_lcb=estimate.accuracy_lcb,
                    uncertainty=estimate.uncertainty,
                    payload_kb=estimate.payload_kb,
                    sample_count=estimate.sample_count,
                    semantic_quality_gap=round(gap, 6),
                    semantic_efficiency=semantic_efficiency,
                    estimated_delay_s=round(estimated_delay_s, 6),
                    semantic_feasible=semantic_feasible,
                    deadline_feasible=deadline_feasible,
                    estimated_delay_feasible=deadline_feasible,
                    joint_feasible=semantic_feasible and deadline_feasible,
                    cache_accuracy_mean=cache.accuracy_mean,
                    cache_accuracy_lcb=cache.accuracy_lcb,
                    cache_uncertainty=cache.uncertainty,
                    cache_quality_gap=cache.quality_gap,
                    cache_recommended=cache.recommended,
                    cache_eligible=cache.eligible,
                    candidate_path_metrics={
                        "semantic_path": semantic_path_name(int(level)),
                        "accuracy_lcb": estimate.accuracy_lcb,
                        "semantic_quality_gap": round(gap, 6),
                        "payload_kb": estimate.payload_kb,
                        "cache_accuracy_lcb": cache.accuracy_lcb,
                        "cache_quality_gap": cache.quality_gap,
                        "cache_recommended": cache.recommended,
                    },
                    is_snr_sensitive=snr_sensitive,
                    recommended_for_low_snr=self._recommended_for_low_snr(int(level), estimate, gap, freshness),
                    recommended_for_critical=self._recommended_for_critical(int(level), estimate, gap, risk, freshness),
                )
            )
        return candidates

    def _cache_estimate(
        self,
        task_type: str,
        snr_bin: str | int | float,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
    ) -> SemanticUtilityEstimate:
        label = snr_bin_label(snr_bin) if isinstance(snr_bin, (int, float)) else str(snr_bin)
        cell = self.table.get((task_type, 0, label, view_quality_bin, freshness_bin, risk_level))
        if cell is None:
            matching = [
                item
                for item in self.cells
                if item.question_type == task_type
                and item.service_level == 0
                and item.view_quality_bin == view_quality_bin
                and item.freshness_bin == freshness_bin
                and item.risk_level == risk_level
            ]
            if matching:
                cell = sorted(matching, key=lambda item: snr_db_from_label(item.snr_bin))[0]
        if cell is not None:
            return cell.estimate()
        return self.U_sem(task_type, 0, label, view_quality_bin, freshness_bin, risk_level)

    @staticmethod
    def _cache_recommended(
        estimate: SemanticUtilityEstimate,
        semantic_quality_gap: float,
        freshness_bin: str,
        risk_level: str,
    ) -> bool:
        if estimate.sample_count <= 0 or semantic_quality_gap > 0.0:
            return False
        if freshness_bin == "expired":
            return False
        if risk_level == "critical":
            return freshness_bin == "fresh" and estimate.uncertainty <= 0.40
        return True

    def _nearest_snr_cell(
        self,
        task_type: str,
        service_level: int,
        snr_bin: str,
        view_quality_bin: str,
        freshness_bin: str,
        risk_level: str,
    ) -> SemanticUtilityCell | None:
        try:
            target_snr = snr_db_from_label(snr_bin)
        except ValueError:
            return None
        candidates = [
            cell
            for cell in self.cells
            if cell.question_type == task_type
            and cell.service_level == service_level
            and cell.view_quality_bin == view_quality_bin
            and cell.freshness_bin == freshness_bin
            and cell.risk_level == risk_level
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda cell: abs(snr_db_from_label(cell.snr_bin) - target_snr))

    @staticmethod
    def _semantic_efficiency(estimate: SemanticUtilityEstimate, semantic_quality_gap: float) -> float:
        usable_lcb = max(0.0, estimate.accuracy_lcb - semantic_quality_gap)
        confidence_discount = max(0.0, 1.0 - min(1.0, estimate.uncertainty))
        return round((usable_lcb * confidence_discount) / (1.0 + max(0.0, estimate.payload_kb)), 6)

    @staticmethod
    def _estimated_delay_for_service(
        obs: dict[str, Any],
        service_level: int,
        estimate: SemanticUtilityEstimate,
    ) -> float:
        for key in (
            "estimated_delay_by_service",
            "delay_by_service",
            "service_delay_s",
            "service_delay_by_level",
            "estimated_delay_s_by_service",
        ):
            value = obs.get(key)
            delay = _lookup_service_value(value, service_level)
            if delay is not None:
                return max(0.0, delay)
        scalar = obs.get("estimated_delay_s", obs.get("delay_s", obs.get("total_delay_s", None)))
        if scalar is not None and scalar != "":
            return max(0.0, _float_value(scalar))
        return _default_delay_estimate(service_level, estimate.payload_kb)

    @staticmethod
    def _recommended_for_low_snr(
        service_level: int,
        estimate: SemanticUtilityEstimate,
        semantic_quality_gap: float,
        freshness_bin: str,
    ) -> bool:
        if semantic_quality_gap > 0.0:
            return False
        if service_level == 1:
            return True
        if service_level == 0:
            return freshness_bin != "expired"
        return False

    @staticmethod
    def _recommended_for_critical(
        service_level: int,
        estimate: SemanticUtilityEstimate,
        semantic_quality_gap: float,
        risk_level: str,
        freshness_bin: str,
    ) -> bool:
        if risk_level != "critical" or semantic_quality_gap > 0.0:
            return False
        if estimate.sample_count <= 0 or estimate.uncertainty > 0.40:
            return False
        if service_level == 0 and freshness_bin != "fresh":
            return False
        return True


def service_level_name(service_level: int) -> str:
    names = {
        0: "cache_answer",
        1: "semantic_token",
        2: "image_evidence",
        3: "roi_crop_image",
    }
    return names.get(int(service_level), f"service_{int(service_level)}")


def semantic_path_name(service_level: int) -> str:
    paths = {
        0: "cache",
        1: "token",
        2: "image",
        3: "image",
    }
    return paths.get(int(service_level), f"service_{int(service_level)}")


def _normalize_semantic_path(path: str) -> str:
    value = str(path).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cache_answer": "cache",
        "semantic_cache": "cache",
        "semantic_token": "token",
        "compact_evidence": "token",
        "light": "token",
        "lightweight": "token",
        "image_evidence": "image",
        "raw_image": "image",
        "full_image": "image",
        "roi": "image",
        "roi_crop": "image",
        "cache_refresh": "cache_update",
        "update_cache": "cache_update",
    }
    normalized = aliases.get(value, value)
    if normalized not in {"cache", "token", "image", "cache_update"}:
        raise ValueError(f"unsupported semantic path: {path}")
    return normalized


def _service_level_for_path(path: str, cache_update_service_level: int | str = 1) -> int:
    if path == "cache":
        return 0
    if path == "token":
        return 1
    if path == "image":
        return 2
    if path == "cache_update":
        if isinstance(cache_update_service_level, str):
            return _service_level_for_path(_normalize_semantic_path(cache_update_service_level), 1)
        return int(cache_update_service_level)
    raise ValueError(f"unsupported semantic path: {path}")


def _lookup_service_value(value: Any, service_level: int) -> float | None:
    if isinstance(value, dict):
        for key in (service_level, str(service_level), f"s{service_level}", f"service_{service_level}"):
            if key in value:
                return _float_value(value[key])
    if isinstance(value, (list, tuple)) and 0 <= service_level < len(value):
        return _float_value(value[service_level])
    return None


def _default_delay_estimate(service_level: int, payload_kb: float) -> float:
    base = {0: 0.05, 1: 0.45, 2: 1.20, 3: 0.90}.get(int(service_level), 0.50)
    payload_scale = {0: 0.0, 1: 0.02, 2: 0.04, 3: 0.03}.get(int(service_level), 0.02)
    return round(base + max(0.0, payload_kb) * payload_scale, 6)


def read_semantic_utility_csv(path: Path) -> list[SemanticUtilityCell]:
    cells: list[SemanticUtilityCell] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            accuracy_ci_low = _float_value(row.get("accuracy_ci_low"))
            cells.append(
                SemanticUtilityCell(
                    question_type=row["question_type"],
                    service_level=_int_value(row["service_level"]),
                    channel_bin=row.get("channel_bin", ""),
                    snr_bin=row["snr_bin"],
                    view_quality_bin=row["view_quality_bin"],
                    freshness_bin=row["freshness_bin"],
                    risk_level=row["risk_level"],
                    sample_count=_int_value(row.get("sample_count")),
                    accuracy_mean=_float_value(row.get("accuracy_mean")),
                    accuracy_ci_low=accuracy_ci_low,
                    accuracy_ci_high=_float_value(row.get("accuracy_ci_high")),
                    accuracy_lcb=_float_value(row.get("accuracy_lcb"), accuracy_ci_low),
                    payload_kb=_float_value(row.get("payload_kb")),
                    uncertainty=_float_value(row.get("uncertainty"), 1.0),
                    raw_accuracy_mean=_float_value(row.get("raw_accuracy_mean"), _float_value(row.get("accuracy_mean"))),
                    raw_accuracy_ci_low=_float_value(row.get("raw_accuracy_ci_low"), accuracy_ci_low),
                    raw_accuracy_ci_high=_float_value(row.get("raw_accuracy_ci_high"), _float_value(row.get("accuracy_ci_high"))),
                    raw_payload_kb=_float_value(row.get("raw_payload_kb"), _float_value(row.get("payload_kb"))),
                    payload_bytes=_float_value(row.get("payload_bytes"), _float_value(row.get("payload_kb")) * 1024.0),
                    calibration_note=row.get("calibration_note", ""),
                )
            )
    return cells


def _mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def write_calibration_report(
    cells: list[SemanticUtilityCell],
    path: Path,
    predictions_path: Path,
    output_csv_path: Path,
    snr_bins: list[float],
) -> None:
    ensure_parent(path)
    sparse_cells = [cell for cell in cells if "sparse_cell" in cell.calibration_note]
    adjusted_cells = [cell for cell in cells if "snr_monotonic_adjusted" in cell.calibration_note]
    cache_cells = [cell for cell in cells if "cache_snr_invariant" in cell.calibration_note]
    total_samples = sum(cell.sample_count for cell in cells)
    service_levels = sorted({cell.service_level for cell in cells})
    snr_labels = [snr_bin_label(item) for item in snr_bins] or sorted({cell.snr_bin for cell in cells}, key=snr_db_from_label)

    lines = [
        "# VQA-grounded Task-conditioned Semantic Utility Model",
        "",
        "This report converts measured VQA correctness into a control-facing semantic utility interface.",
        "The utility is task-conditioned and VQA-grounded: it estimates answer correctness, payload, and uncertainty, not image quality.",
        "",
        "## Artifacts",
        "",
        f"- input predictions: `{predictions_path}`",
        f"- utility CSV: `{output_csv_path}`",
        "- API: `src/vqa_semcom/semantic/utility.py`",
        "",
        "## Interface",
        "",
        "```python",
        "U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)",
        "# -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count",
        "```",
        "",
        "The RL/environment side should prefer `accuracy_lcb` when it needs conservative QoS decisions and use `uncertainty` to down-weight sparse cells.",
        "",
        "## Calibration Summary",
        "",
        f"- utility cells: {len(cells)}",
        f"- total measured samples: {total_samples}",
        f"- SNR bins: {', '.join(snr_labels)}",
        f"- sparse cells: {len(sparse_cells)}",
        f"- SNR monotonic adjusted cells: {len(adjusted_cells)}",
        f"- cache SNR-invariant cells: {len(cache_cells)}",
        "- confidence interval: Wilson 95% binomial interval from answer correctness",
        "- uncertainty: CI half width plus finite-sample penalty, clipped to [0, 1]",
        "",
        "## Service-level Summary",
        "",
        "| service level | cells | mean accuracy | mean LCB | mean payload KB | mean uncertainty | samples |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for service_level in service_levels:
        level_cells = [cell for cell in cells if cell.service_level == service_level]
        lines.append(
            f"| {service_level} | {len(level_cells)} | "
            f"{_mean_or_zero([cell.accuracy_mean for cell in level_cells]):.3f} | "
            f"{_mean_or_zero([cell.accuracy_lcb for cell in level_cells]):.3f} | "
            f"{_mean_or_zero([cell.payload_kb for cell in level_cells]):.3f} | "
            f"{_mean_or_zero([cell.uncertainty for cell in level_cells]):.3f} | "
            f"{sum(cell.sample_count for cell in level_cells)} |"
        )

    lines.extend(["", "## Accuracy Mean by SNR", ""])
    lines.append("| service level | " + " | ".join(snr_labels) + " |")
    lines.append("|---:|" + "|".join(["---:" for _ in snr_labels]) + "|")
    for service_level in service_levels:
        row_values = []
        for label in snr_labels:
            level_snr_cells = [cell for cell in cells if cell.service_level == service_level and cell.snr_bin == label]
            row_values.append(f"{_mean_or_zero([cell.accuracy_mean for cell in level_snr_cells]):.3f}")
        lines.append(f"| {service_level} | " + " | ".join(row_values) + " |")

    lines.extend(
        [
            "",
            "## Notes for Paper Writing",
            "",
            "- The original VLM prediction CSV is unchanged; this file is a calibrated semantic utility layer built on top of it.",
            "- `s=0` cache is forced to be SNR-invariant because it does not transmit visual evidence.",
            "- `s=1` semantic tokens and `s=2` image evidence are sanity-checked so higher sensed SNR does not reduce the calibrated mean utility for the same task condition.",
            "- Sparse cells remain visible through `sample_count` and `uncertainty`; they are not silently treated as high-confidence measurements.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_utility_output_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg.get("paths", {}).get("semantic_utility_csv", "outputs/lut/v1_9_semantic_utility_with_ci.csv"))


def default_report_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg.get("paths", {}).get("semantic_utility_report_md", "outputs/reports/semantic_utility_calibration.md"))


def build_from_config(
    config_path: Path,
    predictions_csv: Path | None = None,
    output_csv: Path | None = None,
    report_md: Path | None = None,
    min_samples: int = 5,
) -> list[SemanticUtilityCell]:
    cfg = load_config(config_path)
    predictions_path = predictions_csv or resolve_path(cfg["paths"]["vlm_predictions_csv"])
    output_path = output_csv or default_utility_output_path(cfg)
    report_path = report_md or default_report_path(cfg)
    cells = build_semantic_utility_from_predictions(read_prediction_rows(predictions_path), min_samples=min_samples)
    write_semantic_utility_csv(cells, output_path)
    write_calibration_report(cells, report_path, predictions_path, output_path, snr_bins_from_config(cfg))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VQA-grounded semantic utility model with CI and SNR calibration.")
    parser.add_argument("--config", default="configs/v1_9_snr_lut.yaml")
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()

    build_from_config(
        resolve_path(args.config),
        predictions_csv=resolve_path(args.predictions_csv) if args.predictions_csv else None,
        output_csv=resolve_path(args.output_csv) if args.output_csv else None,
        report_md=resolve_path(args.report_md) if args.report_md else None,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
