from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vqa_semcom.config import ensure_parent
from vqa_semcom.snr import parse_snr_bins, snr_bins_from_config, snr_db_to_bin_label


@dataclass(frozen=True)
class LUTEntry:
    accuracy: float
    payload_bytes: float


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    episodes: int
    tasks: int
    task_success_rate: float
    average_accuracy: float
    average_delay: float
    average_energy: float
    average_payload_kb: float
    payload_reduction_vs_always_image: float
    quality_violation_rate: float
    deadline_violation_rate: float
    service_level_0_ratio: float
    service_level_1_ratio: float
    service_level_2_ratio: float
    service_level_3_ratio: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _default_payload_bytes(service_level: int) -> float:
    return {0: 0.0, 1: 2_048.0, 2: 300_000.0, 3: 80_000.0}.get(service_level, 0.0)


def load_lut(path: Path) -> dict[tuple[str, int, str, str, str, str], LUTEntry]:
    table: dict[tuple[str, int, str, str, str, str], LUTEntry] = {}
    for row in read_csv(path):
        level = int(row["service_level"])
        link_quality = row.get("snr_bin", "") or row["channel_bin"]
        key = (
            row["question_type"],
            level,
            link_quality,
            row["view_quality_bin"],
            row["freshness_bin"],
            row["risk_level"],
        )
        payload_raw = row.get("avg_payload_bytes", "")
        if payload_raw in {"", None}:
            payload = _default_payload_bytes(level)
        else:
            try:
                payload = max(0.0, float(payload_raw))
            except ValueError:
                payload = _default_payload_bytes(level)
        table[key] = LUTEntry(accuracy=float(row["expected_accuracy"]), payload_bytes=payload)
    return table


def _choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def _choice_float(rng: random.Random, values: list[float]) -> float:
    return values[rng.randrange(len(values))]


def _lookup_entry(
    lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
    task: dict[str, str],
    level: int,
    link_quality: str,
    freshness: str,
) -> LUTEntry:
    key = (
        task["question_type"],
        level,
        link_quality,
        task["view_quality_bin"],
        freshness,
        task["risk_level"],
    )
    if key in lut:
        return lut[key]
    return LUTEntry(accuracy=0.0, payload_bytes=_default_payload_bytes(level))


def _lookup_accuracy(
    lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
    task: dict[str, str],
    level: int,
    link_quality: str,
    freshness: str,
) -> float:
    return _lookup_entry(lut, task, level, link_quality, freshness).accuracy


def _greedy_level(
    candidate_levels: list[int],
    task: dict[str, str],
    link_quality: str,
    freshness: str,
    lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
) -> int:
    epsilon = float(task["epsilon_k"])
    for level in candidate_levels:
        if _lookup_accuracy(lut, task, level, link_quality, freshness) >= epsilon:
            return level
    return candidate_levels[-1]


def _choose_level(
    policy: str,
    task: dict[str, str],
    link_quality: str,
    freshness: str,
    lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
    service_level_order: list[int],
) -> int:
    if policy == "always_cache":
        return 0
    if policy == "always_light":
        return 1
    if policy == "always_image":
        return 2
    if policy == "always_roi":
        return 3
    if policy == "greedy_min_sufficient_evidence":
        return _greedy_level(service_level_order, task, link_quality, freshness, lut)
    if policy == "no_cache_greedy":
        candidate_levels = [level for level in service_level_order if level != 0]
        return _greedy_level(candidate_levels or service_level_order, task, link_quality, freshness, lut)
    if policy == "no_semantic_tokens_greedy":
        candidate_levels = [level for level in service_level_order if level != 1]
        return _greedy_level(candidate_levels or service_level_order, task, link_quality, freshness, lut)
    if policy == "no_roi_greedy":
        candidate_levels = [level for level in service_level_order if level != 3]
        return _greedy_level(candidate_levels or service_level_order, task, link_quality, freshness, lut)
    if policy == "oracle_best_feasible_evidence":
        epsilon = float(task["epsilon_k"])
        feasible = [level for level in service_level_order if _lookup_accuracy(lut, task, level, link_quality, freshness) >= epsilon]
        if feasible:
            return min(feasible, key=lambda level: _lookup_entry(lut, task, level, link_quality, freshness).payload_bytes)
        return max(service_level_order, key=lambda level: _lookup_accuracy(lut, task, level, link_quality, freshness))
    raise ValueError(f"unknown policy: {policy}")


def filter_tasks_supported_by_lut(
    tasks: list[dict[str, str]],
    lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
) -> list[dict[str, str]]:
    supported_keys = {(qtype, view, risk) for qtype, _level, _channel, view, _fresh, risk in lut}
    return [task for task in tasks if (task["question_type"], task["view_quality_bin"], task["risk_level"]) in supported_keys]


def _delay_for_level(
    level: int,
    payload_bytes: float,
    link_quality: str,
    delay_by_level: dict[int, float],
    channel_delay: dict[str, float],
    cfg: dict[str, Any],
) -> float:
    sim_cfg = cfg["simulation"]
    if sim_cfg.get("use_snr_rate_model", False):
        sensed_snr_db = float(link_quality.replace("dB", "").replace("db", ""))
        bandwidth_hz = float(sim_cfg.get("bandwidth_hz", 1_000_000.0))
        processing = {int(k): float(v) for k, v in sim_cfg.get("processing_delay_by_level", {}).items()}
        proc_delay = processing.get(level, delay_by_level.get(level, 0.0))
        snr_linear = 10.0 ** (sensed_snr_db / 10.0)
        rate_bps = max(1.0, bandwidth_hz * math.log2(1.0 + snr_linear))
        tx_delay = (8.0 * max(0.0, payload_bytes)) / rate_bps
        return proc_delay + tx_delay
    return delay_by_level.get(level, delay_by_level.get(2, 3.4)) * float(channel_delay[link_quality])


def run_simulation(
    tasks: list[dict[str, str]],
    lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
    cfg: dict[str, Any],
    episodes: int,
) -> list[PolicyResult]:
    rng = random.Random(cfg["simulation"]["seed"])
    policies = cfg["simulation"]["policies"]
    freshness_bins = cfg["bins"]["freshness"]
    snr_bins_db = snr_bins_from_config(cfg)
    use_snr = bool(snr_bins_db)
    sensed_snr_values = parse_snr_bins(cfg["simulation"].get("sensed_snr_db_values")) if use_snr else []
    if use_snr and not sensed_snr_values:
        sensed_snr_values = list(snr_bins_db)
    channel_bins = cfg["bins"].get("channel", ["bad", "medium", "good"])
    tasks_per_episode = int(cfg["simulation"]["tasks_per_episode"])
    delay_by_level = {int(k): float(v) for k, v in cfg["simulation"]["delay_by_level"].items()}
    energy_by_level = {int(k): float(v) for k, v in cfg["simulation"]["energy_by_level"].items()}
    channel_delay = cfg["simulation"].get("channel_delay_multiplier", {"bad": 1.45, "medium": 1.1, "good": 0.9})
    levels_in_lut = sorted({level for _qtype, level, _channel, _view, _fresh, _risk in lut})
    service_level_order = [int(level) for level in cfg["simulation"].get("service_level_order", levels_in_lut)]
    service_level_order = [level for level in service_level_order if level in levels_in_lut]
    if not service_level_order:
        raise ValueError("simulation needs at least one measured service level")
    aggregates: dict[str, list[dict[str, float]]] = defaultdict(list)
    if not tasks:
        raise ValueError("simulation needs at least one task")
    for _episode in range(episodes):
        episode_tasks = [tasks[rng.randrange(len(tasks))] for _ in range(tasks_per_episode)]
        episode_contexts: list[tuple[dict[str, str], str, str]] = []
        for task in episode_tasks:
            if use_snr:
                sensed_snr_db = _choice_float(rng, sensed_snr_values)
                link_quality = snr_db_to_bin_label(sensed_snr_db, snr_bins_db)
            else:
                link_quality = _choice(rng, channel_bins)
            freshness = _choice(rng, freshness_bins)
            episode_contexts.append((task, link_quality, freshness))
        for policy in policies:
            for task, link_quality, freshness in episode_contexts:
                level = _choose_level(policy, task, link_quality, freshness, lut, service_level_order)
                entry = _lookup_entry(lut, task, level, link_quality, freshness)
                acc = entry.accuracy
                delay = _delay_for_level(level, entry.payload_bytes, link_quality, delay_by_level, channel_delay, cfg)
                energy = energy_by_level.get(level, energy_by_level.get(2, 2.5))
                epsilon = float(task["epsilon_k"])
                tau = float(task["tau_k"])
                quality_ok = acc >= epsilon
                deadline_ok = delay <= tau
                success = quality_ok and deadline_ok
                aggregates[policy].append(
                    {
                        "success": float(success),
                        "accuracy": acc,
                        "delay": delay,
                        "energy": energy,
                        "payload_bytes": entry.payload_bytes,
                        "quality_violation": float(not quality_ok),
                        "deadline_violation": float(not deadline_ok),
                        "level_0": float(level == 0),
                        "level_1": float(level == 1),
                        "level_2": float(level == 2),
                        "level_3": float(level == 3),
                    }
                )
    raw_results: dict[str, dict[str, float]] = {}
    for policy in policies:
        rows = aggregates[policy]
        denom = max(1, len(rows))
        raw_results[policy] = {
            "episodes": float(episodes),
            "tasks": float(len(rows)),
            "task_success_rate": sum(r["success"] for r in rows) / denom,
            "average_accuracy": sum(r["accuracy"] for r in rows) / denom,
            "average_delay": sum(r["delay"] for r in rows) / denom,
            "average_energy": sum(r["energy"] for r in rows) / denom,
            "average_payload_kb": (sum(r["payload_bytes"] for r in rows) / denom) / 1024.0,
            "quality_violation_rate": sum(r["quality_violation"] for r in rows) / denom,
            "deadline_violation_rate": sum(r["deadline_violation"] for r in rows) / denom,
            "service_level_0_ratio": sum(r["level_0"] for r in rows) / denom,
            "service_level_1_ratio": sum(r["level_1"] for r in rows) / denom,
            "service_level_2_ratio": sum(r["level_2"] for r in rows) / denom,
            "service_level_3_ratio": sum(r["level_3"] for r in rows) / denom,
        }
    baseline_payload = raw_results.get("always_image", {}).get("average_payload_kb", 0.0)
    results: list[PolicyResult] = []
    for policy in policies:
        row = raw_results[policy]
        payload_reduction = 0.0 if baseline_payload <= 0 else (baseline_payload - row["average_payload_kb"]) / baseline_payload
        results.append(
            PolicyResult(
                policy=policy,
                episodes=episodes,
                tasks=int(row["tasks"]),
                task_success_rate=round(row["task_success_rate"], 6),
                average_accuracy=round(row["average_accuracy"], 6),
                average_delay=round(row["average_delay"], 6),
                average_energy=round(row["average_energy"], 6),
                average_payload_kb=round(row["average_payload_kb"], 6),
                payload_reduction_vs_always_image=round(payload_reduction, 6),
                quality_violation_rate=round(row["quality_violation_rate"], 6),
                deadline_violation_rate=round(row["deadline_violation_rate"], 6),
                service_level_0_ratio=round(row["service_level_0_ratio"], 6),
                service_level_1_ratio=round(row["service_level_1_ratio"], 6),
                service_level_2_ratio=round(row["service_level_2_ratio"], 6),
                service_level_3_ratio=round(row["service_level_3_ratio"], 6),
            )
        )
    return results


def write_results(results: list[PolicyResult], csv_path: Path, md_path: Path) -> None:
    ensure_parent(csv_path)
    ensure_parent(md_path)
    fieldnames = list(PolicyResult.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    lines = ["# V1 Qwen Resource Simulation Summary", ""]
    lines.append("| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.policy} | {r.task_success_rate:.3f} | {r.average_accuracy:.3f} | "
            f"{r.average_delay:.3f} | {r.average_energy:.3f} | {r.average_payload_kb:.3f} | "
            f"{r.payload_reduction_vs_always_image:.3f} | {r.quality_violation_rate:.3f} | {r.deadline_violation_rate:.3f} |"
        )
    lines.extend(["", "## Service Level Selection Ratio", ""])
    lines.append("| policy | cache s=0 | light s=1 | image s=2 | roi s=3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.policy} | {r.service_level_0_ratio:.3f} | "
            f"{r.service_level_1_ratio:.3f} | {r.service_level_2_ratio:.3f} | {r.service_level_3_ratio:.3f} |"
        )
    lines.extend(["", "## Approximate 95% CI for Rate Metrics", ""])
    lines.append("| policy | success CI | quality violation CI | deadline violation CI | tasks |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        success_ci = _normal_ci(r.task_success_rate, r.tasks)
        quality_ci = _normal_ci(r.quality_violation_rate, r.tasks)
        deadline_ci = _normal_ci(r.deadline_violation_rate, r.tasks)
        lines.append(
            f"| {r.policy} | [{success_ci[0]:.3f}, {success_ci[1]:.3f}] | "
            f"[{quality_ci[0]:.3f}, {quality_ci[1]:.3f}] | "
            f"[{deadline_ci[0]:.3f}, {deadline_ci[1]:.3f}] | {r.tasks} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normal_ci(rate: float, n: int) -> tuple[float, float]:
    n = max(1, n)
    half_width = 1.96 * math.sqrt(max(0.0, rate * (1.0 - rate)) / n)
    return max(0.0, rate - half_width), min(1.0, rate + half_width)
