#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.sim.multi_uav_env import SEMANTIC_PATHS, load_multi_uav_env


SCENARIOS = (
    "normal_patrol",
    "disaster_hotspot",
    "low_snr_soft",
    "low_snr_blockage",
    "edge_overload",
    "edge_overload_soft",
    "utm_conflict",
    "utm_conflict_soft",
)
MOBILITY_MODES = ("stay", "serve_task", "avoid_conflict", "reposition")
SERVICE_PATHS = ("cache", "token", "image", "cache_update")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose semantic-path soft scenarios and reject feasibility.")
    parser.add_argument("--config", default="configs/v1_9_snr_lut.yaml")
    parser.add_argument("--output-dir", default="outputs/env/semantic_path_soft_reject_diagnosis_20260624")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=48)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_records: list[dict[str, Any]] = []
    mobility_records: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            env = load_multi_uav_env(args.config, seed=seed, scenario=scenario)
            obs = env.reset(seed=seed, options={"scenario": scenario})
            for step in range(max(1, args.max_steps)):
                path_metrics = obs.get("candidate_path_metrics", {}) or {}
                mobility_metrics = obs.get("candidate_mobility_metrics", {}) or {}
                if not path_metrics:
                    break
                task_id = str(obs.get("task_id", ""))
                for path in SEMANTIC_PATHS:
                    metric = dict(path_metrics.get(path, {}))
                    path_records.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "step": step,
                            "task_id": task_id,
                            "semantic_path": path,
                            **_flatten_metric(metric),
                        }
                    )
                    for mode in MOBILITY_MODES:
                        mobility = dict(mobility_metrics.get(path, {}).get(mode, {}))
                        if not mobility:
                            continue
                        mobility_records.append(
                            {
                                "scenario": scenario,
                                "seed": seed,
                                "step": step,
                                "task_id": task_id,
                                "semantic_path": path,
                                "mobility_mode": mode,
                                **_flatten_metric(mobility),
                            }
                        )
                action = _oracle_step_action(env, obs, path_metrics)
                obs, _reward, done, info = env.step(action)
                if info:
                    path_records.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "step": step,
                            "task_id": task_id,
                            "semantic_path": "_oracle_action",
                            "oracle_path": str(action.get("semantic_path", "")),
                            "task_success": int(bool(info.get("success", False))),
                            "semantic_success": int(bool(info.get("semantic_success", False))),
                            "deadline_violation": int(bool(info.get("deadline_violation", False))),
                            "utm_conflict_violation": int(bool(info.get("utm_conflict_violation", False))),
                            "rejected": int(str(action.get("semantic_path", "")) == "reject"),
                            "reject_reason": str(info.get("reject_reason", "")),
                        }
                    )
                if done:
                    break

    summary_rows = _summarize_paths(path_records)
    mobility_rows = _summarize_mobility(mobility_records)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "mobility_summary.csv", mobility_rows)
    _write_csv(output_dir / "path_candidate_trace.csv", path_records)
    _write_csv(output_dir / "mobility_candidate_trace.csv", mobility_records)
    (output_dir / "report.md").write_text(_report(summary_rows, mobility_rows), encoding="utf-8")
    print(output_dir / "report.md")
    print(output_dir / "summary.csv")
    print(output_dir / "mobility_summary.csv")
    return 0


def _flatten_metric(metric: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metric.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, (int, float, str)):
            out[key] = value
    return out


def _oracle_step_action(env: Any, obs: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    task_id = obs.get("task_id")
    task = env._task_by_id(str(task_id)) if task_id else None
    feasible = [path for path in SERVICE_PATHS if bool(metrics.get(path, {}).get("joint_feasible", False))]
    if feasible:
        path = max(
            feasible,
            key=lambda item: (
                float(metrics[item].get("accuracy_lcb", 0.0)),
                float(metrics[item].get("deadline_slack_s", -1e9)),
                -float(metrics[item].get("energy_j", 0.0)),
            ),
        )
    elif bool(metrics.get("reject", {}).get("joint_feasible", False)):
        path = "reject"
    elif bool(metrics.get("defer", {}).get("joint_feasible", False)):
        path = "defer"
    else:
        path = min(
            SERVICE_PATHS,
            key=lambda item: (
                float(metrics.get(item, {}).get("required_deadline_reduction_s", 1e9)),
                float(metrics.get(item, {}).get("quality_gap", 1e9)),
            ),
        )
    action = env._path_action(path, task) if task is not None else {"semantic_path": path}
    action.setdefault("bandwidth", 1_000_000.0)
    action.setdefault("power", 1.0)
    action.setdefault("cpu_share", 1.0)
    action.setdefault("gpu_share", 1.0)
    return action


def _summarize_paths(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["scenario"]), str(row["semantic_path"]))].append(row)
    out: list[dict[str, Any]] = []
    for (scenario, path), rows in sorted(groups.items()):
        denom = max(1, len(rows))
        bottlenecks = Counter(str(row.get("bottleneck_type", "")) for row in rows)
        reject_reasons = Counter(str(row.get("reject_reason", "")) for row in rows if str(row.get("reject_reason", "")))
        oracle_paths = Counter(str(row.get("oracle_path", "")) for row in rows if str(row.get("oracle_path", "")))
        out.append(
            {
                "scenario": scenario,
                "semantic_path": path,
                "samples": len(rows),
                "joint_feasible_ratio": _ratio(rows, "joint_feasible"),
                "semantic_feasible_ratio": _ratio(rows, "semantic_feasible"),
                "deadline_feasible_ratio": _ratio(rows, "deadline_feasible"),
                "utm_feasible_ratio": _ratio(rows, "utm_feasible"),
                "avg_tx_delay_s": _avg(rows, "tx_delay_s"),
                "avg_queue_delay_s": _avg(rows, "queue_delay_s"),
                "avg_infer_delay_s": _avg(rows, "infer_delay_s"),
                "avg_load_delay_s": _avg(rows, "load_delay_s"),
                "avg_arrival_delay_s": _avg(rows, "arrival_delay_s"),
                "avg_required_rate_mbps": _avg(rows, "required_rate_mbps"),
                "avg_required_bandwidth_hz": _avg(rows, "required_bandwidth_hz"),
                "avg_required_deadline_reduction_s": _avg(rows, "required_deadline_reduction_s"),
                "avg_edge_queue_pressure": _avg(rows, "edge_queue_pressure"),
                "model_cache_hit_ratio": _ratio(rows, "model_cache_hit"),
                "reject_feasible_ratio": _ratio(rows, "reject_feasible"),
                "oracle_task_success_ratio": _ratio(rows, "task_success"),
                "oracle_reject_ratio": _ratio(rows, "rejected"),
                "bottleneck_distribution": ";".join(
                    f"{name}:{count / denom:.3f}" for name, count in sorted(bottlenecks.items())
                ),
                "reject_reason_distribution": ";".join(
                    f"{name}:{count / denom:.3f}" for name, count in sorted(reject_reasons.items())
                ),
                "oracle_path_distribution": ";".join(
                    f"{name}:{count / denom:.3f}" for name, count in sorted(oracle_paths.items())
                ),
            }
        )
    return out


def _summarize_mobility(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["scenario"]), str(row["semantic_path"]), str(row["mobility_mode"]))].append(row)
    out: list[dict[str, Any]] = []
    for (scenario, path, mode), rows in sorted(groups.items()):
        out.append(
            {
                "scenario": scenario,
                "semantic_path": path,
                "mobility_mode": mode,
                "samples": len(rows),
                "joint_feasible_ratio": _ratio(rows, "joint_feasible"),
                "semantic_feasible_ratio": _ratio(rows, "semantic_feasible"),
                "deadline_feasible_ratio": _ratio(rows, "deadline_feasible"),
                "utm_feasible_ratio": _ratio(rows, "utm_feasible"),
                "avg_arrival_delay_s": _avg(rows, "arrival_delay_s"),
                "avg_tx_delay_s": _avg(rows, "tx_delay_s"),
                "avg_total_delay_s": _avg(rows, "total_delay_s"),
                "avg_deadline_slack_s": _avg(rows, "deadline_slack_s"),
                "avg_utm_conflict_risk": _avg(rows, "utm_conflict_risk"),
            }
        )
    return out


def _ratio(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(bool(row.get(key, 0))) for row in rows) / max(1, len(rows))


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row.get(key, 0.0)) for row in rows if str(row.get(key, "")) not in {"", "inf", "nan"}]
    return mean(vals) if vals else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: list[dict[str, Any]], mobility: list[dict[str, Any]]) -> str:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_scenario[str(row["scenario"])].append(row)
    lines = [
        "# Semantic Path Soft Scenario and Reject Diagnosis",
        "",
        "Environment-only diagnosis. PPO training and Semantic Utility LUT are not modified.",
        "",
        "## Path Feasibility Summary",
        "",
        "| scenario | path | joint | semantic | deadline | UTM | reject feasible | oracle success | bottlenecks/reasons | avg tx | avg queue | avg infer | avg arrival | req. rate | req. bandwidth |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario']} | {row['semantic_path']} | {float(row['joint_feasible_ratio']):.3f} | "
            f"{float(row['semantic_feasible_ratio']):.3f} | {float(row['deadline_feasible_ratio']):.3f} | "
            f"{float(row['utm_feasible_ratio']):.3f} | {float(row['reject_feasible_ratio']):.3f} | "
            f"{float(row['oracle_task_success_ratio']):.3f} | "
            f"{row['bottleneck_distribution'] or row['reject_reason_distribution'] or row['oracle_path_distribution']} | "
            f"{float(row['avg_tx_delay_s']):.3f} | {float(row['avg_queue_delay_s']):.3f} | "
            f"{float(row['avg_infer_delay_s']):.3f} | {float(row['avg_arrival_delay_s']):.3f} | "
            f"{float(row['avg_required_rate_mbps']):.3f} | {float(row['avg_required_bandwidth_hz']):.1f} |"
        )
    lines.extend(["", "## Scenario Diagnosis", ""])
    lines.extend(_scenario_notes("edge_overload", by_scenario.get("edge_overload", []), mobility))
    lines.extend(_scenario_notes("edge_overload_soft", by_scenario.get("edge_overload_soft", []), mobility))
    lines.extend(_scenario_notes("utm_conflict", by_scenario.get("utm_conflict", []), mobility))
    lines.extend(_scenario_notes("utm_conflict_soft", by_scenario.get("utm_conflict_soft", []), mobility))
    lines.extend(["", "## Soft vs Hard Comparison", ""])
    lines.extend(_hard_soft_notes(by_scenario))
    lines.extend(["", "## Calibrated Scenario Suggestions", ""])
    lines.append(
        "- If `edge_overload` oracle feasible ratios remain near zero, keep it as a hard stress scenario and add a separate `edge_overload_soft` preset instead of weakening the hard preset."
    )
    lines.append(
        "- If `utm_conflict` remains semantic-infeasible after UTM-safe mobility, add safer non-overlapping task subsets or lower cache/task epsilon for a soft UTM scenario; do not relax hard UTM buffers silently."
    )
    return "\n".join(lines) + "\n"


def _hard_soft_notes(by_scenario: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    pairs = [("edge_overload", "edge_overload_soft"), ("utm_conflict", "utm_conflict_soft")]
    for hard, soft in pairs:
        hard_best = _best_joint(by_scenario.get(hard, []))
        soft_best = _best_joint(by_scenario.get(soft, []))
        hard_reject = _reject_ratio(by_scenario.get(hard, []))
        soft_reject = _reject_ratio(by_scenario.get(soft, []))
        lines.append(
            f"- `{soft}` vs `{hard}`: best service joint feasibility {soft_best:.3f} vs {hard_best:.3f}; "
            f"reject feasible ratio {soft_reject:.3f} vs {hard_reject:.3f}."
        )
    return lines


def _best_joint(rows: list[dict[str, Any]]) -> float:
    vals = [
        float(row.get("joint_feasible_ratio", 0.0))
        for row in rows
        if str(row.get("semantic_path", "")) in SERVICE_PATHS
    ]
    return max(vals) if vals else 0.0


def _reject_ratio(rows: list[dict[str, Any]]) -> float:
    reject = next((row for row in rows if str(row.get("semantic_path", "")) == "reject"), None)
    return float(reject.get("reject_feasible_ratio", 0.0)) if reject else 0.0


def _scenario_notes(scenario: str, rows: list[dict[str, Any]], mobility: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {scenario}", ""]
    if not rows:
        return lines + ["- No records.", ""]
    token = next((row for row in rows if row["semantic_path"] == "token"), None)
    cache = next((row for row in rows if row["semantic_path"] == "cache"), None)
    update = next((row for row in rows if row["semantic_path"] == "cache_update"), None)
    service_joint = max(float(row["joint_feasible_ratio"]) for row in rows if row["semantic_path"] in SERVICE_PATHS)
    if scenario == "edge_overload":
        if token:
            lines.append(
                f"- Token joint feasible ratio is {float(token['joint_feasible_ratio']):.3f}; deadline feasible ratio is {float(token['deadline_feasible_ratio']):.3f}."
            )
            lines.append(
                f"- Token average queue/infer/load delays are {float(token['avg_queue_delay_s']):.3f}/{float(token['avg_infer_delay_s']):.3f}/{float(token['avg_load_delay_s']):.3f} s; bottlenecks: {token['bottleneck_distribution']}."
            )
            lines.append(
                f"- When token misses deadline, average required rate/bandwidth estimates are {float(token['avg_required_rate_mbps']):.3f} Mbps and {float(token['avg_required_bandwidth_hz']):.1f} Hz."
            )
        if update:
            lines.append(
                f"- Cache-update uses token evidence but adds cache refresh semantics; its joint feasible ratio is {float(update['joint_feasible_ratio']):.3f}, so it should not be selected only from semantic quality."
            )
        if service_joint <= 0.02:
            lines.append("- Diagnosis: current edge-overload preset is nearly infeasible under the physical queue/model-load parameters.")
    elif scenario == "utm_conflict":
        if cache:
            lines.append(
                f"- Stay/cache is UTM-safe by construction, but cache semantic feasibility is {float(cache['semantic_feasible_ratio']):.3f}; low cache quality explains zero task success when cache dominates."
            )
        token_mob = [
            row
            for row in mobility
            if row["scenario"] == scenario and row["semantic_path"] == "token"
        ]
        serve = next((row for row in token_mob if row["mobility_mode"] == "serve_task"), None)
        avoid = next((row for row in token_mob if row["mobility_mode"] == "avoid_conflict"), None)
        if serve and avoid:
            lines.append(
                f"- Token avoid_conflict UTM risk {float(avoid['avg_utm_conflict_risk']):.3f} vs serve_task {float(serve['avg_utm_conflict_risk']):.3f}; avoid_conflict exposes a safer candidate if lower."
            )
            lines.append(
                f"- Token with avoid_conflict deadline feasible ratio is {float(avoid['deadline_feasible_ratio']):.3f}, semantic feasible ratio is {float(avoid['semantic_feasible_ratio']):.3f}, joint feasible ratio is {float(avoid['joint_feasible_ratio']):.3f}."
            )
        if service_joint <= 0.02:
            lines.append("- Diagnosis: UTM-safe actions exist, but current service candidates are semantically or deadline infeasible; this is not only a UTM violation issue.")
    return lines + [""]


if __name__ == "__main__":
    raise SystemExit(main())
