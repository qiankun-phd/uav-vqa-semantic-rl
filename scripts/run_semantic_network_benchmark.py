#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.sim.multi_uav_env import (
    available_formal_scenarios,
    load_multi_uav_env,
    scalability_presets,
    write_formal_scenario_specs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_9_snr_lut.yaml")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output-dir", default="outputs/env")
    parser.add_argument("--formal-scenarios", default=",".join(available_formal_scenarios()))
    parser.add_argument("--include-scalability", action="store_true")
    parser.add_argument("--scalability-mode", choices=["compact", "full"], default="compact")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_path = output_dir / "formal_scenario_specs.md"
    write_formal_scenario_specs(specs_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_csv = output_dir / f"scenario_smoke_{stamp}.csv"
    formal_scenarios = [item.strip() for item in args.formal_scenarios.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    run_index = 0
    for formal_scenario in formal_scenarios:
        for scalability in _scalability_cases(args.include_scalability, args.scalability_mode):
            env = load_multi_uav_env(args.config, seed=args.seed + run_index)
            options = {"formal_scenario": formal_scenario, **scalability}
            obs = env.reset(seed=args.seed + run_index, options=options)
            done = False
            for step in range(max(1, args.steps)):
                action = _benchmark_action(str(obs.get("scenario", "nominal")), obs, step)
                obs, reward, done, info = env.step(action)
                rows.append(
                    {
                        "run_index": run_index,
                        "step": step,
                        "formal_scenario": formal_scenario,
                        "scenario": info.get("scenario", obs.get("scenario", "")),
                        "benchmark_split": info.get("benchmark_split", obs.get("benchmark_split", "")),
                        "uav_count_profile": scalability.get("num_uavs", "default"),
                        "task_arrival_profile": scalability.get("task_arrival", "default"),
                        "edge_load_profile": scalability.get("edge_load", "default"),
                        "reward": reward,
                        **info,
                    }
                )
                if done:
                    break
            run_index += 1
    _write_csv(smoke_csv, rows)
    print(f"formal_specs={specs_path}")
    print(f"scenario_smoke_csv={smoke_csv}")
    print(f"rows={len(rows)}")
    return 0


def _scalability_cases(enabled: bool, mode: str) -> list[dict[str, str]]:
    if not enabled:
        return [{}]
    if mode == "compact":
        return [
            {"num_uavs": "M2", "task_arrival": "low", "edge_load": "light"},
            {"num_uavs": "M4", "task_arrival": "medium", "edge_load": "medium"},
            {"num_uavs": "M8", "task_arrival": "high", "edge_load": "heavy"},
        ]
    presets = scalability_presets()
    cases: list[dict[str, str]] = []
    for uav_count in presets["uav_count"]:
        for task_arrival in presets["task_arrival"]:
            for edge_load in presets["edge_load"]:
                cases.append({"num_uavs": uav_count, "task_arrival": task_arrival, "edge_load": edge_load})
    return cases


def _benchmark_action(scenario: str, obs: dict[str, Any], step: int) -> dict[str, Any]:
    queue = obs.get("task_queue", [])
    active_tasks = [item for item in queue if isinstance(item, dict)] if isinstance(queue, list) else []
    service_level = 0 if step % 3 == 0 else (1 if step % 3 == 1 else 2)
    action: dict[str, Any] = {
        "service_level": service_level,
        "bandwidth": 1_000_000.0,
        "power": 0.2 + 0.1 * (step % 3),
        "cpu_share": 0.5,
        "gpu_share": 0.5,
        "waypoint": None,
    }
    if scenario == "conflict-heavy":
        action.update(
            {
                "service_level": 1 if step % 2 == 0 else 2,
                "sensing_decision": "observe",
                "concurrent_actions": _concurrent_actions(active_tasks, service_level=1, limit=2),
            }
        )
    elif scenario == "interference-heavy":
        action.update(
            {
                "service_level": 2,
                "sensing_decision": "observe",
                "bandwidth": 700_000.0,
                "power": 0.8,
                "cpu_share": 0.65,
                "gpu_share": 0.65,
                "concurrent_actions": _concurrent_actions(
                    active_tasks,
                    service_level=2,
                    limit=1,
                    sensing_decision="upload",
                ),
            }
        )
    elif scenario == "cache-heavy":
        action.update(
            {
                "service_level": 0 if step % 2 == 0 else 1,
                "sensing_decision": "reuse_cache" if step % 2 == 0 else "observe",
                "bandwidth": 300_000.0,
                "power": 0.1,
                "cpu_share": 0.25,
                "gpu_share": 0.05,
            }
        )
    elif scenario == "mobility-stress":
        task = active_tasks[0] if active_tasks else {}
        area = task.get("area4d", {}) if isinstance(task, dict) else {}
        waypoint = None
        if isinstance(area, dict):
            waypoint = [float(area.get("center_x_m", 0.0)), float(area.get("center_y_m", 0.0))]
        action.update(
            {
                "service_level": 2 if step % 2 == 0 else 1,
                "sensing_decision": "revisit",
                "bandwidth": 900_000.0,
                "power": 0.7,
                "cpu_share": 0.55,
                "gpu_share": 0.55,
                "waypoint": waypoint,
            }
        )
    return action


def _concurrent_actions(
    active_tasks: list[dict[str, Any]],
    service_level: int,
    limit: int,
    sensing_decision: str = "observe",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in active_tasks[1 : limit + 1]:
        task_id = task.get("task_id")
        if not task_id:
            continue
        out.append(
            {
                "task_id": task_id,
                "service_level": service_level,
                "sensing_decision": sensing_decision,
                "bandwidth": 700_000.0,
                "power": 0.8,
                "cpu_share": 0.5,
                "gpu_share": 0.5,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
