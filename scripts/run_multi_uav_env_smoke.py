#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.sim.multi_uav_env import available_scenarios, load_multi_uav_env, write_env_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_9_snr_lut.yaml")
    parser.add_argument("--scenario", default="nominal", choices=available_scenarios())
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    env = load_multi_uav_env(args.config, seed=args.seed, scenario=args.scenario)
    obs = env.reset(seed=args.seed, options={"scenario": args.scenario})
    rows: list[dict[str, object]] = []
    done = False
    for step in range(max(1, args.steps)):
        action = _scenario_action(args.scenario, obs, step)
        obs, reward, done, info = env.step(action)
        rows.append({"step": step, "reward": reward, **info})
        if done:
            break

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_label = args.scenario.replace("-", "_")
        output_dir = Path("outputs") / "env" / f"env_smoke_{scenario_label}_{stamp}"
    csv_path = output_dir / "trace.csv"
    summary_path = output_dir / "summary.md"
    write_env_trace(rows, csv_path, summary_path)
    print(f"output_dir={output_dir}")
    print(f"trace_csv={csv_path}")
    print(f"summary_md={summary_path}")
    if rows:
        last = rows[-1]
        print(
            "last_step "
            f"success={last.get('success')} "
            f"acc={float(last.get('answer_accuracy_est', 0.0)):.3f} "
            f"delay={float(last.get('delay_s', 0.0)):.3f} "
            f"energy={float(last.get('energy_j', 0.0)):.3f} "
            f"snr_bin={last.get('snr_bin')}"
        )
    return 0


def _scenario_action(scenario: str, obs: dict[str, object], step: int) -> dict[str, object]:
    queue = obs.get("task_queue", [])
    active_tasks = [item for item in queue if isinstance(item, dict)] if isinstance(queue, list) else []
    service_level = 0 if step % 3 == 0 else (1 if step % 3 == 1 else 2)
    action: dict[str, object] = {
        "service_level": service_level,
        "bandwidth": 1_000_000.0,
        "power": 0.1 + 0.05 * (step % 3),
        "cpu_share": 0.5,
        "gpu_share": 0.5,
        "waypoint": None,
    }
    if scenario == "conflict-heavy":
        action.update(
            {
                "service_level": 1 if step % 2 == 0 else 2,
                "sensing_decision": "observe",
                "bandwidth": 800_000.0,
                "power": 0.4,
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
    active_tasks: list[dict[str, object]],
    service_level: int,
    limit: int,
    sensing_decision: str = "observe",
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
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


if __name__ == "__main__":
    raise SystemExit(main())
