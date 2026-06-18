#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv, run_simulation, write_results


def _resolve_lut_path(cfg: dict, source: str):
    if source == "v0_rule":
        return resolve_path(cfg["paths"]["lut_csv"])
    if source in {"v1_mock", "v1_qwen", "v1_measured"}:
        return resolve_path(cfg["paths"]["vlm_lut_csv"])
    return resolve_path(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--lut-source", default="v0_rule", help="v0_rule, v1_mock, v1_qwen, v1_measured, or a CSV path")
    parser.add_argument("--filter-supported-tasks", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks_path = resolve_path(cfg["paths"]["tasks_csv"])
    lut_path = _resolve_lut_path(cfg, args.lut_source)
    if not tasks_path.exists() or not lut_path.exists():
        raise RuntimeError("Tasks/LUT not found. Run scripts/build_v0_lut.py first.")
    tasks = read_csv(tasks_path)
    lut = load_lut(lut_path)
    if args.filter_supported_tasks:
        tasks = filter_tasks_supported_by_lut(tasks, lut)
        if not tasks:
            raise RuntimeError(f"No tasks are supported by LUT source: {lut_path}")
    results = run_simulation(tasks, lut, cfg, episodes=args.episodes)
    write_results(results, resolve_path(cfg["paths"]["sim_results_csv"]), resolve_path(cfg["paths"]["sim_summary_md"]))
    for r in results:
        print(
            f"{r.policy}: success={r.task_success_rate:.3f} "
            f"acc={r.average_accuracy:.3f} delay={r.average_delay:.3f} energy={r.average_energy:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
