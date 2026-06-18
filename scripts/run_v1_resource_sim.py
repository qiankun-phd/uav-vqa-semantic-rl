#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.snr import parse_snr_bins
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv, run_simulation, write_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_qwen.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--lut-csv", default=None)
    parser.add_argument("--snr-bins", default=None, help="Optional comma-separated sensed SNR bins for SNR-LUT smoke runs.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.snr_bins:
        snr_bins = parse_snr_bins(args.snr_bins)
        cfg.setdefault("bins", {})["snr_db"] = snr_bins
        cfg.setdefault("simulation", {})["sensed_snr_db_values"] = snr_bins
    tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut_path = resolve_path(args.lut_csv) if args.lut_csv else resolve_path(cfg["paths"]["vlm_lut_csv"])
    lut = load_lut(lut_path)
    supported_tasks = filter_tasks_supported_by_lut(tasks, lut)
    if not supported_tasks:
        raise RuntimeError(f"No VQA tasks are supported by measured LUT: {lut_path}")
    results = run_simulation(supported_tasks, lut, cfg, episodes=args.episodes)
    write_results(results, resolve_path(cfg["paths"]["sim_results_csv"]), resolve_path(cfg["paths"]["sim_summary_md"]))
    print(f"lut_csv={lut_path}")
    print(f"supported_tasks={len(supported_tasks)} episodes={args.episodes}")
    for r in results:
        print(
            f"{r.policy}: success={r.task_success_rate:.3f} "
            f"acc={r.average_accuracy:.3f} delay={r.average_delay:.3f} "
            f"energy={r.average_energy:.3f} payload_kb={r.average_payload_kb:.3f} "
            f"payload_reduction={r.payload_reduction_vs_always_image:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
