#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "rl" / "v1_9_formal_hybrid_tch_ppo_20260618"
DEFAULT_SIM_RESULTS = ROOT / "outputs" / "sim" / "v1_9_snr_resource_results.csv"
RESOURCE_SCRIPT = ROOT / "scripts" / "run_v1_9_resource_alloc.py"

METRICS = (
    "task_success_rate",
    "average_accuracy",
    "average_delay",
    "average_energy",
    "average_payload_kb",
    "quality_violation_rate",
    "deadline_violation_rate",
    "service_level_0_ratio",
    "service_level_1_ratio",
    "service_level_2_ratio",
    "service_level_3_ratio",
)

VARIANTS = (
    {
        "method": "service_only_ppo",
        "dir_prefix": "service_only_ppo",
        "args": ["--service-only-ppo", "--no-constrained-ppo"],
    },
    {
        "method": "hybrid_ppo",
        "dir_prefix": "hybrid_ppo",
        "args": ["--no-constrained-ppo"],
    },
    {
        "method": "hybrid_tch_ppo",
        "dir_prefix": "hybrid_tch_ppo",
        "args": [],
    },
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v1_9_snr_lut.yaml"))
    parser.add_argument("--lut-csv", default=str(ROOT / "outputs" / "lut" / "v1_9_snr_semantic_quality_lut.csv"))
    parser.add_argument("--sim-results", default=str(DEFAULT_SIM_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--tasks-per-episode", type=int, default=None)
    parser.add_argument("--snr-bins", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command_rows: list[dict[str, Any]] = []
    for seed in seeds:
        baseline_dir = output_dir / f"baselines_seed{seed}"
        baseline_cmd = _base_cmd(args, baseline_dir, seed) + ["--policy", "all"]
        _run_if_needed(baseline_cmd, baseline_dir, "heuristic_baselines", seed, args.skip_existing, command_rows)

        for variant in VARIANTS:
            run_dir = output_dir / f"{variant['dir_prefix']}_seed{seed}"
            variant_cmd = (
                _base_cmd(args, run_dir, seed)
                + ["--policy", "ppo", "--train-ppo", "--train-episodes", str(args.train_episodes)]
                + list(variant["args"])
            )
            _run_if_needed(variant_cmd, run_dir, str(variant["method"]), seed, args.skip_existing, command_rows)

    all_rows = _collect_result_rows(output_dir, seeds)
    _write_dict_csv(output_dir / "all_seed_results.csv", all_rows)
    summary_rows = _summarize_rows(all_rows)
    _write_dict_csv(output_dir / "formal_comparison_summary.csv", summary_rows)

    lambda_rows = _collect_trace_rows(output_dir, seeds, "ppo_lambda_trace.csv")
    trace_rows = _collect_trace_rows(output_dir, seeds, "ppo_training_trace.csv")
    _write_dict_csv(output_dir / "all_lambda_trace.csv", lambda_rows)
    _write_dict_csv(output_dir / "all_training_trace.csv", trace_rows)
    _write_dict_csv(output_dir / "formal_commands.csv", command_rows)

    sim_rows = _read_sim_results(Path(args.sim_results))
    merged_rows = _merge_with_sim(summary_rows, sim_rows)
    _write_dict_csv(output_dir / "merged_with_sim_results.csv", merged_rows)
    _write_report(output_dir / "merged_comparison_report.md", summary_rows, sim_rows, args)

    print(f"wrote {output_dir / 'all_seed_results.csv'}")
    print(f"wrote {output_dir / 'formal_comparison_summary.csv'}")
    print(f"wrote {output_dir / 'merged_with_sim_results.csv'}")
    print(f"wrote {output_dir / 'merged_comparison_report.md'}")
    return 0


def _base_cmd(args: argparse.Namespace, output_dir: Path, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        str(RESOURCE_SCRIPT),
        "--config",
        str(args.config),
        "--lut-csv",
        str(args.lut_csv),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if args.tasks_per_episode is not None:
        cmd.extend(["--tasks-per-episode", str(args.tasks_per_episode)])
    if args.snr_bins:
        cmd.extend(["--snr-bins", str(args.snr_bins)])
    return cmd


def _run_if_needed(
    cmd: list[str],
    run_dir: Path,
    method: str,
    seed: int,
    skip_existing: bool,
    command_rows: list[dict[str, Any]],
) -> None:
    results_csv = run_dir / "v1_9_resource_alloc_results.csv"
    status = "ran"
    if skip_existing and results_csv.exists():
        status = "skipped_existing"
        print(f"[skip] {method} seed={seed}: {results_csv}")
    else:
        print(f"[run] {method} seed={seed}: {' '.join(cmd)}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    command_rows.append(
        {
            "method": method,
            "seed": seed,
            "status": status,
            "run_dir": str(run_dir),
            "command": " ".join(cmd),
        }
    )


def _collect_result_rows(output_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        baseline_path = output_dir / f"baselines_seed{seed}" / "v1_9_resource_alloc_results.csv"
        for row in _read_dict_csv(baseline_path):
            row = dict(row)
            row["method"] = row.get("policy", "")
            row["variant"] = "heuristic"
            row["seed"] = seed
            row["run_dir"] = str(baseline_path.parent)
            rows.append(row)

        for variant in VARIANTS:
            result_path = output_dir / f"{variant['dir_prefix']}_seed{seed}" / "v1_9_resource_alloc_results.csv"
            ppo_rows = _read_dict_csv(result_path)
            if not ppo_rows:
                raise RuntimeError(f"missing PPO result row in {result_path}")
            row = dict(ppo_rows[0])
            row["method"] = str(variant["method"])
            row["variant"] = str(variant["method"])
            row["seed"] = seed
            row["run_dir"] = str(result_path.parent)
            rows.append(row)
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    summary: list[dict[str, Any]] = []
    for method, items in grouped.items():
        out: dict[str, Any] = {
            "method": method,
            "seeds": ",".join(str(int(float(row["seed"]))) for row in items),
            "runs": len(items),
            "episodes_per_run": _first(items, "episodes"),
            "tasks_total": int(sum(float(row.get("tasks", 0.0)) for row in items)),
        }
        for metric in METRICS:
            values = [float(row.get(metric, 0.0)) for row in items]
            out[f"{metric}_mean"] = round(_mean(values), 6)
            out[f"{metric}_std"] = round(_std(values), 6)
        summary.append(out)
    return sorted(summary, key=lambda row: _method_order(str(row["method"])))


def _collect_trace_rows(output_dir: Path, seeds: list[int], filename: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in VARIANTS:
            path = output_dir / f"{variant['dir_prefix']}_seed{seed}" / filename
            if not path.exists():
                continue
            for row in _read_dict_csv(path):
                row = dict(row)
                row["method"] = str(variant["method"])
                row["seed"] = seed
                out.append(row)
    return out


def _read_sim_results(path: Path) -> list[dict[str, Any]]:
    rows = _read_dict_csv(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        converted = {"source": "sim_v1_9_snr_resource", "method": row.get("policy", "")}
        for metric in METRICS:
            if metric in row:
                converted[f"{metric}_mean"] = row[metric]
                converted[f"{metric}_std"] = ""
        out.append(converted)
    return out


def _merge_with_sim(summary_rows: list[dict[str, Any]], sim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in sim_rows:
        merged.append(dict(row))
    for row in summary_rows:
        converted = {"source": "rl_formal_hybrid_tch_ppo", "method": row["method"]}
        for metric in METRICS:
            converted[f"{metric}_mean"] = row.get(f"{metric}_mean", "")
            converted[f"{metric}_std"] = row.get(f"{metric}_std", "")
        merged.append(converted)
    return merged


def _write_report(path: Path, summary_rows: list[dict[str, Any]], sim_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# V1.9 Formal RL Resource Allocation Comparison",
        "",
        f"- output dir: `{Path(args.output_dir)}`",
        f"- seeds: `{args.seeds}`",
        f"- episodes per seed: `{args.episodes}`",
        f"- PPO train episodes per seed: `{args.train_episodes}`",
        "- LUT oracle: `outputs/lut/v1_9_snr_semantic_quality_lut.csv`",
        "- aligned simulator baseline: `outputs/sim/v1_9_snr_resource_results.csv`",
        "",
        "## RL Formal Summary",
        "",
        "| method | success | accuracy | delay | energy | payload KB | quality vio | deadline vio | s0 | s1 | s2 | s3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | "
            f"{_fmt_pm(row, 'task_success_rate')} | {_fmt_pm(row, 'average_accuracy')} | "
            f"{_fmt_pm(row, 'average_delay')} | {_fmt_pm(row, 'average_energy')} | "
            f"{_fmt_pm(row, 'average_payload_kb')} | {_fmt_pm(row, 'quality_violation_rate')} | "
            f"{_fmt_pm(row, 'deadline_violation_rate')} | {_fmt_pm(row, 'service_level_0_ratio')} | "
            f"{_fmt_pm(row, 'service_level_1_ratio')} | {_fmt_pm(row, 'service_level_2_ratio')} | "
            f"{_fmt_pm(row, 'service_level_3_ratio')} |"
        )
    lines.extend(
        [
            "",
            "## Simulator Baseline Alignment",
            "",
            "| sim policy | success | accuracy | delay | energy | payload KB | quality vio | deadline vio | s0 | s1 | s2 | s3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sim_rows:
        lines.append(
            f"| {row['method']} | "
            f"{_fmt_raw(row, 'task_success_rate')} | {_fmt_raw(row, 'average_accuracy')} | "
            f"{_fmt_raw(row, 'average_delay')} | {_fmt_raw(row, 'average_energy')} | "
            f"{_fmt_raw(row, 'average_payload_kb')} | {_fmt_raw(row, 'quality_violation_rate')} | "
            f"{_fmt_raw(row, 'deadline_violation_rate')} | {_fmt_raw(row, 'service_level_0_ratio')} | "
            f"{_fmt_raw(row, 'service_level_1_ratio')} | {_fmt_raw(row, 'service_level_2_ratio')} | "
            f"{_fmt_raw(row, 'service_level_3_ratio')} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `all_seed_results.csv`: per-seed baseline and PPO variant results.",
            "- `formal_comparison_summary.csv`: mean/std over seeds.",
            "- `all_training_trace.csv`: PPO training traces with method and seed.",
            "- `all_lambda_trace.csv`: Lagrangian dual traces with method and seed.",
            "- `merged_with_sim_results.csv`: simulator rows plus RL formal rows in one table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_pm(row: dict[str, Any], metric: str) -> str:
    mean = float(row.get(f"{metric}_mean", 0.0))
    std = float(row.get(f"{metric}_std", 0.0))
    return f"{mean:.3f}+/-{std:.3f}"


def _fmt_raw(row: dict[str, Any], metric: str) -> str:
    value = row.get(f"{metric}_mean", "")
    if value == "":
        return ""
    return f"{float(value):.3f}"


def _method_order(method: str) -> tuple[int, str]:
    order = [
        "always_cache",
        "always_light",
        "always_image",
        "greedy_min_sufficient_evidence",
        "no_cache_greedy",
        "no_semantic_tokens_greedy",
        "oracle_best_feasible_evidence",
        "service_only_ppo",
        "hybrid_ppo",
        "hybrid_tch_ppo",
    ]
    return (order.index(method) if method in order else len(order), method)


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def _read_dict_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _first(rows: list[dict[str, Any]], key: str) -> Any:
    return rows[0].get(key, "") if rows else ""


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    raise SystemExit(main())
