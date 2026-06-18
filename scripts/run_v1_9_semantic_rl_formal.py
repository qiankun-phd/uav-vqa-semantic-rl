#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_SCRIPT = ROOT / "scripts" / "run_v1_9_resource_alloc.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "rl" / "v1_9_formal_semantic_rl"
DEFAULT_SIM_RESULTS = ROOT / "outputs" / "sim" / "v1_9_snr_resource_results.csv"

METRICS = (
    "task_success_rate",
    "average_accuracy",
    "average_delay",
    "average_energy",
    "average_payload_kb",
    "quality_violation_rate",
    "deadline_violation_rate",
    "battery_violation_rate",
    "resource_violation_rate",
    "airspace_conflict_rate",
    "service_level_0_ratio",
    "service_level_1_ratio",
    "service_level_2_ratio",
    "service_level_3_ratio",
)

BASELINE_POLICIES = (
    "always_cache",
    "always_light",
    "always_image",
    "greedy_min_sufficient_evidence",
    "no_cache_greedy",
    "no_semantic_tokens_greedy",
    "oracle_best_feasible_evidence",
)

METHODS = (
    {
        "method": "service_only_ppo",
        "kind": "baseline_rl",
        "dir_prefix": "service_only_ppo",
        "args": ["--service-only-ppo", "--no-constrained-ppo", "--semantic-reward-mode", "env"],
    },
    {
        "method": "hybrid_ppo_no_lagrangian",
        "kind": "baseline_rl",
        "dir_prefix": "hybrid_ppo_no_lagrangian",
        "args": ["--no-constrained-ppo", "--semantic-reward-mode", "semantic_utility"],
    },
    {
        "method": "hybrid_ppo_lagrangian",
        "kind": "baseline_rl",
        "dir_prefix": "hybrid_ppo_lagrangian",
        "args": ["--semantic-reward-mode", "semantic_utility"],
    },
    {
        "method": "proposed_risk_aware_semantic_rl",
        "kind": "proposed",
        "dir_prefix": "proposed_risk_aware_semantic_rl",
        "args": [
            "--proposed-semantic-rl",
            "--demo-policy",
            "oracle_best_feasible_evidence",
            "--demo-episodes",
            "80",
            "--bc-epochs",
            "12",
            "--bc-aux-weight",
            "0.10",
            "--lambda-lr",
            "0.03",
        ],
    },
    {
        "method": "ablation_no_semantic_utility",
        "kind": "ablation",
        "dir_prefix": "ablation_no_semantic_utility",
        "args": [
            "--risk-aware-constraints",
            "--semantic-reward-mode",
            "no_semantic_utility",
            "--imitation-warm-start",
            "--demo-policy",
            "oracle_best_feasible_evidence",
            "--demo-episodes",
            "80",
            "--bc-epochs",
            "12",
            "--bc-aux-weight",
            "0.10",
        ],
    },
    {
        "method": "ablation_accuracy_only_utility",
        "kind": "ablation",
        "dir_prefix": "ablation_accuracy_only_utility",
        "args": [
            "--risk-aware-constraints",
            "--semantic-reward-mode",
            "accuracy_only",
            "--imitation-warm-start",
            "--demo-policy",
            "oracle_best_feasible_evidence",
            "--demo-episodes",
            "80",
            "--bc-epochs",
            "12",
            "--bc-aux-weight",
            "0.10",
        ],
    },
    {
        "method": "ablation_uncertainty_aware_utility",
        "kind": "ablation",
        "dir_prefix": "ablation_uncertainty_aware_utility",
        "args": [
            "--risk-aware-constraints",
            "--semantic-reward-mode",
            "uncertainty_aware",
            "--imitation-warm-start",
            "--demo-policy",
            "oracle_best_feasible_evidence",
            "--demo-episodes",
            "80",
            "--bc-epochs",
            "12",
            "--bc-aux-weight",
            "0.10",
        ],
    },
    {
        "method": "ablation_no_safety_layer",
        "kind": "ablation",
        "dir_prefix": "ablation_no_safety_layer",
        "args": [
            "--risk-aware-constraints",
            "--semantic-reward-mode",
            "semantic_utility",
            "--imitation-warm-start",
            "--demo-policy",
            "oracle_best_feasible_evidence",
            "--demo-episodes",
            "80",
            "--bc-epochs",
            "12",
            "--bc-aux-weight",
            "0.10",
            "--no-safety-layer",
        ],
    },
    {
        "method": "ablation_no_imitation_warm_start",
        "kind": "ablation",
        "dir_prefix": "ablation_no_imitation_warm_start",
        "args": ["--risk-aware-constraints", "--semantic-reward-mode", "semantic_utility"],
    },
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v1_9_snr_lut.yaml"))
    parser.add_argument("--lut-csv", default=str(ROOT / "outputs" / "lut" / "v1_9_snr_semantic_quality_lut.csv"))
    parser.add_argument("--sim-results", default=str(DEFAULT_SIM_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--tasks-per-episode", type=int, default=None)
    parser.add_argument("--snr-bins", default=None)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--generalization-scenarios", default="conflict-heavy,interference-heavy,mobility-stress")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    commands = _build_nominal_jobs(args, seeds, output_dir)
    _run_jobs(commands, max_workers=max(1, int(args.max_workers)), skip_existing=args.skip_existing)

    generalization_jobs = _build_generalization_jobs(args, seeds, output_dir)
    _run_jobs(generalization_jobs, max_workers=max(1, int(args.max_workers)), skip_existing=args.skip_existing)

    all_rows = _collect_nominal_rows(output_dir, seeds)
    _write_dict_csv(output_dir / "all_seed_results.csv", all_rows)
    summary_rows = _summarize_rows(all_rows)
    _write_dict_csv(output_dir / "formal_summary.csv", summary_rows)
    _write_dict_csv(output_dir / "ppo_training_trace.csv", _collect_trace_rows(output_dir, seeds, "ppo_training_trace.csv"))
    _write_dict_csv(output_dir / "ppo_lambda_trace.csv", _collect_trace_rows(output_dir, seeds, "ppo_lambda_trace.csv"))

    generalization_rows = _collect_generalization_rows(output_dir, seeds, args.generalization_scenarios)
    _write_dict_csv(output_dir / "generalization_summary.csv", _summarize_rows(generalization_rows, include_scenario=True))
    _write_dict_csv(output_dir / "generalization_all_seed_results.csv", generalization_rows)

    ablation_dir = output_dir / "ablation_tables"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    _write_ablation_tables(ablation_dir, summary_rows)

    sim_rows = _read_sim_results(Path(args.sim_results))
    _write_dict_csv(output_dir / "merged_with_sim_results.csv", _merge_with_sim(summary_rows, sim_rows))
    _write_report(output_dir / "merged_comparison.md", summary_rows, sim_rows, generalization_rows, args)
    _write_dict_csv(output_dir / "formal_commands.csv", [job["record"] for job in [*commands, *generalization_jobs]])

    print(f"wrote {output_dir / 'merged_comparison.md'}")
    print(f"wrote {output_dir / 'formal_summary.csv'}")
    print(f"wrote {output_dir / 'ablation_tables'}")
    return 0


def _build_nominal_jobs(args: argparse.Namespace, seeds: list[int], output_dir: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in seeds:
        baseline_dir = output_dir / f"baselines_seed{seed}"
        jobs.append(_job("heuristic_baselines", seed, baseline_dir, _base_cmd(args, baseline_dir, seed) + ["--policy", "all"]))
        for method in METHODS:
            run_dir = output_dir / f"{method['dir_prefix']}_seed{seed}"
            cmd = (
                _base_cmd(args, run_dir, seed)
                + ["--policy", "ppo", "--train-ppo", "--train-episodes", str(args.train_episodes)]
                + list(method["args"])
            )
            jobs.append(_job(str(method["method"]), seed, run_dir, cmd))
    return jobs


def _build_generalization_jobs(args: argparse.Namespace, seeds: list[int], output_dir: Path) -> list[dict[str, Any]]:
    scenarios = [item.strip() for item in str(args.generalization_scenarios).split(",") if item.strip()]
    jobs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            baseline_dir = output_dir / "generalization" / f"baselines_{scenario}_seed{seed}"
            jobs.append(
                _job(
                    f"generalization_baselines_{scenario}",
                    seed,
                    baseline_dir,
                    _base_cmd(args, baseline_dir, seed, scenario=scenario) + ["--policy", "all"],
                )
            )
            proposed_dir = output_dir / f"proposed_risk_aware_semantic_rl_seed{seed}"
            model_path = proposed_dir / "ppo_hybrid_policy.pt"
            eval_dir = output_dir / "generalization" / f"proposed_{scenario}_seed{seed}"
            jobs.append(
                _job(
                    f"generalization_proposed_{scenario}",
                    seed,
                    eval_dir,
                    _base_cmd(args, eval_dir, seed, scenario=scenario)
                    + ["--policy", "ppo", "--load-ppo-model", str(model_path)],
                    dependency=model_path,
                )
            )
    return jobs


def _base_cmd(args: argparse.Namespace, output_dir: Path, seed: int, scenario: str | None = None) -> list[str]:
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
    if scenario:
        cmd.extend(["--scenario", scenario])
    return cmd


def _job(method: str, seed: int, run_dir: Path, cmd: list[str], dependency: Path | None = None) -> dict[str, Any]:
    return {
        "method": method,
        "seed": seed,
        "run_dir": run_dir,
        "cmd": cmd,
        "dependency": dependency,
        "record": {"method": method, "seed": seed, "run_dir": str(run_dir), "command": " ".join(cmd)},
    }


def _run_jobs(jobs: list[dict[str, Any]], max_workers: int, skip_existing: bool) -> None:
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one_job, job, skip_existing) for job in jobs]
        for future in as_completed(futures):
            future.result()


def _run_one_job(job: dict[str, Any], skip_existing: bool) -> None:
    run_dir = Path(job["run_dir"])
    result_path = run_dir / "v1_9_resource_alloc_results.csv"
    if skip_existing and result_path.exists():
        print(f"[skip] {job['method']} seed={job['seed']}: {result_path}", flush=True)
        return
    dependency = job.get("dependency")
    if dependency is not None and not Path(dependency).exists():
        raise FileNotFoundError(f"dependency for {job['method']} is missing: {dependency}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    print(f"[run] {job['method']} seed={job['seed']} -> {run_dir}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(job["cmd"]) + "\n")
        log.flush()
        subprocess.run(job["cmd"], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def _collect_nominal_rows(output_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for row in _read_dict_csv(output_dir / f"baselines_seed{seed}" / "v1_9_resource_alloc_results.csv"):
            row = dict(row)
            row["method"] = row.get("policy", "")
            row["variant_kind"] = "heuristic"
            row["seed"] = seed
            rows.append(row)
        for method in METHODS:
            path = output_dir / f"{method['dir_prefix']}_seed{seed}" / "v1_9_resource_alloc_results.csv"
            ppo_rows = _read_dict_csv(path)
            if not ppo_rows:
                raise RuntimeError(f"missing PPO row in {path}")
            row = dict(ppo_rows[0])
            row["method"] = str(method["method"])
            row["variant_kind"] = str(method["kind"])
            row["seed"] = seed
            rows.append(row)
    return rows


def _collect_generalization_rows(output_dir: Path, seeds: list[int], scenarios_raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = [item.strip() for item in scenarios_raw.split(",") if item.strip()]
    for scenario in scenarios:
        for seed in seeds:
            for row in _read_dict_csv(output_dir / "generalization" / f"baselines_{scenario}_seed{seed}" / "v1_9_resource_alloc_results.csv"):
                row = dict(row)
                row["method"] = row.get("policy", "")
                row["scenario"] = scenario
                row["seed"] = seed
                rows.append(row)
            proposed = _read_dict_csv(output_dir / "generalization" / f"proposed_{scenario}_seed{seed}" / "v1_9_resource_alloc_results.csv")
            if proposed:
                row = dict(proposed[0])
                row["method"] = "proposed_risk_aware_semantic_rl"
                row["scenario"] = scenario
                row["seed"] = seed
                rows.append(row)
    return rows


def _summarize_rows(rows: list[dict[str, Any]], include_scenario: bool = False) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("scenario", "")) if include_scenario else "", str(row["method"]))].append(row)
    out_rows: list[dict[str, Any]] = []
    for (scenario, method), items in grouped.items():
        out: dict[str, Any] = {
            "scenario": scenario,
            "method": method,
            "seeds": ",".join(str(int(float(row["seed"]))) for row in items),
            "runs": len(items),
            "episodes_per_run": items[0].get("episodes", ""),
            "tasks_total": int(sum(float(row.get("tasks", 0.0)) for row in items)),
        }
        for metric in METRICS:
            values = [float(row.get(metric, 0.0)) for row in items]
            out[f"{metric}_mean"] = round(_mean(values), 6)
            out[f"{metric}_std"] = round(_std(values), 6)
        out_rows.append(out)
    return sorted(out_rows, key=lambda row: (str(row.get("scenario", "")), _method_order(str(row["method"]))))


def _collect_trace_rows(output_dir: Path, seeds: list[int], filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in METHODS:
            path = output_dir / f"{method['dir_prefix']}_seed{seed}" / filename
            if not path.exists():
                continue
            for row in _read_dict_csv(path):
                row = dict(row)
                row["method"] = str(method["method"])
                row["seed"] = seed
                rows.append(row)
    return rows


def _write_ablation_tables(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    by_method = {str(row["method"]): row for row in summary_rows}
    utility_methods = [
        "proposed_risk_aware_semantic_rl",
        "ablation_no_semantic_utility",
        "ablation_accuracy_only_utility",
        "ablation_uncertainty_aware_utility",
    ]
    safety_methods = [
        "proposed_risk_aware_semantic_rl",
        "ablation_no_safety_layer",
        "ablation_no_imitation_warm_start",
    ]
    _write_dict_csv(path / "semantic_utility_ablation.csv", [by_method[m] for m in utility_methods if m in by_method])
    _write_dict_csv(path / "safety_imitation_ablation.csv", [by_method[m] for m in safety_methods if m in by_method])


def _read_sim_results(path: Path) -> list[dict[str, Any]]:
    rows = _read_dict_csv(path)
    out = []
    for row in rows:
        converted = {"source": "sim_v1_9_snr_resource", "method": row.get("policy", "")}
        for metric in METRICS:
            if metric in row:
                converted[f"{metric}_mean"] = row[metric]
                converted[f"{metric}_std"] = ""
        out.append(converted)
    return out


def _merge_with_sim(summary_rows: list[dict[str, Any]], sim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [dict(row) for row in sim_rows]
    for row in summary_rows:
        converted = {"source": "semantic_rl_formal", "method": row["method"]}
        for metric in METRICS:
            converted[f"{metric}_mean"] = row.get(f"{metric}_mean", "")
            converted[f"{metric}_std"] = row.get(f"{metric}_std", "")
        merged.append(converted)
    return merged


def _write_report(path: Path, summary_rows: list[dict[str, Any]], sim_rows: list[dict[str, Any]], generalization_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# V1.9 Semantic-Utility-Guided Cognitive RL Comparison",
        "",
        "## CMDP Formulation",
        "",
        "- state: semantic task state plus UAV/network/edge/cache state.",
        "- action: service_level, bandwidth, power, cpu_share, gpu_share, and UAV assignment.",
        "- reward: semantic success utility minus delay, energy, payload, and conflict costs.",
        "- constraints: quality, deadline, battery, airspace conflict, and GPU memory feasibility.",
        "",
        f"- output dir: `{Path(args.output_dir)}`",
        f"- seeds: `{args.seeds}`",
        f"- eval episodes per seed: `{args.episodes}`",
        f"- train episodes per PPO seed: `{args.train_episodes}`",
        "",
        "## Main Results",
        "",
        "| method | success | accuracy | delay | energy | payload KB | quality vio | deadline vio | battery vio | GPU vio | conflict | s0 | s1 | s2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(_summary_table_row(row))
    lines.extend(["", "## Simulator Alignment", ""])
    lines.append("| sim policy | success | accuracy | delay | energy | payload KB | quality vio | deadline vio | s0 | s1 | s2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sim_rows:
        lines.append(
            f"| {row['method']} | {_fmt_raw(row, 'task_success_rate')} | {_fmt_raw(row, 'average_accuracy')} | "
            f"{_fmt_raw(row, 'average_delay')} | {_fmt_raw(row, 'average_energy')} | {_fmt_raw(row, 'average_payload_kb')} | "
            f"{_fmt_raw(row, 'quality_violation_rate')} | {_fmt_raw(row, 'deadline_violation_rate')} | "
            f"{_fmt_raw(row, 'service_level_0_ratio')} | {_fmt_raw(row, 'service_level_1_ratio')} | {_fmt_raw(row, 'service_level_2_ratio')} |"
        )
    if generalization_rows:
        lines.extend(["", "## Generalization Rows", ""])
        for row in _summarize_rows(generalization_rows, include_scenario=True):
            lines.append(f"- {row['scenario']} / {row['method']}: success={_fmt_pm(row, 'task_success_rate')}, energy={_fmt_pm(row, 'average_energy')}, payload={_fmt_pm(row, 'average_payload_kb')}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `all_seed_results.csv`",
            "- `formal_summary.csv`",
            "- `ppo_training_trace.csv`",
            "- `ppo_lambda_trace.csv`",
            "- `ablation_tables/semantic_utility_ablation.csv`",
            "- `ablation_tables/safety_imitation_ablation.csv`",
            "- `generalization_summary.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_table_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['method']} | {_fmt_pm(row, 'task_success_rate')} | {_fmt_pm(row, 'average_accuracy')} | "
        f"{_fmt_pm(row, 'average_delay')} | {_fmt_pm(row, 'average_energy')} | {_fmt_pm(row, 'average_payload_kb')} | "
        f"{_fmt_pm(row, 'quality_violation_rate')} | {_fmt_pm(row, 'deadline_violation_rate')} | "
        f"{_fmt_pm(row, 'battery_violation_rate')} | {_fmt_pm(row, 'resource_violation_rate')} | {_fmt_pm(row, 'airspace_conflict_rate')} | "
        f"{_fmt_pm(row, 'service_level_0_ratio')} | {_fmt_pm(row, 'service_level_1_ratio')} | {_fmt_pm(row, 'service_level_2_ratio')} |"
    )


def _fmt_pm(row: dict[str, Any], metric: str) -> str:
    return f"{float(row.get(f'{metric}_mean', 0.0)):.3f}+/-{float(row.get(f'{metric}_std', 0.0)):.3f}"


def _fmt_raw(row: dict[str, Any], metric: str) -> str:
    value = row.get(f"{metric}_mean", "")
    return "" if value == "" else f"{float(value):.3f}"


def _method_order(method: str) -> tuple[int, str]:
    order = [*BASELINE_POLICIES, *(str(item["method"]) for item in METHODS)]
    return (order.index(method) if method in order else len(order), method)


def _parse_seeds(raw: str) -> list[int]:
    out = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not out:
        raise ValueError("--seeds must include at least one integer")
    return out


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


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    raise SystemExit(main())
