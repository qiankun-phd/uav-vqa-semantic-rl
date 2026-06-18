#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.rl.v19_ppo import PPOServicePolicy, PPOTrainConfig, load_ppo_policy, save_ppo_model, train_ppo
from vqa_semcom.rl.v19_resource_env import V19LUTResourceEnv, V19StepRecord
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv
from vqa_semcom.snr import parse_snr_bins


BASELINE_POLICIES = (
    "always_cache",
    "always_light",
    "always_image",
    "greedy_min_sufficient_evidence",
    "no_cache_greedy",
    "no_semantic_tokens_greedy",
    "oracle_best_feasible_evidence",
)


@dataclass(frozen=True)
class EvalSummary:
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
    battery_violation_rate: float
    resource_violation_rate: float
    airspace_conflict_rate: float
    service_level_0_ratio: float
    service_level_1_ratio: float
    service_level_2_ratio: float
    service_level_3_ratio: float
    average_reward: float


def make_env(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tasks: list[dict[str, str]],
    lut: dict[tuple[str, int, str, str, str, str], Any],
    policy_name: str,
) -> V19LUTResourceEnv:
    snr_bins = parse_snr_bins(args.snr_bins) if args.snr_bins else None
    env_cfg = _cfg_with_scenario(cfg, args.scenario)
    return V19LUTResourceEnv(
        tasks=tasks,
        lut=lut,
        cfg=env_cfg,
        seed=args.seed,
        snr_bins_db=snr_bins,
        tasks_per_episode=args.tasks_per_episode,
        policy_name=policy_name,
    )


def choose_baseline_action(policy: str, env: V19LUTResourceEnv, obs: dict[str, Any]) -> dict[str, Any]:
    if policy == "always_cache":
        return env.candidate_action(0, obs)
    if policy == "always_light":
        return env.candidate_action(1, obs)
    if policy == "always_image":
        return env.candidate_action(2, obs)
    if policy == "greedy_min_sufficient_evidence":
        return env.candidate_action(_first_quality_level(env, obs, env.service_levels), obs)
    if policy == "no_cache_greedy":
        candidates = [level for level in env.service_levels if level != 0]
        return env.candidate_action(_first_quality_level(env, obs, candidates or env.service_levels), obs)
    if policy == "no_semantic_tokens_greedy":
        candidates = [level for level in env.service_levels if level != 1]
        return env.candidate_action(_first_quality_level(env, obs, candidates or env.service_levels), obs)
    if policy == "oracle_best_feasible_evidence":
        candidates = []
        for level in env.service_levels:
            metrics = env.candidate_metrics(level, obs)
            if metrics["success"] > 0.0:
                candidates.append((metrics["payload_kb"], level))
        if candidates:
            return env.candidate_action(min(candidates)[1], obs)
        best_level = max(env.service_levels, key=lambda level: env.candidate_metrics(level, obs)["accuracy"])
        return env.candidate_action(best_level, obs)
    raise ValueError(f"unknown policy: {policy}")


def evaluate_policy(
    env: V19LUTResourceEnv,
    policy: str,
    episodes: int,
    seed: int,
    action_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[V19StepRecord]:
    records: list[V19StepRecord] = []
    for episode in range(episodes):
        obs = env.reset(seed=seed + episode, options={"policy_name": policy})
        done = False
        while not done:
            obs, _reward, done, info = env.step(action_fn(obs))
            records.append(V19StepRecord(**info["record"]))
    return records


def summarize(records_by_policy: dict[str, list[V19StepRecord]], episodes: int) -> list[EvalSummary]:
    baseline_payload = _mean(records_by_policy.get("always_image", []), "payload_kb")
    out: list[EvalSummary] = []
    for policy, records in records_by_policy.items():
        tasks = len(records)
        payload = _mean(records, "payload_kb")
        reduction = 0.0 if baseline_payload <= 0.0 else (baseline_payload - payload) / baseline_payload
        out.append(
            EvalSummary(
                policy=policy,
                episodes=episodes,
                tasks=tasks,
                task_success_rate=round(_rate(records, "success"), 6),
                average_accuracy=round(_mean(records, "answer_accuracy_est"), 6),
                average_delay=round(_mean(records, "delay_s"), 6),
                average_energy=round(_mean(records, "energy_j"), 6),
                average_payload_kb=round(payload, 6),
                payload_reduction_vs_always_image=round(reduction, 6),
                quality_violation_rate=round(_rate(records, "quality_violation"), 6),
                deadline_violation_rate=round(_rate(records, "deadline_violation"), 6),
                battery_violation_rate=round(_rate(records, "battery_violation"), 6),
                resource_violation_rate=round(_rate(records, "resource_violation"), 6),
                airspace_conflict_rate=round(_rate(records, "airspace_conflict"), 6),
                service_level_0_ratio=round(_service_ratio(records, 0), 6),
                service_level_1_ratio=round(_service_ratio(records, 1), 6),
                service_level_2_ratio=round(_service_ratio(records, 2), 6),
                service_level_3_ratio=round(_service_ratio(records, 3), 6),
                average_reward=round(_mean(records, "reward"), 6),
            )
        )
    return out


def write_outputs(
    output_dir: Path,
    summaries: list[EvalSummary],
    records_by_policy: dict[str, list[V19StepRecord]],
    train_trace: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "v1_9_resource_alloc_results.csv"
    rollout_csv = output_dir / "v1_9_resource_alloc_rollout.csv"
    summary_md = output_dir / "v1_9_resource_alloc_summary.md"
    trace_csv = output_dir / "ppo_training_trace.csv"
    lambda_trace_csv = output_dir / "ppo_lambda_trace.csv"
    _write_dict_csv(results_csv, [asdict(row) for row in summaries])
    rollout_rows = [record.to_row() for records in records_by_policy.values() for record in records]
    _write_dict_csv(rollout_csv, rollout_rows)
    if train_trace:
        _write_dict_csv(trace_csv, train_trace)
        lambda_rows = []
        for row in train_trace:
            lambda_row = {"episode": row.get("episode", 0.0)}
            for key, value in row.items():
                if key.startswith("lambda_") or key.endswith("_cost") or key.endswith("_cost_normal") or key.endswith("_cost_critical"):
                    lambda_row[key] = value
            lambda_rows.append(lambda_row)
        _write_dict_csv(lambda_trace_csv, lambda_rows)
    _write_summary_md(summary_md, summaries, len(rollout_rows), bool(train_trace))
    print(f"wrote {results_csv}")
    print(f"wrote {rollout_csv}")
    print(f"wrote {summary_md}")
    if train_trace:
        print(f"wrote {trace_csv}")
        print(f"wrote {lambda_trace_csv}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v1_9_snr_lut.yaml"))
    parser.add_argument("--lut-csv", default=str(ROOT / "outputs" / "lut" / "v1_9_snr_semantic_quality_lut.csv"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--tasks-per-episode", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-bins", default=None, help="Optional comma-separated sensed SNR bins; defaults to measured LUT bins.")
    parser.add_argument("--scenario", default=None, help="Optional canonical multi-UAV scenario, e.g. conflict-heavy/interference-heavy/mobility-stress.")
    parser.add_argument("--policy", default="all", choices=["all", "ppo", *BASELINE_POLICIES])
    parser.add_argument("--train-ppo", action="store_true", help="Train and evaluate centralized hybrid PPO.")
    parser.add_argument("--load-ppo-model", default=None, help="Load a trained PPO checkpoint and evaluate policy=ppo without training.")
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--service-only-ppo", action="store_true", help="Disable continuous resource heads and train legacy service-level PPO.")
    parser.add_argument("--no-constrained-ppo", action="store_true", help="Disable Lagrangian quality/deadline constraint penalties.")
    parser.add_argument("--risk-aware-constraints", action="store_true", help="Use risk-aware CMDP dual variables for normal/critical tasks.")
    parser.add_argument("--semantic-reward-mode", default="env", choices=["env", "semantic_utility", "no_semantic_utility", "accuracy_only", "uncertainty_aware"])
    parser.add_argument("--proposed-semantic-rl", action="store_true", help="Enable the proposed semantic-utility-guided cognitive constrained hybrid controller.")
    parser.add_argument("--imitation-warm-start", action="store_true", help="Behavior clone demonstrations before PPO fine-tuning.")
    parser.add_argument("--demo-policy", default="oracle_best_feasible_evidence", choices=["greedy_min_sufficient_evidence", "oracle_best_feasible_evidence"])
    parser.add_argument("--demo-episodes", type=int, default=40)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-aux-weight", type=float, default=0.05)
    parser.add_argument("--no-safety-layer", action="store_true", help="Disable the service/GPU/battery/conflict safety projection layer.")
    parser.add_argument("--lambda-lr", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=20.0)
    parser.add_argument("--quality-cost-limit", type=float, default=0.0)
    parser.add_argument("--deadline-cost-limit", type=float, default=0.0)
    parser.add_argument("--power-min-w", type=float, default=0.05)
    parser.add_argument("--power-max-w", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "rl" / "v1_9_resource_alloc"))
    args = parser.parse_args()
    default_output_dir = str(ROOT / "outputs" / "rl" / "v1_9_resource_alloc")
    if args.smoke:
        args.episodes = min(args.episodes, 2)
        args.train_episodes = min(args.train_episodes, 8)
        args.demo_episodes = min(args.demo_episodes, 2)
        args.bc_epochs = min(args.bc_epochs, 2)
        args.train_ppo = True
        if args.output_dir == default_output_dir:
            args.output_dir = str(ROOT / "outputs" / "rl" / "v1_9_hybrid_tch_ppo_smoke")
    if args.proposed_semantic_rl:
        args.train_ppo = True
        args.risk_aware_constraints = True
        args.semantic_reward_mode = "semantic_utility"
        args.imitation_warm_start = True

    cfg = load_config(args.config)
    tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut = load_lut(resolve_path(args.lut_csv))
    tasks = filter_tasks_supported_by_lut(tasks, lut)
    if not tasks:
        raise RuntimeError("No tasks are supported by the V1.9 LUT.")

    selected = list(BASELINE_POLICIES) if args.policy == "all" else [args.policy]
    if args.train_ppo and "ppo" not in selected:
        selected.append("ppo")
    if args.load_ppo_model and "ppo" not in selected:
        selected.append("ppo")
    records_by_policy: dict[str, list[V19StepRecord]] = {}
    train_trace: list[dict[str, float]] = []
    ppo_policy: PPOServicePolicy | None = None

    if args.train_ppo or (args.policy == "ppo" and not args.load_ppo_model):
        train_env = make_env(args, cfg, tasks, lut, "ppo_train")
        ppo_cfg = PPOTrainConfig(
            train_episodes=args.train_episodes,
            hidden_size=args.hidden_size,
            hybrid_actions=not args.service_only_ppo,
            constrained=not args.no_constrained_ppo,
            risk_aware_constraints=args.risk_aware_constraints,
            safety_layer=not args.no_safety_layer,
            semantic_reward_mode=args.semantic_reward_mode,
            lambda_lr=args.lambda_lr,
            lambda_max=args.lambda_max,
            quality_cost_limit=args.quality_cost_limit,
            deadline_cost_limit=args.deadline_cost_limit,
            power_min_w=args.power_min_w,
            power_max_w=args.power_max_w,
            imitation_warm_start=args.imitation_warm_start,
            demo_policy=args.demo_policy,
            demo_episodes=args.demo_episodes,
            bc_epochs=args.bc_epochs,
            bc_aux_weight=args.bc_aux_weight,
        )
        model, train_trace = train_ppo(train_env, ppo_cfg, seed=args.seed)
        model_name = "ppo_service_policy.pt" if args.service_only_ppo else "ppo_hybrid_policy.pt"
        model_path = Path(args.output_dir) / model_name
        save_ppo_model(model_path, model, train_env, ppo_cfg)
        ppo_policy = PPOServicePolicy(train_env, model, ppo_cfg)
        print(f"wrote {model_path}")

    for policy in selected:
        eval_env = make_env(args, cfg, tasks, lut, policy)
        if policy == "ppo":
            if args.load_ppo_model:
                loaded_policy = load_ppo_policy(Path(args.load_ppo_model), eval_env, hidden_size=args.hidden_size)
                action_fn = lambda obs, policy=loaded_policy: policy.act(obs, deterministic=True)
            elif ppo_policy is not None:
                eval_policy = PPOServicePolicy(eval_env, ppo_policy.model, ppo_policy.cfg)
                action_fn = lambda obs, policy=eval_policy: policy.act(obs, deterministic=True)
            else:
                raise RuntimeError("Use --train-ppo before evaluating policy=ppo.")
        else:
            action_fn = lambda obs, name=policy, env=eval_env: choose_baseline_action(name, env, obs)
        records_by_policy[policy] = evaluate_policy(eval_env, policy, args.episodes, args.seed, action_fn)

    summaries = summarize(records_by_policy, args.episodes)
    write_outputs(Path(args.output_dir), summaries, records_by_policy, train_trace)
    for row in summaries:
        print(
            f"{row.policy}: success={row.task_success_rate:.3f} acc={row.average_accuracy:.3f} "
            f"delay={row.average_delay:.3f} energy={row.average_energy:.3f} payload_kb={row.average_payload_kb:.3f} "
            f"quality_vio={row.quality_violation_rate:.3f} deadline_vio={row.deadline_violation_rate:.3f} "
            f"battery_vio={row.battery_violation_rate:.3f} resource_vio={row.resource_violation_rate:.3f} "
            f"conflict={row.airspace_conflict_rate:.3f}"
        )
    return 0


def _cfg_with_scenario(cfg: dict[str, Any], scenario: str | None) -> dict[str, Any]:
    if not scenario:
        return cfg
    out = dict(cfg)
    env_cfg = dict(out.get("multi_uav_env", {}))
    env_cfg["scenario"] = scenario
    out["multi_uav_env"] = env_cfg
    return out


def _first_quality_level(env: V19LUTResourceEnv, obs: dict[str, Any], levels: list[int]) -> int:
    epsilon = float(obs["epsilon_k"])
    for level in levels:
        if env.candidate_metrics(level, obs)["accuracy"] >= epsilon:
            return level
    return levels[-1]


def _mean(records: list[V19StepRecord], field: str) -> float:
    if not records:
        return 0.0
    return sum(float(getattr(record, field)) for record in records) / len(records)


def _rate(records: list[V19StepRecord], field: str) -> float:
    if not records:
        return 0.0
    return sum(float(bool(getattr(record, field))) for record in records) / len(records)


def _service_ratio(records: list[V19StepRecord], level: int) -> float:
    if not records:
        return 0.0
    return sum(float(record.service_level == level) for record in records) / len(records)


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


def _write_summary_md(path: Path, summaries: list[EvalSummary], rollout_rows: int, trained_ppo: bool) -> None:
    lines = [
        "# V1.9 LUT Resource Allocation Summary",
        "",
        f"- rollout rows: {rollout_rows}",
        f"- trained PPO: {trained_ppo}",
        "- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv",
        "",
        "| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row.policy} | {row.task_success_rate:.3f} | {row.average_accuracy:.3f} | {row.average_delay:.3f} | "
            f"{row.average_energy:.3f} | {row.average_payload_kb:.3f} | {row.payload_reduction_vs_always_image:.3f} | "
            f"{row.quality_violation_rate:.3f} | {row.deadline_violation_rate:.3f} | {row.battery_violation_rate:.3f} | "
            f"{row.resource_violation_rate:.3f} | {row.airspace_conflict_rate:.3f} | {row.average_reward:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Service Level Selection Ratio",
            "",
            "| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row.policy} | {row.service_level_0_ratio:.3f} | {row.service_level_1_ratio:.3f} | "
            f"{row.service_level_2_ratio:.3f} | {row.service_level_3_ratio:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
