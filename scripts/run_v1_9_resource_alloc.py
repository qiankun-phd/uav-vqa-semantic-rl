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


SCENARIO_BENCHMARK_SCENARIOS = (
    "nominal_patrol",
    "disaster_hotspot",
    "low_snr_blockage",
    "edge_overload",
    "utm_conflict",
)

SCENARIO_BENCHMARK_POLICIES = (
    "always_cache",
    "always_semantic_token",
    "always_image",
    "semantic_greedy",
    "lyapunov_greedy",
    "ppo",
)

BASELINE_POLICIES = (
    "always_cache",
    "always_light",
    "always_semantic_token",
    "always_image",
    "greedy_min_sufficient_evidence",
    "semantic_greedy",
    "no_cache_greedy",
    "no_semantic_tokens_greedy",
    "semantic_lcb_greedy",
    "lyapunov_greedy",
    "oracle_best_feasible_evidence",
)


@dataclass(frozen=True)
class EvalSummary:
    policy: str
    scenario: str
    episodes: int
    tasks: int
    semantic_success_rate: float
    task_success_rate: float
    average_accuracy: float
    average_accuracy_mean: float
    average_uncertainty: float
    average_delay: float
    average_energy: float
    average_payload_kb: float
    average_semantic_payload_kb: float
    average_semantic_quality_gap: float
    average_q_quality: float
    average_q_deadline: float
    average_q_energy: float
    average_q_risk: float
    average_q_utm: float
    payload_reduction_vs_always_image: float
    quality_violation_rate: float
    deadline_violation_rate: float
    battery_violation_rate: float
    resource_violation_rate: float
    airspace_conflict_rate: float
    utm_constraint_violation_rate: float
    utm_conflict_violation_rate: float
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
        service_levels=_service_levels_for_args(args, cfg),
        formal_scenario=args.formal_scenario,
        policy_name=policy_name,
    )


def choose_baseline_action(policy: str, env: V19LUTResourceEnv, obs: dict[str, Any]) -> dict[str, Any]:
    if policy == "always_cache":
        return _with_action_defaults(env, env.candidate_action(0, obs))
    if policy in {"always_light", "always_semantic_token"}:
        return _with_action_defaults(env, _candidate_for_level(env, 1, obs))
    if policy == "always_image":
        return _with_action_defaults(env, _candidate_for_level(env, 2, obs))
    if policy in {"greedy_min_sufficient_evidence", "semantic_greedy"}:
        return _with_action_defaults(env, env.candidate_action(_first_quality_level(env, obs, env.service_levels), obs))
    if policy == "no_cache_greedy":
        candidates = [level for level in env.service_levels if level != 0]
        return _with_action_defaults(env, env.candidate_action(_first_quality_level(env, obs, candidates or env.service_levels), obs))
    if policy == "no_semantic_tokens_greedy":
        candidates = [level for level in env.service_levels if level != 1]
        return _with_action_defaults(env, env.candidate_action(_first_quality_level(env, obs, candidates or env.service_levels), obs))
    if policy == "semantic_lcb_greedy":
        return _with_action_defaults(env, env.candidate_action(_semantic_lcb_greedy_level(env, obs), obs))
    if policy == "lyapunov_greedy":
        return _with_action_defaults(env, _lyapunov_greedy_action(env, obs))
    if policy == "oracle_best_feasible_evidence":
        candidates = []
        for level in env.service_levels:
            metrics = env.candidate_metrics(level, obs)
            if metrics["success"] > 0.0:
                candidates.append((metrics["payload_kb"], level))
        if candidates:
            return _with_action_defaults(env, env.candidate_action(min(candidates)[1], obs))
        best_level = max(env.service_levels, key=lambda level: env.candidate_metrics(level, obs)["accuracy"])
        return _with_action_defaults(env, env.candidate_action(best_level, obs))
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


def summarize(records_by_policy: dict[str, list[V19StepRecord]], episodes: int, scenario: str = "") -> list[EvalSummary]:
    baseline_payload = _mean(records_by_policy.get("always_image", []), "payload_kb")
    out: list[EvalSummary] = []
    for policy, records in records_by_policy.items():
        tasks = len(records)
        payload = _mean(records, "payload_kb")
        reduction = 0.0 if baseline_payload <= 0.0 else (baseline_payload - payload) / baseline_payload
        out.append(
            EvalSummary(
                policy=policy,
                scenario=scenario,
                episodes=episodes,
                tasks=tasks,
                semantic_success_rate=round(_rate(records, "semantic_success"), 6),
                task_success_rate=round(_rate(records, "success"), 6),
                average_accuracy=round(_mean(records, "answer_accuracy_est"), 6),
                average_accuracy_mean=round(_mean(records, "semantic_accuracy_mean"), 6),
                average_uncertainty=round(_mean(records, "semantic_uncertainty"), 6),
                average_delay=round(_mean(records, "delay_s"), 6),
                average_energy=round(_mean(records, "energy_j"), 6),
                average_payload_kb=round(payload, 6),
                average_semantic_payload_kb=round(_mean(records, "semantic_payload_kb"), 6),
                average_semantic_quality_gap=round(_mean(records, "semantic_quality_gap"), 6),
                average_q_quality=round(_mean(records, "q_quality"), 6),
                average_q_deadline=round(_mean(records, "q_deadline"), 6),
                average_q_energy=round(_mean(records, "q_energy"), 6),
                average_q_risk=round(_mean(records, "q_risk"), 6),
                average_q_utm=round(_mean(records, "q_utm"), 6),
                payload_reduction_vs_always_image=round(reduction, 6),
                quality_violation_rate=round(_rate(records, "quality_violation"), 6),
                deadline_violation_rate=round(_rate(records, "deadline_violation"), 6),
                battery_violation_rate=round(_rate(records, "battery_violation"), 6),
                resource_violation_rate=round(_rate(records, "resource_violation"), 6),
                airspace_conflict_rate=round(_rate(records, "airspace_conflict"), 6),
                utm_constraint_violation_rate=round(_rate(records, "utm_constraint_violation"), 6),
                utm_conflict_violation_rate=round(_rate(records, "utm_conflict_violation"), 6),
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
    parser.add_argument("--formal-scenario", default=None, help="Optional formal semantic-network scenario, e.g. train_mixed_random/test_utm_dss_outage.")
    parser.add_argument("--scenario-benchmark", action="store_true", help="Run the five scenario-aware semantic communication benchmark smokes.")
    parser.add_argument("--benchmark-scenarios", default=",".join(SCENARIO_BENCHMARK_SCENARIOS), help="Comma-separated scenario benchmark list.")
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
    parser.add_argument("--entropy-start", type=float, default=0.05)
    parser.add_argument("--entropy-end", type=float, default=0.005)
    parser.add_argument("--entropy-decay-episodes", type=int, default=120)
    parser.add_argument("--service-prior-weight", type=float, default=0.25)
    parser.add_argument("--service-prior-decay-episodes", type=int, default=200)
    parser.add_argument("--no-semantic-projection", action="store_true", help="Disable LCB/deadline semantic projection inside the safety layer.")
    parser.add_argument("--semantic-lyapunov-control", action="store_true", help="Enable semantic LCB drift-plus-penalty reward with virtual Lyapunov queues.")
    parser.add_argument("--no-semantic-lcb", "--no-lcb", dest="no_semantic_lcb", action="store_true", help="Use raw/mean accuracy instead of conservative semantic accuracy LCB.")
    parser.add_argument("--no-lyapunov-queues", action="store_true", help="Disable virtual queue penalties while keeping other semantic PPO settings.")
    parser.add_argument("--no-resource-projection", "--no-projection", dest="no_resource_projection", action="store_true", help="Disable service-dependent resource floors/projection.")
    parser.add_argument("--disable-semantic-token", action="store_true", help="Disable service level 1 semantic-token evidence for ablations.")
    parser.add_argument("--queue-quality-weight", type=float, default=1.5)
    parser.add_argument("--queue-deadline-weight", type=float, default=0.8)
    parser.add_argument("--queue-energy-weight", type=float, default=0.25)
    parser.add_argument("--queue-risk-weight", type=float, default=1.0)
    parser.add_argument("--queue-utm-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-cost-weight", type=float, default=0.35)
    parser.add_argument("--energy-budget-j", type=float, default=500.0)
    parser.add_argument("--utm-conflict-cost-weight", type=float, default=1.5)
    parser.add_argument("--dss-delay-cost-weight", type=float, default=0.25)
    parser.add_argument("--off-nominal-cost-weight", type=float, default=1.0)
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
    if args.scenario_benchmark and args.output_dir == default_output_dir:
        args.output_dir = str(ROOT / "outputs" / "rl" / "semantic_scenario_benchmark")
    if args.smoke:
        args.episodes = min(args.episodes, 2)
        args.train_episodes = min(args.train_episodes, 8)
        args.demo_episodes = min(args.demo_episodes, 2)
        args.bc_epochs = min(args.bc_epochs, 2)
        args.train_ppo = True
        args.semantic_lyapunov_control = True
        args.risk_aware_constraints = True
        if args.semantic_reward_mode == "env":
            args.semantic_reward_mode = "semantic_utility"
        if args.output_dir == default_output_dir:
            args.output_dir = str(ROOT / "outputs" / "rl" / "v1_9_hybrid_tch_ppo_smoke")
    if args.proposed_semantic_rl:
        args.train_ppo = True
        args.risk_aware_constraints = True
        args.semantic_reward_mode = "semantic_utility"
        args.semantic_lyapunov_control = True
        args.imitation_warm_start = True
        args.entropy_start = max(float(args.entropy_start), 0.06)
        args.service_prior_weight = max(float(args.service_prior_weight), 0.30)
        args.bc_aux_weight = max(float(args.bc_aux_weight), 0.10)
        args.demo_episodes = max(int(args.demo_episodes), min(20, int(args.train_episodes)))

    cfg = load_config(args.config)
    tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut = load_lut(resolve_path(args.lut_csv))
    tasks = filter_tasks_supported_by_lut(tasks, lut)
    if not tasks:
        raise RuntimeError("No tasks are supported by the V1.9 LUT.")

    if args.scenario_benchmark:
        _run_scenario_benchmark(args, cfg, tasks, lut)
        return 0

    summaries = run_experiment(args, cfg, tasks, lut)
    for row in summaries:
        _print_summary_row(row)
    return 0


def run_experiment(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tasks: list[dict[str, str]],
    lut: dict[tuple[str, int, str, str, str, str], Any],
) -> list[EvalSummary]:
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
            entropy_weight_start=args.entropy_start,
            entropy_weight_end=args.entropy_end,
            entropy_decay_episodes=args.entropy_decay_episodes,
            service_prior_weight=args.service_prior_weight,
            service_prior_decay_episodes=args.service_prior_decay_episodes,
            semantic_projection=not args.no_semantic_projection,
            resource_projection=not args.no_resource_projection,
            use_semantic_lcb=not args.no_semantic_lcb,
            lyapunov_reward=args.semantic_lyapunov_control,
            use_lyapunov_queues=not args.no_lyapunov_queues,
            queue_quality_weight=args.queue_quality_weight,
            queue_deadline_weight=args.queue_deadline_weight,
            queue_energy_weight=args.queue_energy_weight,
            queue_risk_weight=args.queue_risk_weight,
            queue_utm_weight=args.queue_utm_weight,
            uncertainty_cost_weight=args.uncertainty_cost_weight,
            energy_budget_j=args.energy_budget_j,
            utm_conflict_cost_weight=args.utm_conflict_cost_weight,
            dss_delay_cost_weight=args.dss_delay_cost_weight,
            off_nominal_cost_weight=args.off_nominal_cost_weight,
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

    summaries = summarize(records_by_policy, args.episodes, scenario=str(args.scenario or args.formal_scenario or ""))
    write_outputs(Path(args.output_dir), summaries, records_by_policy, train_trace)
    return summaries


def _run_scenario_benchmark(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tasks: list[dict[str, str]],
    lut: dict[tuple[str, int, str, str, str, str], Any],
) -> None:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    scenarios = [item.strip() for item in str(args.benchmark_scenarios).split(",") if item.strip()]
    for scenario in scenarios:
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.scenario_benchmark = False
        scenario_args.scenario = scenario
        scenario_args.formal_scenario = None
        scenario_args.output_dir = str(root / scenario)
        scenario_args.policy = "all"
        scenario_args.train_ppo = True
        scenario_args.semantic_lyapunov_control = True
        scenario_args.risk_aware_constraints = True
        if scenario_args.semantic_reward_mode == "env":
            scenario_args.semantic_reward_mode = "semantic_utility"
        summaries = run_experiment(scenario_args, cfg, tasks, lut)
        by_policy = {row.policy: row for row in summaries}
        for policy in SCENARIO_BENCHMARK_POLICIES:
            if policy in by_policy:
                row = by_policy[policy]
            elif policy == "semantic_greedy" and "greedy_min_sufficient_evidence" in by_policy:
                row = by_policy["greedy_min_sufficient_evidence"]
            elif policy == "always_semantic_token" and "always_light" in by_policy:
                row = by_policy["always_light"]
            else:
                continue
            all_rows.append(asdict(row) | {"scenario": scenario, "benchmark_policy": policy})
        for row in summaries:
            _print_summary_row(row)
    _write_dict_csv(root / "scenario_comparison_summary.csv", all_rows)
    _write_scenario_benchmark_report(root / "scenario_comparison_report.md", all_rows, args)
    print(f"wrote {root / 'scenario_comparison_summary.csv'}")
    print(f"wrote {root / 'scenario_comparison_report.md'}")


def _print_summary_row(row: EvalSummary) -> None:
    print(
        f"{row.policy}: semantic_success={row.semantic_success_rate:.3f} success={row.task_success_rate:.3f} "
        f"acc_lcb={row.average_accuracy:.3f} acc_mean={row.average_accuracy_mean:.3f} uncertainty={row.average_uncertainty:.3f} "
        f"gap={row.average_semantic_quality_gap:.3f} q_quality={row.average_q_quality:.3f} q_deadline={row.average_q_deadline:.3f} "
        f"q_energy={row.average_q_energy:.3f} q_utm={row.average_q_utm:.3f} delay={row.average_delay:.3f} "
        f"energy={row.average_energy:.3f} payload_kb={row.average_payload_kb:.3f} deadline_vio={row.deadline_violation_rate:.3f} "
        f"utm_conflict={row.utm_conflict_violation_rate:.3f}"
    )


def _cfg_with_scenario(cfg: dict[str, Any], scenario: str | None) -> dict[str, Any]:
    if not scenario:
        return cfg
    out = dict(cfg)
    env_cfg = dict(out.get("multi_uav_env", {}))
    env_cfg["scenario"] = scenario
    out["multi_uav_env"] = env_cfg
    return out


def _service_levels_for_args(args: argparse.Namespace, cfg: dict[str, Any]) -> list[int] | None:
    if not bool(getattr(args, "disable_semantic_token", False)):
        return None
    bins = cfg.get("bins", {}) if isinstance(cfg, dict) else {}
    raw_levels = bins.get("service_levels", [0, 1, 2])
    levels = [int(level) for level in raw_levels if int(level) != 1]
    return levels or [0, 2]


def _candidate_for_level(env: V19LUTResourceEnv, level: int, obs: dict[str, Any]) -> dict[str, Any]:
    if int(level) in env.service_levels:
        return env.candidate_action(int(level), obs)
    fallback = min(env.service_levels, key=lambda item: abs(int(item) - int(level)))
    return env.candidate_action(fallback, obs)


def _with_action_defaults(env: V19LUTResourceEnv, action: dict[str, Any]) -> dict[str, Any]:
    return env.parse_action(action)


def _first_quality_level(env: V19LUTResourceEnv, obs: dict[str, Any], levels: list[int]) -> int:
    epsilon = float(obs["epsilon_k"])
    for level in levels:
        if env.candidate_metrics(level, obs)["accuracy"] >= epsilon:
            return level
    return levels[-1]


def _semantic_lcb_greedy_level(env: V19LUTResourceEnv, obs: dict[str, Any]) -> int:
    epsilon = float(obs["epsilon_k"])
    feasible: list[tuple[float, int]] = []
    for level in env.service_levels:
        action = env.candidate_action(level, obs)
        info = env.evaluate_action(action, obs)
        accuracy = float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0)))
        if accuracy >= epsilon and not _hard_violation(info):
            cost = (
                float(info.get("payload_kb", 0.0))
                + 0.3 * float(info.get("energy_j", 0.0))
                + 50.0 * float(info.get("delay_s", 0.0))
                + 20.0 * float(info.get("semantic_uncertainty", 0.0))
            )
            feasible.append((cost, level))
    if feasible:
        return min(feasible)[1]
    return max(env.service_levels, key=lambda level: env.candidate_metrics(level, obs)["accuracy"])


def _lyapunov_greedy_action(env: V19LUTResourceEnv, obs: dict[str, Any]) -> dict[str, Any]:
    queues = obs.get("lyapunov_queues", {}) if isinstance(obs.get("lyapunov_queues", {}), dict) else {}
    best_action = env.candidate_action(env.service_levels[0], obs)
    best_score = float("inf")
    for level in env.service_levels:
        action = env.candidate_action(level, obs)
        info = env.evaluate_action(action, obs)
        score = _lyapunov_candidate_score(info, queues)
        if score < best_score:
            best_score = score
            best_action = action
    return best_action


def _lyapunov_candidate_score(info: dict[str, Any], queues: dict[str, Any]) -> float:
    accuracy = float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0)))
    quality_gap = max(0.0, float(info.get("semantic_quality_gap", 0.0)))
    deadline_over = float(info.get("q_deadline_increment", 0.0))
    energy_over = float(info.get("q_energy_increment", 0.0))
    risk_over = float(info.get("q_risk_increment", 0.0))
    utm_over = float(info.get("q_utm_increment", 0.0))
    score = (
        -4.0 * accuracy
        + 0.01 * float(info.get("payload_kb", 0.0))
        + 0.02 * float(info.get("energy_j", 0.0))
        + 2.0 * float(info.get("delay_s", 0.0))
        + float(queues.get("quality", 0.0)) * quality_gap
        + float(queues.get("deadline", 0.0)) * deadline_over
        + 0.002 * float(queues.get("energy", 0.0)) * energy_over
        + float(queues.get("risk", 0.0)) * risk_over
        + float(queues.get("utm", 0.0)) * utm_over
        + 10.0 * float(_hard_violation(info))
        + 0.5 * float(info.get("semantic_uncertainty", 0.0))
    )
    return float(score)


def _hard_violation(info: dict[str, Any]) -> bool:
    return bool(
        info.get("battery_violation", False)
        or info.get("resource_violation", False)
        or info.get("airspace_conflict", False)
        or info.get("utm_constraint_violation", False)
    )


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
        "| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row.policy} | {row.semantic_success_rate:.3f} | {row.task_success_rate:.3f} | {row.average_accuracy:.3f} | {row.average_accuracy_mean:.3f} | "
            f"{row.average_uncertainty:.3f} | {row.average_semantic_quality_gap:.3f} | {row.average_q_quality:.3f} | "
            f"{row.average_q_deadline:.3f} | {row.average_q_energy:.3f} | {row.average_q_utm:.3f} | {row.average_delay:.3f} | {row.average_energy:.3f} | "
            f"{row.average_payload_kb:.3f} | {row.payload_reduction_vs_always_image:.3f} | "
            f"{row.quality_violation_rate:.3f} | {row.deadline_violation_rate:.3f} | {row.battery_violation_rate:.3f} | "
            f"{row.resource_violation_rate:.3f} | {row.airspace_conflict_rate:.3f} | {row.utm_conflict_violation_rate:.3f} | "
            f"{row.average_reward:.3f} |"
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


def _write_scenario_benchmark_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Scenario-Aware Semantic Control Benchmark",
        "",
        f"- scenarios: `{args.benchmark_scenarios}`",
        f"- smoke: `{bool(args.smoke)}`",
        f"- episodes per scenario/policy: `{args.episodes}`",
        f"- train episodes per PPO smoke: `{args.train_episodes}`",
        f"- tasks per episode: `{args.tasks_per_episode}`",
        "",
        "| scenario | policy | semantic success | accuracy LCB | accuracy mean | uncertainty | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | payload KB | deadline vio | UTM conflict | cache | token | image |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('scenario', '')} | {row.get('benchmark_policy', row.get('policy', ''))} | "
            f"{float(row.get('semantic_success_rate', 0.0)):.3f} | {float(row.get('average_accuracy', 0.0)):.3f} | "
            f"{float(row.get('average_accuracy_mean', 0.0)):.3f} | {float(row.get('average_uncertainty', 0.0)):.3f} | "
            f"{float(row.get('average_semantic_quality_gap', 0.0)):.3f} | {float(row.get('average_q_quality', 0.0)):.3f} | "
            f"{float(row.get('average_q_deadline', 0.0)):.3f} | {float(row.get('average_q_energy', 0.0)):.3f} | "
            f"{float(row.get('average_q_utm', 0.0)):.3f} | {float(row.get('average_delay', 0.0)):.3f} | "
            f"{float(row.get('average_energy', 0.0)):.3f} | {float(row.get('average_payload_kb', 0.0)):.3f} | "
            f"{float(row.get('deadline_violation_rate', 0.0)):.3f} | {float(row.get('utm_conflict_violation_rate', 0.0)):.3f} | "
            f"{float(row.get('service_level_0_ratio', 0.0)):.3f} | {float(row.get('service_level_1_ratio', 0.0)):.3f} | "
            f"{float(row.get('service_level_2_ratio', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)`.",
            "- PPO rows use Semantic-LCB Lyapunov reward/projection in smoke mode.",
            "- Scenario subdirectories contain per-scenario summaries and traces; large rollout/model artifacts remain ignored.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
