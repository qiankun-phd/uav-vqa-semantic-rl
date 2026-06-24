#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.rl.v19_ppo import (
    PPOServicePolicy,
    PPOTrainConfig,
    TwoTimescalePPOPolicy,
    load_ppo_policy,
    load_two_timescale_policy,
    normalize_hidden_layers,
    save_ppo_model,
    save_two_timescale_model,
    train_ppo,
    train_two_timescale_ppo,
)
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
    "monolithic_ppo",
    "no_mobility_actor",
    "ppo_without_lcb",
    "ppo_without_queues",
    "ppo_without_projection",
    "proposed_ppo",
    "proposed_two_timescale_ppo",
    "proposed_v2_deadline_guard",
    "proposed_v2_no_image_under_low_snr",
    "proposed_v2_nearest_uav_mobility",
)

SCENARIO_BENCHMARK_BASELINES = (
    "always_cache",
    "always_semantic_token",
    "always_image",
    "semantic_greedy",
    "lyapunov_greedy",
)

SCENARIO_BENCHMARK_PPO_VARIANTS = (
    "monolithic_ppo",
    "no_mobility_actor",
    "ppo_without_lcb",
    "ppo_without_queues",
    "ppo_without_projection",
    "proposed_ppo",
    "proposed_two_timescale_ppo",
    "proposed_v2_deadline_guard",
    "proposed_v2_no_image_under_low_snr",
    "proposed_v2_nearest_uav_mobility",
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
    average_epsilon_k: float
    failed_epsilon_k_mean: float
    failed_epsilon_k_min: float
    failed_epsilon_k_max: float
    average_q_quality: float
    average_q_deadline: float
    average_q_energy: float
    average_q_risk: float
    average_q_utm: float
    average_mobility_energy_j: float
    average_arrival_delay_s: float
    average_coverage_gain: float
    average_utm_conflict_risk: float
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
        state_version=args.state_version,
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
                average_epsilon_k=round(_mean(records, "epsilon_k"), 6),
                failed_epsilon_k_mean=round(_failed_epsilon_mean(records), 6),
                failed_epsilon_k_min=round(_failed_epsilon_min(records), 6),
                failed_epsilon_k_max=round(_failed_epsilon_max(records), 6),
                average_q_quality=round(_mean(records, "q_quality"), 6),
                average_q_deadline=round(_mean(records, "q_deadline"), 6),
                average_q_energy=round(_mean(records, "q_energy"), 6),
                average_q_risk=round(_mean(records, "q_risk"), 6),
                average_q_utm=round(_mean(records, "q_utm"), 6),
                average_mobility_energy_j=round(_mean(records, "mobility_energy_j"), 6),
                average_arrival_delay_s=round(_mean(records, "arrival_delay_s"), 6),
                average_coverage_gain=round(_mean(records, "coverage_gain"), 6),
                average_utm_conflict_risk=round(_mean(records, "utm_conflict_risk"), 6),
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


def write_run_config(output_dir: Path, args: argparse.Namespace, ppo_cfg: PPOTrainConfig | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "argv": list(sys.argv),
        "args": vars(args),
        "ppo_config": asdict(ppo_cfg) if ppo_cfg is not None else None,
    }
    path = output_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


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
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds for scenario benchmark mode.")
    parser.add_argument("--policy", default="all", choices=["all", "ppo", *BASELINE_POLICIES])
    parser.add_argument("--train-ppo", action="store_true", help="Train and evaluate centralized hybrid PPO.")
    parser.add_argument("--load-ppo-model", default=None, help="Load a trained PPO checkpoint and evaluate policy=ppo without training.")
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", default=None, help="Comma-separated PPO encoder widths. Defaults to --hidden-size,--hidden-size.")
    parser.add_argument("--state-version", default="v1", choices=["v1", "v2"], help="Observation state vector version.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"], help="Torch device for PPO training/evaluation.")
    parser.add_argument("--service-only-ppo", action="store_true", help="Disable continuous resource heads and train legacy service-level PPO.")
    parser.add_argument("--two-timescale-ppo", action="store_true", help="Train Two-timescale Mobility-aware Semantic Resource PPO.")
    parser.add_argument("--mobility-update-interval", type=int, default=3, help="Slow mobility actor update interval K in slots.")
    parser.add_argument("--no-mobility-actor", action="store_true", help="Ablation: disable learned mobility actor and use deterministic mobility defaults.")
    parser.add_argument("--benchmark-ppo-variants", default=None, help="Optional comma-separated PPO variants for scenario benchmark mode.")
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
    parser.add_argument("--deadline-aware-evidence-guard", action="store_true", help="Penalize token/image evidence whose tx+queue+infer delay exceeds the deadline.")
    parser.add_argument("--payload-delay-aware-projection", action="store_true", help="Avoid image escalation when low SNR, high payload, and short deadline coincide.")
    parser.add_argument("--no-image-under-low-snr", action="store_true", help="Diagnostic ablation: forbid image projection under low-SNR/blockage conditions.")
    parser.add_argument("--nearest-uav-mobility", action="store_true", help="Diagnostic ablation: force nearest feasible UAV mobility and suppress extra reposition.")
    parser.add_argument("--deadline-slack-reward", action="store_true", help="Tuning: reward positive deadline slack and penalize low-SNR deadline overruns.")
    parser.add_argument("--token-fast-resource-projection", action="store_true", help="Tuning: raise token bandwidth/power/CPU/GPU floors under low-SNR deadline pressure.")
    parser.add_argument("--deadline-token-cache-fallback", action="store_true", help="Tuning: allow cache fallback when token evidence is clearly deadline-infeasible and cache gap is small.")
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
        _enable_proposed_semantic_rl_defaults(args)

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
    ppo_policy: PPOServicePolicy | TwoTimescalePPOPolicy | None = None
    ppo_cfg: PPOTrainConfig | None = None

    if args.train_ppo or (args.policy == "ppo" and not args.load_ppo_model):
        train_env = make_env(args, cfg, tasks, lut, "ppo_train")
        hidden_layers = normalize_hidden_layers(args.hidden_layers, args.hidden_size)
        ppo_cfg = PPOTrainConfig(
            train_episodes=args.train_episodes,
            hidden_size=args.hidden_size,
            hidden_layers=hidden_layers,
            device=args.device,
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
            two_timescale=args.two_timescale_ppo,
            mobility_update_interval=args.mobility_update_interval,
            no_mobility_actor=args.no_mobility_actor,
            deadline_aware_evidence_guard=args.deadline_aware_evidence_guard,
            payload_delay_aware_projection=args.payload_delay_aware_projection,
            no_image_under_low_snr=args.no_image_under_low_snr,
            nearest_uav_mobility=args.nearest_uav_mobility,
            deadline_slack_reward=args.deadline_slack_reward,
            token_fast_resource_projection=args.token_fast_resource_projection,
            deadline_token_cache_fallback=args.deadline_token_cache_fallback,
        )
        if args.two_timescale_ppo:
            model, train_trace = train_two_timescale_ppo(train_env, ppo_cfg, seed=args.seed)
            model_name = "ppo_two_timescale_policy.pt"
        else:
            model, train_trace = train_ppo(train_env, ppo_cfg, seed=args.seed)
            model_name = "ppo_service_policy.pt" if args.service_only_ppo else "ppo_hybrid_policy.pt"
        model_path = Path(args.output_dir) / model_name
        if args.two_timescale_ppo:
            save_two_timescale_model(model_path, model, train_env, ppo_cfg)
            ppo_policy = TwoTimescalePPOPolicy(train_env, model, ppo_cfg)
        else:
            save_ppo_model(model_path, model, train_env, ppo_cfg)
            ppo_policy = PPOServicePolicy(train_env, model, ppo_cfg)
        print(f"wrote {model_path}")

    for policy in selected:
        eval_env = make_env(args, cfg, tasks, lut, policy)
        if policy == "ppo":
            if args.load_ppo_model:
                if args.two_timescale_ppo:
                    loaded_policy = load_two_timescale_policy(Path(args.load_ppo_model), eval_env, hidden_size=args.hidden_size, device=args.device)
                else:
                    loaded_policy = load_ppo_policy(Path(args.load_ppo_model), eval_env, hidden_size=args.hidden_size, device=args.device)
                action_fn = lambda obs, policy=loaded_policy: policy.act(obs, deterministic=True)
            elif ppo_policy is not None:
                if args.two_timescale_ppo:
                    eval_policy = TwoTimescalePPOPolicy(eval_env, ppo_policy.model, ppo_policy.cfg)
                else:
                    eval_policy = PPOServicePolicy(eval_env, ppo_policy.model, ppo_policy.cfg)
                action_fn = lambda obs, policy=eval_policy: policy.act(obs, deterministic=True)
            else:
                raise RuntimeError("Use --train-ppo before evaluating policy=ppo.")
        else:
            action_fn = lambda obs, name=policy, env=eval_env: choose_baseline_action(name, env, obs)
        records_by_policy[policy] = evaluate_policy(eval_env, policy, args.episodes, args.seed, action_fn)

    summaries = summarize(records_by_policy, args.episodes, scenario=str(args.scenario or args.formal_scenario or ""))
    write_outputs(Path(args.output_dir), summaries, records_by_policy, train_trace)
    write_run_config(Path(args.output_dir), args, ppo_cfg)
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
    seeds = _parse_seed_list(str(args.seeds))
    if args.benchmark_ppo_variants:
        ppo_variants = tuple(item.strip() for item in str(args.benchmark_ppo_variants).split(",") if item.strip())
    else:
        ppo_variants = ("proposed_two_timescale_ppo",) if bool(args.smoke) else SCENARIO_BENCHMARK_PPO_VARIANTS
    for scenario in scenarios:
        for seed in seeds:
            baseline_args = _scenario_args(args, root, scenario, seed, "baselines")
            baseline_args.policy = "all"
            baseline_args.train_ppo = False
            summaries = run_experiment(baseline_args, cfg, tasks, lut)
            by_policy = {row.policy: row for row in summaries}
            for policy in SCENARIO_BENCHMARK_BASELINES:
                row = by_policy.get(policy)
                if row is None:
                    continue
                all_rows.append(asdict(row) | {"seed": seed, "benchmark_policy": policy, "variant_kind": "heuristic"})
            for row in summaries:
                _print_summary_row(row)
            for variant in ppo_variants:
                variant_args = _scenario_args(args, root, scenario, seed, variant)
                _configure_ppo_variant(variant_args, variant)
                summaries = run_experiment(variant_args, cfg, tasks, lut)
                ppo_row = next((row for row in summaries if row.policy == "ppo"), summaries[0])
                all_rows.append(asdict(ppo_row) | {"seed": seed, "benchmark_policy": variant, "variant_kind": "rl"})
                _print_summary_row(ppo_row)
    _write_dict_csv(root / "scenario_comparison_all_seed_results.csv", all_rows)
    summary_rows = _aggregate_benchmark_rows(all_rows)
    _write_dict_csv(root / "scenario_comparison_summary.csv", summary_rows)
    _write_scenario_benchmark_report(root / "scenario_comparison_report.md", summary_rows, args)
    _write_benchmark_analysis(root / "cache_collapse_analysis.md", all_rows, summary_rows)
    print(f"wrote {root / 'scenario_comparison_all_seed_results.csv'}")
    print(f"wrote {root / 'scenario_comparison_summary.csv'}")
    print(f"wrote {root / 'scenario_comparison_report.md'}")


def _enable_proposed_semantic_rl_defaults(args: argparse.Namespace) -> None:
    args.train_ppo = True
    args.risk_aware_constraints = True
    args.semantic_reward_mode = "semantic_utility"
    args.semantic_lyapunov_control = True
    args.imitation_warm_start = True
    args.demo_policy = "semantic_greedy" if str(args.demo_policy) == "oracle_best_feasible_evidence" else args.demo_policy
    args.entropy_start = max(float(args.entropy_start), 0.10)
    args.entropy_end = max(float(args.entropy_end), 0.02)
    args.service_prior_weight = max(float(args.service_prior_weight), 0.65)
    args.service_prior_decay_episodes = max(int(args.service_prior_decay_episodes), 360)
    args.bc_aux_weight = max(float(args.bc_aux_weight), 0.28)
    args.demo_episodes = max(int(args.demo_episodes), min(50, int(args.train_episodes)))
    args.bc_epochs = max(int(args.bc_epochs), 6)


def _enable_proposed_v2_defaults(args: argparse.Namespace) -> None:
    _enable_proposed_semantic_rl_defaults(args)
    args.two_timescale_ppo = True
    args.deadline_aware_evidence_guard = True
    args.payload_delay_aware_projection = True


def _scenario_args(args: argparse.Namespace, root: Path, scenario: str, seed: int, run_name: str) -> argparse.Namespace:
    scenario_args = argparse.Namespace(**vars(args))
    scenario_args.scenario_benchmark = False
    scenario_args.scenario = scenario
    scenario_args.formal_scenario = None
    scenario_args.seed = int(seed)
    scenario_args.output_dir = str(root / scenario / f"seed{seed}" / run_name)
    return scenario_args


def _configure_ppo_variant(args: argparse.Namespace, variant: str) -> None:
    args.policy = "ppo"
    args.train_ppo = True
    args.semantic_lyapunov_control = True
    args.risk_aware_constraints = True
    args.semantic_reward_mode = "semantic_utility"
    args.two_timescale_ppo = False
    args.no_mobility_actor = False
    if variant == "monolithic_ppo":
        _enable_proposed_semantic_rl_defaults(args)
    elif variant == "no_mobility_actor":
        _enable_proposed_semantic_rl_defaults(args)
        args.two_timescale_ppo = True
        args.no_mobility_actor = True
    elif variant == "ppo_without_lcb":
        _enable_proposed_semantic_rl_defaults(args)
        args.no_semantic_lcb = True
    elif variant == "ppo_without_queues":
        _enable_proposed_semantic_rl_defaults(args)
        args.no_lyapunov_queues = True
    elif variant == "ppo_without_projection":
        _enable_proposed_semantic_rl_defaults(args)
        args.no_resource_projection = True
        args.no_semantic_projection = True
    elif variant == "proposed_ppo":
        _enable_proposed_semantic_rl_defaults(args)
    elif variant == "proposed_two_timescale_ppo":
        _enable_proposed_semantic_rl_defaults(args)
        args.two_timescale_ppo = True
    elif variant == "proposed_v2_deadline_guard":
        _enable_proposed_v2_defaults(args)
    elif variant == "proposed_v2_no_image_under_low_snr":
        _enable_proposed_v2_defaults(args)
        args.no_image_under_low_snr = True
    elif variant == "proposed_v2_nearest_uav_mobility":
        _enable_proposed_v2_defaults(args)
        args.nearest_uav_mobility = True
    else:
        raise ValueError(f"unknown PPO benchmark variant: {variant}")


def _parse_seed_list(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one seed")
    return seeds


def _aggregate_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("scenario", "")), str(row.get("benchmark_policy", row.get("policy", ""))))
        grouped.setdefault(key, []).append(row)
    metric_keys = [
        "semantic_success_rate",
        "task_success_rate",
        "average_accuracy",
        "average_accuracy_mean",
        "average_uncertainty",
        "average_semantic_quality_gap",
        "average_epsilon_k",
        "failed_epsilon_k_mean",
        "failed_epsilon_k_min",
        "failed_epsilon_k_max",
        "average_q_quality",
        "average_q_deadline",
        "average_q_energy",
        "average_q_utm",
        "average_mobility_energy_j",
        "average_arrival_delay_s",
        "average_coverage_gain",
        "average_utm_conflict_risk",
        "average_delay",
        "average_energy",
        "average_payload_kb",
        "deadline_violation_rate",
        "utm_conflict_violation_rate",
        "service_level_0_ratio",
        "service_level_1_ratio",
        "service_level_2_ratio",
    ]
    out: list[dict[str, Any]] = []
    for (scenario, policy), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "scenario": scenario,
            "benchmark_policy": policy,
            "seeds": ",".join(str(int(item.get("seed", 0))) for item in items),
            "runs": len(items),
            "episodes_per_seed": items[0].get("episodes", ""),
            "tasks_total": int(sum(float(item.get("tasks", 0.0)) for item in items)),
            "variant_kind": items[0].get("variant_kind", ""),
        }
        for key in metric_keys:
            values = [float(item.get(key, 0.0)) for item in items]
            row[f"{key}_mean"] = round(_mean_float(values), 6)
            row[f"{key}_std"] = round(_std_float(values), 6)
        out.append(row)
    return out


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
    if int(action.get("service_level", 0)) == 0:
        action = dict(action)
        action.update({"bandwidth": 0.0, "power": 0.0, "cpu_share": 0.0, "gpu_share": 0.0})
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


def _failed_epsilons(records: list[V19StepRecord]) -> list[float]:
    return [float(record.epsilon_k) for record in records if not bool(record.semantic_success)]


def _failed_epsilon_mean(records: list[V19StepRecord]) -> float:
    values = _failed_epsilons(records)
    return sum(values) / len(values) if values else 0.0


def _failed_epsilon_min(records: list[V19StepRecord]) -> float:
    values = _failed_epsilons(records)
    return min(values) if values else 0.0


def _failed_epsilon_max(records: list[V19StepRecord]) -> float:
    values = _failed_epsilons(records)
    return max(values) if values else 0.0


def _mean_float(values: list[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def _std_float(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean_float(values)
    return (sum((float(value) - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


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
        "| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row.policy} | {row.semantic_success_rate:.3f} | {row.task_success_rate:.3f} | {row.average_accuracy:.3f} | {row.average_accuracy_mean:.3f} | "
            f"{row.average_uncertainty:.3f} | {row.average_epsilon_k:.3f} | {row.failed_epsilon_k_mean:.3f} | "
            f"{row.failed_epsilon_k_min:.3f}-{row.failed_epsilon_k_max:.3f} | {row.average_semantic_quality_gap:.3f} | {row.average_q_quality:.3f} | "
            f"{row.average_q_deadline:.3f} | {row.average_q_energy:.3f} | {row.average_q_utm:.3f} | {row.average_delay:.3f} | {row.average_energy:.3f} | "
            f"{row.average_mobility_energy_j:.3f} | {row.average_arrival_delay_s:.3f} | {row.average_coverage_gain:.3f} | "
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
        f"- seeds: `{args.seeds}`",
        f"- episodes per scenario/policy: `{args.episodes}`",
        f"- train episodes per PPO variant: `{args.train_episodes}`",
        f"- tasks per episode: `{args.tasks_per_episode}`",
        "",
        "| scenario | policy | semantic success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed eps mean | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage | payload KB | deadline vio | UTM conflict | cache | token | image |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('scenario', '')} | {row.get('benchmark_policy', row.get('policy', ''))} | "
            f"{_metric(row, 'semantic_success_rate'):.3f} | {_metric(row, 'average_accuracy'):.3f} | "
            f"{_metric(row, 'average_accuracy_mean'):.3f} | {_metric(row, 'average_uncertainty'):.3f} | "
            f"{_metric(row, 'average_epsilon_k'):.3f} | {_metric(row, 'failed_epsilon_k_mean'):.3f} | "
            f"{_metric(row, 'average_semantic_quality_gap'):.3f} | {_metric(row, 'average_q_quality'):.3f} | "
            f"{_metric(row, 'average_q_deadline'):.3f} | {_metric(row, 'average_q_energy'):.3f} | "
            f"{_metric(row, 'average_q_utm'):.3f} | {_metric(row, 'average_delay'):.3f} | "
            f"{_metric(row, 'average_energy'):.3f} | {_metric(row, 'average_mobility_energy_j'):.3f} | "
            f"{_metric(row, 'average_arrival_delay_s'):.3f} | {_metric(row, 'average_coverage_gain'):.3f} | "
            f"{_metric(row, 'average_payload_kb'):.3f} | "
            f"{_metric(row, 'deadline_violation_rate'):.3f} | {_metric(row, 'utm_conflict_violation_rate'):.3f} | "
            f"{_metric(row, 'service_level_0_ratio'):.3f} | {_metric(row, 'service_level_1_ratio'):.3f} | "
            f"{_metric(row, 'service_level_2_ratio'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)`.",
            "- Proposed PPO rows use Semantic-LCB Lyapunov reward/projection, oracle warm-start, service-level curriculum, and cache shortfall penalties.",
            "- Scenario subdirectories contain per-scenario summaries and traces; large rollout/model artifacts remain ignored.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(row: dict[str, Any], key: str) -> float:
    if f"{key}_mean" in row:
        return float(row.get(f"{key}_mean", 0.0))
    return float(row.get(key, 0.0))


def _write_benchmark_analysis(path: Path, all_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    proposed_rows = [row for row in summary_rows if str(row.get("benchmark_policy", "")) == "proposed_ppo"]
    cache_rows = [row for row in summary_rows if str(row.get("benchmark_policy", "")) == "always_cache"]
    cache_by_scenario = {str(row.get("scenario", "")): row for row in cache_rows}
    lines = [
        "# Scenario Benchmark Cache-Collapse Analysis",
        "",
        "## Diagnosis",
        "",
        "- Previous smoke PPO had high cache ratios because cache minimized delay/energy/payload while the semantic shortfall penalty was too small.",
        "- The v3 controller persists `epsilon_k`, uses risk/staleness/UTM-aware cache shortfall penalties, distills semantic-greedy routing, and keeps a stronger semantic-token prior.",
        "- Compute-aware projection prefers semantic tokens over cache when token evidence reduces LCB shortfall under edge/deadline pressure; UTM conflicts are recorded as risk/queue costs instead of being hidden by cache fallback.",
        "- Cache actions are projected to zero bandwidth/power/cpu/gpu; token/image actions retain service-dependent resource floors.",
        "",
        "## Proposed PPO vs Always Cache",
        "",
        "| scenario | proposed semantic success | cache semantic success | proposed cache mix | proposed token mix | proposed image mix | cache gap | proposed gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in proposed_rows:
        scenario = str(row.get("scenario", ""))
        cache = cache_by_scenario.get(scenario, {})
        lines.append(
            f"| {scenario} | {_metric(row, 'semantic_success_rate'):.3f} | {_metric(cache, 'semantic_success_rate'):.3f} | "
            f"{_metric(row, 'service_level_0_ratio'):.3f} | {_metric(row, 'service_level_1_ratio'):.3f} | "
            f"{_metric(row, 'service_level_2_ratio'):.3f} | {_metric(cache, 'average_semantic_quality_gap'):.3f} | "
            f"{_metric(row, 'average_semantic_quality_gap'):.3f} |"
        )
    lines.extend(["", f"- raw seed rows: {len(all_rows)}", "- `scenario_comparison_summary.csv` reports mean/std across seeds."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
