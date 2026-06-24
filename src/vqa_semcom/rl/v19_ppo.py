from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vqa_semcom.rl.v19_resource_env import V19LUTResourceEnv

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    np = None

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical, Normal
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    torch = None
    nn = None
    Categorical = None
    Normal = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


RESOURCE_KEYS = ("bandwidth", "power", "cpu_share", "gpu_share")
MOBILITY_MODES = ("stay", "serve_task", "reposition", "avoid_conflict", "return_base")
MOBILITY_CONTINUOUS_KEYS = ("waypoint_dx", "waypoint_dy", "altitude_delta")
SEMANTIC_REWARD_MODES = ("env", "semantic_utility", "no_semantic_utility", "accuracy_only", "uncertainty_aware")


def require_torch() -> None:
    if torch is None or nn is None or Categorical is None or Normal is None:
        raise ModuleNotFoundError("V1.9 semantic hybrid PPO requires torch.") from _TORCH_IMPORT_ERROR


def resolve_torch_device(device: str | Any | None = "auto") -> Any:
    require_torch()
    if hasattr(device, "type"):
        requested = str(device)
    else:
        requested = str(device or "auto").strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested torch device {requested!r}, but CUDA is not available")
    return torch.device(requested)


def torch_device_label(device: str | Any | None = "auto") -> str:
    resolved = resolve_torch_device(device)
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return str(resolved)


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return resolve_torch_device("auto")


def _model_obs_dim(model: Any) -> int | None:
    if nn is None:
        return None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)
    return None


def _obs_tensor(obs: dict[str, Any], device: Any, expected_dim: int | None = None) -> Any:
    vector = list(obs.get("vector", []))
    if expected_dim is not None and expected_dim > 0:
        if len(vector) < expected_dim:
            vector = vector + [0.0] * (expected_dim - len(vector))
        elif len(vector) > expected_dim:
            vector = vector[:expected_dim]
    return torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)


def normalize_hidden_layers(hidden_layers: Any = None, hidden_size: int = 128) -> tuple[int, ...]:
    if hidden_layers is None or hidden_layers == "":
        return (int(hidden_size), int(hidden_size))
    if isinstance(hidden_layers, str):
        layers = [int(item.strip()) for item in hidden_layers.split(",") if item.strip()]
    else:
        layers = [int(item) for item in hidden_layers]
    if not layers:
        layers = [int(hidden_size), int(hidden_size)]
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"hidden layers must be positive: {layers}")
    return tuple(layers)


def _build_mlp(input_dim: int, hidden_layers: tuple[int, ...]) -> tuple[Any, int]:
    layers: list[Any] = []
    last_dim = int(input_dim)
    for width in hidden_layers:
        layers.append(nn.Linear(last_dim, int(width)))
        layers.append(nn.ReLU())
        last_dim = int(width)
    return nn.Sequential(*layers), last_dim


@dataclass(frozen=True)
class PPOTrainConfig:
    train_episodes: int = 120
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01
    entropy_weight_start: float = 0.05
    entropy_weight_end: float = 0.005
    entropy_decay_episodes: int = 120
    value_weight: float = 0.5
    hidden_size: int = 128
    hidden_layers: tuple[int, ...] = (128, 128)
    device: str = "auto"
    hybrid_actions: bool = True
    constrained: bool = True
    risk_aware_constraints: bool = False
    safety_layer: bool = True
    semantic_reward_mode: str = "env"
    semantic_success_weight: float = 12.0
    semantic_accuracy_weight: float = 3.5
    semantic_margin_weight: float = 6.0
    semantic_gap_weight: float = 9.0
    cache_shortfall_penalty_weight: float = 16.0
    high_risk_cache_penalty_weight: float = 5.0
    high_epsilon_cache_threshold: float = 0.65
    cache_override_gap_threshold: float = 0.12
    cache_override_min_improvement: float = 0.04
    cache_stale_penalty_weight: float = 3.0
    cache_utm_penalty_weight: float = 4.0
    token_near_success_gap: float = 0.08
    token_projection_bonus: float = 3.0
    non_cache_semantic_bonus: float = 1.5
    semantic_token_exploration_bonus: float = 0.75
    delay_cost_weight: float = 0.25
    energy_cost_weight: float = 0.12
    payload_cost_weight: float = 0.10
    conflict_cost_weight: float = 2.0
    battery_cost_weight: float = 2.0
    gpu_cost_weight: float = 1.5
    utm_conflict_cost_weight: float = 1.5
    dss_delay_cost_weight: float = 0.25
    off_nominal_cost_weight: float = 1.0
    use_semantic_lcb: bool = True
    lyapunov_reward: bool = False
    use_lyapunov_queues: bool = True
    queue_quality_weight: float = 2.5
    queue_deadline_weight: float = 0.8
    queue_energy_weight: float = 0.25
    queue_risk_weight: float = 1.0
    queue_utm_weight: float = 1.0
    uncertainty_cost_weight: float = 0.35
    energy_budget_j: float = 500.0
    quality_cost_limit: float = 0.0
    deadline_cost_limit: float = 0.0
    quality_cost_limit_normal: float = 0.05
    quality_cost_limit_critical: float = 0.02
    deadline_cost_limit_normal: float = 0.05
    deadline_cost_limit_critical: float = 0.02
    conflict_cost_limit: float = 0.0
    battery_cost_limit: float = 0.0
    gpu_cost_limit: float = 0.0
    lambda_lr: float = 0.05
    lambda_max: float = 20.0
    bandwidth_floor: float = 0.02
    share_floor: float = 0.01
    semantic_token_bandwidth_floor: float = 0.35
    image_bandwidth_floor: float = 0.65
    roi_bandwidth_floor: float = 0.45
    semantic_token_cpu_floor: float = 0.25
    image_cpu_floor: float = 0.55
    roi_cpu_floor: float = 0.35
    semantic_token_gpu_floor: float = 0.05
    image_gpu_floor: float = 0.30
    roi_gpu_floor: float = 0.20
    semantic_token_power_floor: float = 0.30
    image_power_floor: float = 0.60
    roi_power_floor: float = 0.45
    power_min_w: float = 0.05
    power_max_w: float = 1.0
    semantic_projection: bool = True
    resource_projection: bool = True
    service_prior_weight: float = 0.35
    service_prior_decay_episodes: int = 240
    imitation_warm_start: bool = False
    demo_policy: str = "oracle_best_feasible_evidence"
    demo_episodes: int = 40
    bc_epochs: int = 8
    bc_batch_size: int = 128
    bc_weight: float = 1.0
    bc_aux_weight: float = 0.05
    two_timescale: bool = False
    mobility_update_interval: int = 3
    no_mobility_actor: bool = False
    waypoint_delta_max_m: float = 80.0
    altitude_delta_max_m: float = 20.0
    flight_energy_cost_weight: float = 0.08
    arrival_delay_cost_weight: float = 0.35
    coverage_gain_weight: float = 0.5
    edge_queue_cost_weight: float = 0.15
    deadline_aware_evidence_guard: bool = False
    deadline_guard_slack: float = 0.95
    payload_delay_aware_projection: bool = False
    no_image_under_low_snr: bool = False
    low_snr_guard_threshold_db: float = -5.0
    high_payload_guard_kb: float = 20.0
    short_deadline_guard_s: float = 6.0
    nearest_uav_mobility: bool = False
    critical_burst_nearest_uav: bool = True


@dataclass
class DualState:
    quality_normal: float = 0.0
    quality_critical: float = 0.0
    deadline_normal: float = 0.0
    deadline_critical: float = 0.0
    conflict: float = 0.0
    battery: float = 0.0
    gpu: float = 0.0

    def penalty(self, info: dict[str, Any]) -> float:
        risk = str(info.get("risk_level", "normal"))
        quality_lambda = self.quality_critical if risk == "critical" else self.quality_normal
        deadline_lambda = self.deadline_critical if risk == "critical" else self.deadline_normal
        return (
            quality_lambda * float(bool(info.get("quality_violation", False)))
            + deadline_lambda * float(bool(info.get("deadline_violation", False)))
            + self.conflict
            * float(bool(info.get("airspace_conflict", False)) or bool(info.get("utm_constraint_violation", False)))
            + self.battery * float(bool(info.get("battery_violation", False)))
            + self.gpu * float(bool(info.get("resource_violation", False)))
        )

    def to_trace(self) -> dict[str, float]:
        return {
            "lambda_quality_normal": float(self.quality_normal),
            "lambda_quality_critical": float(self.quality_critical),
            "lambda_deadline_normal": float(self.deadline_normal),
            "lambda_deadline_critical": float(self.deadline_critical),
            "lambda_conflict": float(self.conflict),
            "lambda_battery": float(self.battery),
            "lambda_gpu": float(self.gpu),
            "lambda_quality": float((self.quality_normal + self.quality_critical) / 2.0),
            "lambda_deadline": float((self.deadline_normal + self.deadline_critical) / 2.0),
        }


class HybridActorCritic(nn.Module if nn is not None else object):
    """Semantic-utility-guided centralized cognitive controller.

    The high-level head selects semantic service level.  Continuous heads
    allocate communication and edge-compute resources.  UAV assignment and
    safety projection are handled by the cognitive safety layer.
    """

    def __init__(
        self,
        obs_dim: int,
        num_service_actions: int,
        hidden_size: int = 128,
        hidden_layers: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        self.hidden_layers = normalize_hidden_layers(hidden_layers, hidden_size)
        self.encoder, encoder_dim = _build_mlp(obs_dim, self.hidden_layers)
        self.service_actor = nn.Linear(encoder_dim, num_service_actions)
        self.resource_actor = nn.Linear(encoder_dim, len(RESOURCE_KEYS))
        self.resource_log_std = nn.Parameter(torch.full((len(RESOURCE_KEYS),), -0.7))
        self.critic = nn.Linear(encoder_dim, 1)

    def forward(self, obs: Any) -> tuple[Any, Any, Any, Any]:
        z = self.encoder(obs.float())
        service_logits = torch.nan_to_num(self.service_actor(z), nan=0.0, posinf=20.0, neginf=-20.0)
        resource_mean = torch.nan_to_num(self.resource_actor(z), nan=0.0, posinf=5.0, neginf=-5.0)
        resource_log_std = torch.nan_to_num(self.resource_log_std, nan=-0.7, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0).expand_as(resource_mean)
        value = torch.nan_to_num(self.critic(z).squeeze(-1), nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        return service_logits, resource_mean, resource_log_std, value


ServiceLevelActorCritic = HybridActorCritic


class TwoTimescaleMobilitySemanticActorCritic(nn.Module if nn is not None else object):
    """Two-timescale mobility-aware semantic resource PPO network.

    A shared encoder feeds a slow mobility actor, a fast semantic-resource
    actor, and a centralized critic that evaluates the joint action context.
    The slow actor is sampled every ``K`` slots by the rollout collector while
    the fast actor is sampled for every task/slot.
    """

    def __init__(
        self,
        obs_dim: int,
        num_service_actions: int,
        num_uavs: int,
        hidden_size: int = 128,
        hidden_layers: tuple[int, ...] | list[int] | None = None,
        mobility_modes: tuple[str, ...] = MOBILITY_MODES,
    ) -> None:
        require_torch()
        super().__init__()
        self.mobility_modes = tuple(mobility_modes)
        self.num_uavs = max(1, int(num_uavs))
        self.hidden_layers = normalize_hidden_layers(hidden_layers, hidden_size)
        self.encoder, encoder_dim = _build_mlp(obs_dim, self.hidden_layers)
        self.mobility_uav_actor = nn.Linear(encoder_dim, self.num_uavs)
        self.mobility_mode_actor = nn.Linear(encoder_dim, len(self.mobility_modes))
        self.mobility_continuous_actor = nn.Linear(encoder_dim, len(MOBILITY_CONTINUOUS_KEYS))
        self.mobility_log_std = nn.Parameter(torch.full((len(MOBILITY_CONTINUOUS_KEYS),), -0.8))
        self.service_actor = nn.Linear(encoder_dim, num_service_actions)
        self.resource_actor = nn.Linear(encoder_dim, len(RESOURCE_KEYS))
        self.resource_log_std = nn.Parameter(torch.full((len(RESOURCE_KEYS),), -0.7))
        self.critic = nn.Linear(encoder_dim, 1)

    def forward(self, obs: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        z = self.encoder(obs.float())
        uav_logits = torch.nan_to_num(self.mobility_uav_actor(z), nan=0.0, posinf=20.0, neginf=-20.0)
        mode_logits = torch.nan_to_num(self.mobility_mode_actor(z), nan=0.0, posinf=20.0, neginf=-20.0)
        mobility_mean = torch.nan_to_num(self.mobility_continuous_actor(z), nan=0.0, posinf=5.0, neginf=-5.0)
        mobility_log_std = torch.nan_to_num(self.mobility_log_std, nan=-0.8, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0).expand_as(mobility_mean)
        service_logits = torch.nan_to_num(self.service_actor(z), nan=0.0, posinf=20.0, neginf=-20.0)
        resource_mean = torch.nan_to_num(self.resource_actor(z), nan=0.0, posinf=5.0, neginf=-5.0)
        resource_log_std = torch.nan_to_num(self.resource_log_std, nan=-0.7, posinf=2.0, neginf=-5.0).clamp(-5.0, 2.0).expand_as(resource_mean)
        value = torch.nan_to_num(self.critic(z).squeeze(-1), nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        return uav_logits, mode_logits, mobility_mean, mobility_log_std, service_logits, resource_mean, resource_log_std, value


class TwoTimescalePPOPolicy:
    def __init__(
        self,
        env: V19LUTResourceEnv,
        model: TwoTimescaleMobilitySemanticActorCritic,
        cfg: PPOTrainConfig | None = None,
    ) -> None:
        require_torch()
        self.env = env
        self.cfg = cfg or PPOTrainConfig(two_timescale=True)
        self.device = _model_device(model)
        self.model = model.to(self.device)
        self.model.eval()
        self.obs_dim = _model_obs_dim(self.model)
        self._cached_mobility: dict[str, Any] | None = None
        self._cached_step = -1

    def act(self, obs: dict[str, Any], deterministic: bool = True) -> dict[str, Any]:
        obs_tensor = _obs_tensor(obs, self.device, self.obs_dim)
        step = int(obs.get("episode_step", 0))
        with torch.no_grad():
            (
                uav_logits,
                mode_logits,
                mobility_mean,
                mobility_log_std,
                service_logits,
                resource_mean,
                resource_log_std,
                _value,
            ) = self.model(obs_tensor)
            if self.cfg.no_mobility_actor:
                mobility = _default_mobility_action(self.env, obs, service_level=None)
            elif self._cached_mobility is None or step % max(1, int(self.cfg.mobility_update_interval)) == 0:
                mobility = _sample_mobility_action(
                    self.env,
                    obs,
                    self.cfg,
                    uav_logits,
                    mode_logits,
                    mobility_mean,
                    mobility_log_std,
                    deterministic=deterministic,
                )[0]
                self._cached_mobility = mobility
                self._cached_step = step
            else:
                mobility = dict(self._cached_mobility)
            mask = _service_mask_tensor(obs, self.env.service_levels, service_logits.device, self.cfg)
            service_logits = _apply_service_mask(service_logits, mask)
            if deterministic:
                service_idx = int(torch.argmax(service_logits, dim=-1).item())
                raw_resources = resource_mean.squeeze(0)
            else:
                service_idx = int(Categorical(logits=service_logits).sample().item())
                raw_resources = Normal(resource_mean, resource_log_std.exp()).sample().squeeze(0)
            resources = torch.sigmoid(raw_resources).detach().cpu().tolist()
        service_level = self.env.service_levels[service_idx]
        return _build_two_timescale_action(self.env, obs, service_level, resources, mobility, self.cfg)


TwoTimescaleActorCritic = TwoTimescaleMobilitySemanticActorCritic


class PPOServicePolicy:
    def __init__(
        self,
        env: V19LUTResourceEnv,
        model: HybridActorCritic,
        cfg: PPOTrainConfig | None = None,
    ) -> None:
        require_torch()
        self.env = env
        self.cfg = cfg or PPOTrainConfig()
        self.device = _model_device(model)
        self.model = model.to(self.device)
        self.model.eval()
        self.obs_dim = _model_obs_dim(self.model)

    def act(self, obs: dict[str, Any], deterministic: bool = True) -> dict[str, Any]:
        obs_tensor = _obs_tensor(obs, self.device, self.obs_dim)
        with torch.no_grad():
            logits, resource_mean, resource_log_std, _value = self.model(obs_tensor)
            mask = _service_mask_tensor(obs, self.env.service_levels, logits.device, self.cfg)
            logits = _apply_service_mask(logits, mask)
            if deterministic:
                action_index = int(torch.argmax(logits, dim=-1).item())
                raw_resources = resource_mean.squeeze(0)
            else:
                action_index = int(Categorical(logits=logits).sample().item())
                raw_resources = Normal(resource_mean, resource_log_std.exp()).sample().squeeze(0)
            resources = torch.sigmoid(raw_resources).detach().cpu().tolist()
        level = self.env.service_levels[action_index]
        return _build_hybrid_action(self.env, obs, level, resources, self.cfg)


def train_ppo(
    env: V19LUTResourceEnv,
    cfg: PPOTrainConfig,
    seed: int = 0,
) -> tuple[HybridActorCritic, list[dict[str, float]]]:
    require_torch()
    if cfg.semantic_reward_mode not in SEMANTIC_REWARD_MODES:
        raise ValueError(f"unknown semantic_reward_mode: {cfg.semantic_reward_mode}")
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    device = resolve_torch_device(cfg.device)
    print(f"PPO training device: {torch_device_label(device)}; torch cuda available: {torch.cuda.is_available()}")
    obs = env.reset(seed=seed, options={"policy_name": "ppo_train_probe"})
    obs_dim = int(len(obs["vector"]))
    hidden_layers = normalize_hidden_layers(cfg.hidden_layers, cfg.hidden_size)
    cfg = PPOTrainConfig(**{**asdict(cfg), "hidden_layers": hidden_layers})
    model = HybridActorCritic(obs_dim, len(env.service_levels), cfg.hidden_size, cfg.hidden_layers).to(device)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    except ImportError:
        optimizer = None

    demos = _collect_demonstrations(env, cfg, seed) if cfg.imitation_warm_start else []
    bc_loss = 0.0
    if optimizer is not None and demos:
        bc_loss = _behavior_clone(model, optimizer, demos, cfg)

    dual = DualState()
    trace: list[dict[str, float]] = []
    for episode in range(cfg.train_episodes):
        rollout = _collect_episode(env, model, seed + episode, cfg, dual)
        if optimizer is not None:
            _ppo_update(model, optimizer, rollout, cfg, demos, episode)
        _update_duals(dual, rollout, cfg)
        row = {
            "episode": float(episode),
            "raw_return": float(sum(rollout["raw_rewards"])),
            "semantic_return": float(sum(rollout["semantic_rewards"])),
            "shaped_return": float(sum(rollout["rewards"])),
            "return": float(sum(rollout["raw_rewards"])),
            "success_rate": float(_mean(rollout["successes"])),
            "mean_reward": float(_mean(rollout["raw_rewards"])),
            "mean_semantic_reward": float(_mean(rollout["semantic_rewards"])),
            "mean_shaped_reward": float(_mean(rollout["rewards"])),
            "mean_semantic_accuracy_lcb": float(_mean(rollout["semantic_accuracy_lcbs"])),
            "mean_semantic_uncertainty": float(_mean(rollout["semantic_uncertainties"])),
            "mean_semantic_payload_kb": float(_mean(rollout["semantic_payload_kbs"])),
            "mean_semantic_quality_gap": float(_mean(rollout["semantic_quality_gaps"])),
            "mean_q_quality": float(_mean(rollout["q_quality"])),
            "mean_q_deadline": float(_mean(rollout["q_deadline"])),
            "mean_q_energy": float(_mean(rollout["q_energy"])),
            "mean_q_risk": float(_mean(rollout["q_risk"])),
            "mean_q_utm": float(_mean(rollout["q_utm"])),
            "max_q_quality": float(max(rollout["q_quality"] or [0.0])),
            "max_q_deadline": float(max(rollout["q_deadline"] or [0.0])),
            "max_q_energy": float(max(rollout["q_energy"] or [0.0])),
            "max_q_risk": float(max(rollout["q_risk"] or [0.0])),
            "max_q_utm": float(max(rollout["q_utm"] or [0.0])),
            "non_cache_ratio": float(_mean(rollout["non_cache_actions"])),
            "quality_cost": float(_mean(rollout["quality_costs"])),
            "deadline_cost": float(_mean(rollout["deadline_costs"])),
            "quality_cost_normal": float(_mean(rollout["quality_costs_normal"])),
            "quality_cost_critical": float(_mean(rollout["quality_costs_critical"])),
            "deadline_cost_normal": float(_mean(rollout["deadline_costs_normal"])),
            "deadline_cost_critical": float(_mean(rollout["deadline_costs_critical"])),
            "conflict_cost": float(_mean(rollout["conflict_costs"])),
            "battery_cost": float(_mean(rollout["battery_costs"])),
            "gpu_cost": float(_mean(rollout["gpu_costs"])),
            "bc_loss": float(bc_loss),
            "demo_samples": float(len(demos)),
            "entropy_weight": float(_scheduled_entropy_weight(cfg, episode)),
            "service_prior_weight": float(_scheduled_service_prior_weight(cfg, episode)),
            "mean_bandwidth_share": float(_mean(rollout["bandwidth_shares"])),
            "mean_power_w": float(_mean(rollout["power_w"])),
            "mean_cpu_share": float(_mean(rollout["cpu_shares"])),
            "mean_gpu_share": float(_mean(rollout["gpu_shares"])),
        }
        row.update(dual.to_trace())
        trace.append(row)
    return model, trace


def train_two_timescale_ppo(
    env: V19LUTResourceEnv,
    cfg: PPOTrainConfig,
    seed: int = 0,
) -> tuple[TwoTimescaleMobilitySemanticActorCritic, list[dict[str, float]]]:
    require_torch()
    if cfg.semantic_reward_mode not in SEMANTIC_REWARD_MODES:
        raise ValueError(f"unknown semantic_reward_mode: {cfg.semantic_reward_mode}")
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    device = resolve_torch_device(cfg.device)
    print(f"PPO training device: {torch_device_label(device)}; torch cuda available: {torch.cuda.is_available()}")
    obs = env.reset(seed=seed, options={"policy_name": "two_timescale_ppo_train_probe"})
    obs_dim = int(len(obs["vector"]))
    num_uavs = int(env.action_spec().get("num_uavs", max(1, len(obs.get("uav_state", []) or []))))
    hidden_layers = normalize_hidden_layers(cfg.hidden_layers, cfg.hidden_size)
    cfg = PPOTrainConfig(**{**asdict(cfg), "two_timescale": True, "hidden_layers": hidden_layers})
    model = TwoTimescaleMobilitySemanticActorCritic(obs_dim, len(env.service_levels), num_uavs, cfg.hidden_size, cfg.hidden_layers).to(device)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    except ImportError:
        optimizer = None

    demos = _collect_demonstrations(env, cfg, seed) if cfg.imitation_warm_start else []
    bc_loss = 0.0
    if optimizer is not None and demos:
        bc_loss = _behavior_clone_two_timescale(model, optimizer, demos, cfg)

    dual = DualState()
    trace: list[dict[str, float]] = []
    for episode in range(cfg.train_episodes):
        rollout = _collect_two_timescale_episode(env, model, seed + episode, cfg, dual)
        if optimizer is not None:
            _two_timescale_ppo_update(model, optimizer, rollout, cfg, demos, episode)
        _update_duals(dual, rollout, cfg)
        row = _trace_row_from_rollout(episode, rollout, cfg, dual, demos, bc_loss)
        row.update(
            {
                "mobility_update_interval": float(cfg.mobility_update_interval),
                "mean_flight_energy_j": float(_mean(rollout["mobility_energy_j"])),
                "mean_arrival_delay_s": float(_mean(rollout["arrival_delay_s"])),
                "mean_coverage_gain": float(_mean(rollout["coverage_gain"])),
                "mean_utm_conflict_risk": float(_mean(rollout["utm_conflict_risk"])),
                "mobility_reuse_ratio": float(_mean(rollout["mobility_reused"])),
                "serve_task_ratio": float(_mean(rollout["serve_task_mode"])),
                "avoid_conflict_ratio": float(_mean(rollout["avoid_conflict_mode"])),
            }
        )
        trace.append(row)
    return model, trace


def save_ppo_model(path: Path, model: HybridActorCritic, env: V19LUTResourceEnv, cfg: PPOTrainConfig | None = None) -> None:
    require_torch()
    cfg = cfg or PPOTrainConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "obs_dim": model.encoder[0].in_features,
        "num_actions": len(env.service_levels),
        "service_levels": env.service_levels,
        "model_type": "semantic_hybrid_actor_critic",
        "resource_keys": RESOURCE_KEYS,
        "hidden_layers": list(getattr(model, "hidden_layers", normalize_hidden_layers(cfg.hidden_layers, cfg.hidden_size))),
        "config": asdict(cfg),
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "obs_dim": payload["obs_dim"],
                "service_levels": env.service_levels,
                "model_type": payload["model_type"],
                "resource_keys": list(RESOURCE_KEYS),
                "hidden_layers": payload["hidden_layers"],
                "config": asdict(cfg),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_two_timescale_model(
    path: Path,
    model: TwoTimescaleMobilitySemanticActorCritic,
    env: V19LUTResourceEnv,
    cfg: PPOTrainConfig | None = None,
) -> None:
    require_torch()
    cfg = cfg or PPOTrainConfig(two_timescale=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "obs_dim": model.encoder[0].in_features,
        "num_actions": len(env.service_levels),
        "num_uavs": model.num_uavs,
        "service_levels": env.service_levels,
        "model_type": "two_timescale_mobility_semantic_actor_critic",
        "resource_keys": RESOURCE_KEYS,
        "mobility_modes": list(model.mobility_modes),
        "mobility_continuous_keys": list(MOBILITY_CONTINUOUS_KEYS),
        "hidden_layers": list(getattr(model, "hidden_layers", normalize_hidden_layers(cfg.hidden_layers, cfg.hidden_size))),
        "config": asdict(cfg),
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "obs_dim": payload["obs_dim"],
                "service_levels": env.service_levels,
                "num_uavs": model.num_uavs,
                "model_type": payload["model_type"],
                "resource_keys": list(RESOURCE_KEYS),
                "mobility_modes": list(model.mobility_modes),
                "hidden_layers": payload["hidden_layers"],
                "mobility_update_interval": int(cfg.mobility_update_interval),
                "config": asdict(cfg),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_two_timescale_policy(
    path: Path,
    env: V19LUTResourceEnv,
    hidden_size: int = 128,
    device: str | Any | None = None,
) -> TwoTimescalePPOPolicy:
    require_torch()
    target_device = resolve_torch_device(device or "auto")
    payload = torch.load(path, map_location=target_device)
    cfg_payload = payload.get("config") or {}
    cfg = PPOTrainConfig(**{key: value for key, value in cfg_payload.items() if key in PPOTrainConfig.__dataclass_fields__})
    hidden_layers = normalize_hidden_layers(
        payload.get("hidden_layers", cfg_payload.get("hidden_layers")),
        int(cfg_payload.get("hidden_size", hidden_size)),
    )
    cfg = PPOTrainConfig(**{**asdict(cfg), "device": str(target_device), "hidden_layers": hidden_layers})
    model = TwoTimescaleMobilitySemanticActorCritic(
        int(payload["obs_dim"]),
        int(payload["num_actions"]),
        int(payload.get("num_uavs", env.action_spec().get("num_uavs", 1))),
        int(cfg_payload.get("hidden_size", hidden_size)),
        hidden_layers,
    ).to(target_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return TwoTimescalePPOPolicy(env, model, cfg)


def load_ppo_policy(
    path: Path,
    env: V19LUTResourceEnv,
    hidden_size: int = 128,
    device: str | Any | None = None,
) -> PPOServicePolicy:
    require_torch()
    target_device = resolve_torch_device(device or "auto")
    payload = torch.load(path, map_location=target_device)
    cfg_payload = payload.get("config") or {}
    cfg = PPOTrainConfig(**{key: value for key, value in cfg_payload.items() if key in PPOTrainConfig.__dataclass_fields__})
    hidden_layers = normalize_hidden_layers(
        payload.get("hidden_layers", cfg_payload.get("hidden_layers")),
        int(cfg_payload.get("hidden_size", hidden_size)),
    )
    cfg = PPOTrainConfig(**{**asdict(cfg), "device": str(target_device), "hidden_layers": hidden_layers})
    model = HybridActorCritic(
        int(payload["obs_dim"]),
        int(payload["num_actions"]),
        int(cfg_payload.get("hidden_size", hidden_size)),
        hidden_layers,
    ).to(target_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return PPOServicePolicy(env, model, cfg)


def _collect_episode(
    env: V19LUTResourceEnv,
    model: HybridActorCritic,
    seed: int,
    cfg: PPOTrainConfig,
    dual: DualState,
) -> dict[str, Any]:
    obs = env.reset(seed=seed, options={"policy_name": "ppo_train"})
    rollout: dict[str, list[Any]] = {
        "observations": [],
        "service_actions": [],
        "service_masks": [],
        "resource_raw_actions": [],
        "old_log_probs": [],
        "values": [],
        "rewards": [],
        "semantic_rewards": [],
        "raw_rewards": [],
        "dones": [],
        "successes": [],
        "quality_costs": [],
        "deadline_costs": [],
        "quality_costs_normal": [],
        "quality_costs_critical": [],
        "deadline_costs_normal": [],
        "deadline_costs_critical": [],
        "conflict_costs": [],
        "battery_costs": [],
        "gpu_costs": [],
        "semantic_accuracy_lcbs": [],
        "semantic_uncertainties": [],
        "semantic_payload_kbs": [],
        "semantic_quality_gaps": [],
        "q_quality": [],
        "q_deadline": [],
        "q_energy": [],
        "q_risk": [],
        "q_utm": [],
        "non_cache_actions": [],
        "bandwidth_shares": [],
        "power_w": [],
        "cpu_shares": [],
        "gpu_shares": [],
    }
    done = False
    device = _model_device(model)
    while not done:
        obs_tensor = torch.as_tensor(obs["vector"], dtype=torch.float32, device=device).unsqueeze(0)
        logits, resource_mean, resource_log_std, value = model(obs_tensor)
        mask = _service_mask_tensor(obs, env.service_levels, logits.device, cfg)
        masked_logits = _apply_service_mask(logits, mask)
        service_dist = Categorical(logits=masked_logits)
        action_idx = service_dist.sample()
        resource_dist = Normal(resource_mean, resource_log_std.exp())
        raw_resource = resource_dist.sample()
        resource_values = torch.sigmoid(raw_resource).squeeze(0).detach().cpu().tolist()
        service_log_prob = service_dist.log_prob(action_idx)
        if cfg.hybrid_actions:
            log_prob = service_log_prob + resource_dist.log_prob(raw_resource).sum(dim=-1)
        else:
            resource_values = _default_resource_values()
            raw_resource = torch.zeros_like(raw_resource)
            log_prob = service_log_prob

        level = env.service_levels[int(action_idx.item())]
        action = _build_hybrid_action(env, obs, level, resource_values, cfg)
        next_obs, raw_reward, done, info = env.step(action)
        semantic_reward = _semantic_controller_reward(obs, info, raw_reward, cfg)
        shaped_reward = semantic_reward - dual.penalty(info) if cfg.constrained else semantic_reward

        risk = str(info.get("risk_level", obs.get("risk_level", "normal")))
        quality_cost = float(bool(info.get("quality_violation", False)))
        deadline_cost = float(bool(info.get("deadline_violation", False)))
        rollout["observations"].append(obs["vector"])
        rollout["service_actions"].append(int(action_idx.item()))
        rollout["service_masks"].append(mask.squeeze(0).detach().cpu().tolist())
        rollout["resource_raw_actions"].append(raw_resource.squeeze(0).detach().cpu().tolist())
        rollout["old_log_probs"].append(float(log_prob.item()))
        rollout["values"].append(float(value.item()))
        rollout["rewards"].append(float(shaped_reward))
        rollout["semantic_rewards"].append(float(semantic_reward))
        rollout["raw_rewards"].append(float(raw_reward))
        rollout["dones"].append(bool(done))
        rollout["successes"].append(float(info["success"]))
        rollout["quality_costs"].append(quality_cost)
        rollout["deadline_costs"].append(deadline_cost)
        rollout["quality_costs_normal"].append(quality_cost if risk != "critical" else 0.0)
        rollout["quality_costs_critical"].append(quality_cost if risk == "critical" else 0.0)
        rollout["deadline_costs_normal"].append(deadline_cost if risk != "critical" else 0.0)
        rollout["deadline_costs_critical"].append(deadline_cost if risk == "critical" else 0.0)
        rollout["conflict_costs"].append(
            float(bool(info.get("airspace_conflict", False)) or bool(info.get("utm_constraint_violation", False)))
        )
        rollout["battery_costs"].append(float(bool(info.get("battery_violation", False))))
        rollout["gpu_costs"].append(float(bool(info.get("resource_violation", False))))
        rollout["semantic_accuracy_lcbs"].append(float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0))))
        rollout["semantic_uncertainties"].append(float(info.get("semantic_uncertainty", 0.0)))
        rollout["semantic_payload_kbs"].append(float(info.get("semantic_payload_kb", info.get("payload_kb", 0.0))))
        rollout["semantic_quality_gaps"].append(float(info.get("semantic_quality_gap", 0.0)))
        rollout["q_quality"].append(float(info.get("q_quality", 0.0)))
        rollout["q_deadline"].append(float(info.get("q_deadline", 0.0)))
        rollout["q_energy"].append(float(info.get("q_energy", 0.0)))
        rollout["q_risk"].append(float(info.get("q_risk", 0.0)))
        rollout["q_utm"].append(float(info.get("q_utm", 0.0)))
        rollout["non_cache_actions"].append(float(int(action["service_level"]) != 0))
        rollout["bandwidth_shares"].append(float(action["bandwidth"]) if float(action["bandwidth"]) <= 1.0 else float(action["bandwidth"]) / max(1.0, env.base_bandwidth_hz))
        rollout["power_w"].append(float(action["power"]))
        rollout["cpu_shares"].append(float(action["cpu_share"]))
        rollout["gpu_shares"].append(float(action["gpu_share"]))
        obs = next_obs

    returns = _discounted_returns([float(x) for x in rollout["rewards"]], [bool(x) for x in rollout["dones"]], cfg.gamma)
    advantages = [float(ret) - float(value) for ret, value in zip(returns, rollout["values"])]
    if len(advantages) > 1:
        avg = _mean(advantages)
        std = _std(advantages, avg)
        advantages = [(value - avg) / (std + 1e-8) for value in advantages]
    rollout["returns"] = returns
    rollout["advantages"] = advantages
    return rollout


def _collect_two_timescale_episode(
    env: V19LUTResourceEnv,
    model: TwoTimescaleMobilitySemanticActorCritic,
    seed: int,
    cfg: PPOTrainConfig,
    dual: DualState,
) -> dict[str, Any]:
    obs = env.reset(seed=seed, options={"policy_name": "two_timescale_ppo_train"})
    rollout: dict[str, list[Any]] = _empty_rollout()
    rollout.update(
        {
            "uav_actions": [],
            "mobility_mode_actions": [],
            "mobility_raw_actions": [],
            "mobility_old_log_probs": [],
            "mobility_reused": [],
            "mobility_energy_j": [],
            "arrival_delay_s": [],
            "coverage_gain": [],
            "utm_conflict_risk": [],
            "serve_task_mode": [],
            "avoid_conflict_mode": [],
        }
    )
    done = False
    cached_mobility: dict[str, Any] | None = None
    cached_action_idx: tuple[int, int] | None = None
    cached_raw = [0.0 for _ in MOBILITY_CONTINUOUS_KEYS]
    device = _model_device(model)
    while not done:
        step = int(obs.get("episode_step", 0))
        obs_tensor = torch.as_tensor(obs["vector"], dtype=torch.float32, device=device).unsqueeze(0)
        (
            uav_logits,
            mode_logits,
            mobility_mean,
            mobility_log_std,
            service_logits,
            resource_mean,
            resource_log_std,
            value,
        ) = model(obs_tensor)
        if cfg.no_mobility_actor:
            mobility_action = _default_mobility_action(env, obs, service_level=None)
            mobility_log_prob = torch.zeros(1, device=device)
            uav_idx = int(mobility_action["uav_assignment"])
            mode_idx = MOBILITY_MODES.index(str(mobility_action["mobility_mode"]))
            raw_mobility = torch.zeros((1, len(MOBILITY_CONTINUOUS_KEYS)), dtype=torch.float32, device=device)
            reused = True
        elif cached_mobility is None or step % max(1, int(cfg.mobility_update_interval)) == 0:
            mobility_action, mobility_log_prob, uav_idx, mode_idx, raw_mobility = _sample_mobility_action(
                env,
                obs,
                cfg,
                uav_logits,
                mode_logits,
                mobility_mean,
                mobility_log_std,
                deterministic=False,
            )
            cached_mobility = mobility_action
            cached_action_idx = (uav_idx, mode_idx)
            cached_raw = raw_mobility.squeeze(0).detach().cpu().tolist()
            reused = False
        else:
            assert cached_mobility is not None and cached_action_idx is not None
            mobility_action = dict(cached_mobility)
            uav_idx, mode_idx = cached_action_idx
            raw_mobility = torch.as_tensor([cached_raw], dtype=torch.float32, device=device)
            mobility_log_prob = _mobility_log_prob(uav_logits, mode_logits, mobility_mean, mobility_log_std, uav_idx, mode_idx, raw_mobility)
            reused = True

        mask = _service_mask_tensor(obs, env.service_levels, service_logits.device, cfg)
        masked_logits = _apply_service_mask(service_logits, mask)
        service_dist = Categorical(logits=masked_logits)
        action_idx = service_dist.sample()
        resource_dist = Normal(resource_mean, resource_log_std.exp())
        raw_resource = resource_dist.sample()
        resource_values = torch.sigmoid(raw_resource).squeeze(0).detach().cpu().tolist()
        service_log_prob = service_dist.log_prob(action_idx)
        log_prob = service_log_prob + resource_dist.log_prob(raw_resource).sum(dim=-1) + mobility_log_prob
        service_level = env.service_levels[int(action_idx.item())]
        action = _build_two_timescale_action(env, obs, service_level, resource_values, mobility_action, cfg)
        next_obs, raw_reward, done, info = env.step(action)
        semantic_reward = _semantic_controller_reward(obs, info, raw_reward, cfg)
        mobility_reward = _mobility_reward_adjustment(info, cfg)
        shaped_reward = semantic_reward + mobility_reward
        shaped_reward = shaped_reward - dual.penalty(info) if cfg.constrained else shaped_reward

        _append_common_rollout(rollout, obs, action_idx, mask, raw_resource, log_prob, value, shaped_reward, semantic_reward, raw_reward, done, info, action)
        rollout["uav_actions"].append(int(uav_idx))
        rollout["mobility_mode_actions"].append(int(mode_idx))
        rollout["mobility_raw_actions"].append(raw_mobility.squeeze(0).detach().cpu().tolist())
        rollout["mobility_old_log_probs"].append(float(mobility_log_prob.item()))
        rollout["mobility_reused"].append(float(reused))
        rollout["mobility_energy_j"].append(float(info.get("mobility_energy_j", 0.0)))
        rollout["arrival_delay_s"].append(float(info.get("arrival_delay_s", 0.0)))
        rollout["coverage_gain"].append(float(info.get("coverage_gain", 0.0)))
        rollout["utm_conflict_risk"].append(float(info.get("utm_conflict_risk", 0.0)))
        rollout["serve_task_mode"].append(float(str(info.get("mobility_mode", "")) == "serve_task"))
        rollout["avoid_conflict_mode"].append(float(str(info.get("mobility_mode", "")) == "avoid_conflict"))
        obs = next_obs

    returns = _discounted_returns([float(x) for x in rollout["rewards"]], [bool(x) for x in rollout["dones"]], cfg.gamma)
    advantages = [float(ret) - float(value) for ret, value in zip(returns, rollout["values"])]
    if len(advantages) > 1:
        avg = _mean(advantages)
        std = _std(advantages, avg)
        advantages = [(value - avg) / (std + 1e-8) for value in advantages]
    rollout["returns"] = returns
    rollout["advantages"] = advantages
    return rollout


def _empty_rollout() -> dict[str, list[Any]]:
    return {
        "observations": [],
        "service_actions": [],
        "service_masks": [],
        "resource_raw_actions": [],
        "old_log_probs": [],
        "values": [],
        "rewards": [],
        "semantic_rewards": [],
        "raw_rewards": [],
        "dones": [],
        "successes": [],
        "quality_costs": [],
        "deadline_costs": [],
        "quality_costs_normal": [],
        "quality_costs_critical": [],
        "deadline_costs_normal": [],
        "deadline_costs_critical": [],
        "conflict_costs": [],
        "battery_costs": [],
        "gpu_costs": [],
        "semantic_accuracy_lcbs": [],
        "semantic_uncertainties": [],
        "semantic_payload_kbs": [],
        "semantic_quality_gaps": [],
        "q_quality": [],
        "q_deadline": [],
        "q_energy": [],
        "q_risk": [],
        "q_utm": [],
        "non_cache_actions": [],
        "bandwidth_shares": [],
        "power_w": [],
        "cpu_shares": [],
        "gpu_shares": [],
    }


def _append_common_rollout(
    rollout: dict[str, list[Any]],
    obs: dict[str, Any],
    action_idx: Any,
    mask: Any,
    raw_resource: Any,
    log_prob: Any,
    value: Any,
    shaped_reward: float,
    semantic_reward: float,
    raw_reward: float,
    done: bool,
    info: dict[str, Any],
    action: dict[str, Any],
) -> None:
    risk = str(info.get("risk_level", obs.get("risk_level", "normal")))
    quality_cost = float(bool(info.get("quality_violation", False)))
    deadline_cost = float(bool(info.get("deadline_violation", False)))
    rollout["observations"].append(obs["vector"])
    rollout["service_actions"].append(int(action_idx.item()))
    rollout["service_masks"].append(mask.squeeze(0).detach().cpu().tolist())
    rollout["resource_raw_actions"].append(raw_resource.squeeze(0).detach().cpu().tolist())
    rollout["old_log_probs"].append(float(log_prob.item()))
    rollout["values"].append(float(value.item()))
    rollout["rewards"].append(float(shaped_reward))
    rollout["semantic_rewards"].append(float(semantic_reward))
    rollout["raw_rewards"].append(float(raw_reward))
    rollout["dones"].append(bool(done))
    rollout["successes"].append(float(info["success"]))
    rollout["quality_costs"].append(quality_cost)
    rollout["deadline_costs"].append(deadline_cost)
    rollout["quality_costs_normal"].append(quality_cost if risk != "critical" else 0.0)
    rollout["quality_costs_critical"].append(quality_cost if risk == "critical" else 0.0)
    rollout["deadline_costs_normal"].append(deadline_cost if risk != "critical" else 0.0)
    rollout["deadline_costs_critical"].append(deadline_cost if risk == "critical" else 0.0)
    rollout["conflict_costs"].append(float(bool(info.get("airspace_conflict", False)) or bool(info.get("utm_constraint_violation", False))))
    rollout["battery_costs"].append(float(bool(info.get("battery_violation", False))))
    rollout["gpu_costs"].append(float(bool(info.get("resource_violation", False))))
    rollout["semantic_accuracy_lcbs"].append(float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0))))
    rollout["semantic_uncertainties"].append(float(info.get("semantic_uncertainty", 0.0)))
    rollout["semantic_payload_kbs"].append(float(info.get("semantic_payload_kb", info.get("payload_kb", 0.0))))
    rollout["semantic_quality_gaps"].append(float(info.get("semantic_quality_gap", 0.0)))
    rollout["q_quality"].append(float(info.get("q_quality", 0.0)))
    rollout["q_deadline"].append(float(info.get("q_deadline", 0.0)))
    rollout["q_energy"].append(float(info.get("q_energy", 0.0)))
    rollout["q_risk"].append(float(info.get("q_risk", 0.0)))
    rollout["q_utm"].append(float(info.get("q_utm", 0.0)))
    rollout["non_cache_actions"].append(float(int(action["service_level"]) != 0))
    rollout["bandwidth_shares"].append(float(action["bandwidth"]) if float(action["bandwidth"]) <= 1.0 else float(action["bandwidth"]) / max(1.0, float(info.get("bandwidth_hz", 1.0))))
    rollout["power_w"].append(float(action["power"]))
    rollout["cpu_shares"].append(float(action["cpu_share"]))
    rollout["gpu_shares"].append(float(action["gpu_share"]))


def _trace_row_from_rollout(
    episode: int,
    rollout: dict[str, Any],
    cfg: PPOTrainConfig,
    dual: DualState,
    demos: list[dict[str, Any]],
    bc_loss: float,
) -> dict[str, float]:
    row = {
        "episode": float(episode),
        "raw_return": float(sum(rollout["raw_rewards"])),
        "semantic_return": float(sum(rollout["semantic_rewards"])),
        "shaped_return": float(sum(rollout["rewards"])),
        "return": float(sum(rollout["raw_rewards"])),
        "success_rate": float(_mean(rollout["successes"])),
        "mean_reward": float(_mean(rollout["raw_rewards"])),
        "mean_semantic_reward": float(_mean(rollout["semantic_rewards"])),
        "mean_shaped_reward": float(_mean(rollout["rewards"])),
        "mean_semantic_accuracy_lcb": float(_mean(rollout["semantic_accuracy_lcbs"])),
        "mean_semantic_uncertainty": float(_mean(rollout["semantic_uncertainties"])),
        "mean_semantic_payload_kb": float(_mean(rollout["semantic_payload_kbs"])),
        "mean_semantic_quality_gap": float(_mean(rollout["semantic_quality_gaps"])),
        "mean_q_quality": float(_mean(rollout["q_quality"])),
        "mean_q_deadline": float(_mean(rollout["q_deadline"])),
        "mean_q_energy": float(_mean(rollout["q_energy"])),
        "mean_q_risk": float(_mean(rollout["q_risk"])),
        "mean_q_utm": float(_mean(rollout["q_utm"])),
        "max_q_quality": float(max(rollout["q_quality"] or [0.0])),
        "max_q_deadline": float(max(rollout["q_deadline"] or [0.0])),
        "max_q_energy": float(max(rollout["q_energy"] or [0.0])),
        "max_q_risk": float(max(rollout["q_risk"] or [0.0])),
        "max_q_utm": float(max(rollout["q_utm"] or [0.0])),
        "non_cache_ratio": float(_mean(rollout["non_cache_actions"])),
        "quality_cost": float(_mean(rollout["quality_costs"])),
        "deadline_cost": float(_mean(rollout["deadline_costs"])),
        "quality_cost_normal": float(_mean(rollout["quality_costs_normal"])),
        "quality_cost_critical": float(_mean(rollout["quality_costs_critical"])),
        "deadline_cost_normal": float(_mean(rollout["deadline_costs_normal"])),
        "deadline_cost_critical": float(_mean(rollout["deadline_costs_critical"])),
        "conflict_cost": float(_mean(rollout["conflict_costs"])),
        "battery_cost": float(_mean(rollout["battery_costs"])),
        "gpu_cost": float(_mean(rollout["gpu_costs"])),
        "bc_loss": float(bc_loss),
        "demo_samples": float(len(demos)),
        "entropy_weight": float(_scheduled_entropy_weight(cfg, episode)),
        "service_prior_weight": float(_scheduled_service_prior_weight(cfg, episode)),
        "mean_bandwidth_share": float(_mean(rollout["bandwidth_shares"])),
        "mean_power_w": float(_mean(rollout["power_w"])),
        "mean_cpu_share": float(_mean(rollout["cpu_shares"])),
        "mean_gpu_share": float(_mean(rollout["gpu_shares"])),
    }
    row.update(dual.to_trace())
    return row


def _ppo_update(
    model: HybridActorCritic,
    optimizer: Any,
    rollout: dict[str, Any],
    cfg: PPOTrainConfig,
    demos: list[dict[str, Any]],
    episode: int,
) -> None:
    device = _model_device(model)
    obs = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=device)
    service_actions = torch.as_tensor(rollout["service_actions"], dtype=torch.int64, device=device)
    service_masks = torch.as_tensor(rollout["service_masks"], dtype=torch.bool, device=device)
    resource_raw_actions = torch.as_tensor(rollout["resource_raw_actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    for _ in range(cfg.update_epochs):
        logits, resource_mean, resource_log_std, values = model(obs)
        logits = _apply_service_mask(logits, service_masks)
        service_dist = Categorical(logits=logits)
        log_probs = service_dist.log_prob(service_actions)
        entropy = service_dist.entropy()
        if cfg.hybrid_actions:
            resource_dist = Normal(resource_mean, resource_log_std.exp())
            log_probs = log_probs + resource_dist.log_prob(resource_raw_actions).sum(dim=-1)
            entropy = entropy + resource_dist.entropy().sum(dim=-1)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clipped).mean()
        value_loss = (returns - values).pow(2).mean()
        entropy_weight = _scheduled_entropy_weight(cfg, episode)
        loss = policy_loss + cfg.value_weight * value_loss - entropy_weight * entropy.mean()
        prior_weight = float(cfg.bc_aux_weight) + _scheduled_service_prior_weight(cfg, episode)
        if demos and prior_weight > 0.0:
            loss = loss + prior_weight * _bc_loss(model, demos, cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sanitize_model_parameters(model)


def _two_timescale_ppo_update(
    model: TwoTimescaleMobilitySemanticActorCritic,
    optimizer: Any,
    rollout: dict[str, Any],
    cfg: PPOTrainConfig,
    demos: list[dict[str, Any]],
    episode: int,
) -> None:
    device = _model_device(model)
    obs = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=device)
    service_actions = torch.as_tensor(rollout["service_actions"], dtype=torch.int64, device=device)
    service_masks = torch.as_tensor(rollout["service_masks"], dtype=torch.bool, device=device)
    resource_raw_actions = torch.as_tensor(rollout["resource_raw_actions"], dtype=torch.float32, device=device)
    uav_actions = torch.as_tensor(rollout["uav_actions"], dtype=torch.int64, device=device)
    mode_actions = torch.as_tensor(rollout["mobility_mode_actions"], dtype=torch.int64, device=device)
    mobility_raw_actions = torch.as_tensor(rollout["mobility_raw_actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    for _ in range(cfg.update_epochs):
        (
            uav_logits,
            mode_logits,
            mobility_mean,
            mobility_log_std,
            service_logits,
            resource_mean,
            resource_log_std,
            values,
        ) = model(obs)
        service_logits = _apply_service_mask(service_logits, service_masks)
        service_dist = Categorical(logits=service_logits)
        resource_dist = Normal(resource_mean, resource_log_std.exp())
        log_probs = service_dist.log_prob(service_actions) + resource_dist.log_prob(resource_raw_actions).sum(dim=-1)
        entropy = service_dist.entropy() + resource_dist.entropy().sum(dim=-1)
        if not cfg.no_mobility_actor:
            uav_dist = Categorical(logits=uav_logits)
            mode_dist = Categorical(logits=mode_logits)
            mobility_dist = Normal(mobility_mean, mobility_log_std.exp())
            log_probs = (
                log_probs
                + uav_dist.log_prob(uav_actions)
                + mode_dist.log_prob(mode_actions)
                + mobility_dist.log_prob(mobility_raw_actions).sum(dim=-1)
            )
            entropy = entropy + uav_dist.entropy() + mode_dist.entropy() + mobility_dist.entropy().sum(dim=-1)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clipped).mean()
        value_loss = (returns - values).pow(2).mean()
        loss = policy_loss + cfg.value_weight * value_loss - _scheduled_entropy_weight(cfg, episode) * entropy.mean()
        prior_weight = float(cfg.bc_aux_weight) + _scheduled_service_prior_weight(cfg, episode)
        if demos and prior_weight > 0.0:
            loss = loss + prior_weight * _bc_loss_two_timescale(model, demos, cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sanitize_model_parameters(model)


def _build_two_timescale_action(
    env: V19LUTResourceEnv,
    obs: dict[str, Any],
    service_level: int,
    resources: list[float] | tuple[float, ...],
    mobility: dict[str, Any],
    cfg: PPOTrainConfig,
) -> dict[str, Any]:
    action = _build_hybrid_action(env, obs, service_level, resources, cfg)
    action.update(
        {
            "uav_assignment": int(mobility.get("uav_assignment", action.get("uav_assignment", 0))),
            "mobility_mode": str(mobility.get("mobility_mode", action.get("mobility_mode", "stay"))),
            "waypoint_delta": list(mobility.get("waypoint_delta", action.get("waypoint_delta", [0.0, 0.0]))),
            "altitude_delta": float(mobility.get("altitude_delta", action.get("altitude_delta", 0.0))),
        }
    )
    if int(service_level) == 0 and action["mobility_mode"] == "serve_task":
        action["mobility_mode"] = "stay"
    return env.parse_action(action)


def _sample_mobility_action(
    env: V19LUTResourceEnv,
    obs: dict[str, Any],
    cfg: PPOTrainConfig,
    uav_logits: Any,
    mode_logits: Any,
    mobility_mean: Any,
    mobility_log_std: Any,
    deterministic: bool,
) -> tuple[dict[str, Any], Any, int, int, Any]:
    uav_logits = _apply_uav_mask(uav_logits, obs)
    mode_logits = _apply_mobility_mode_mask(mode_logits, obs)
    uav_dist = Categorical(logits=uav_logits)
    mode_dist = Categorical(logits=mode_logits)
    mobility_dist = Normal(mobility_mean, mobility_log_std.exp())
    if deterministic:
        uav_action = torch.argmax(uav_logits, dim=-1)
        mode_action = torch.argmax(mode_logits, dim=-1)
        raw = mobility_mean
    else:
        uav_action = uav_dist.sample()
        mode_action = mode_dist.sample()
        raw = mobility_dist.sample()
    uav_idx = int(uav_action.item())
    mode_idx = int(mode_action.item())
    log_prob = uav_dist.log_prob(uav_action) + mode_dist.log_prob(mode_action) + mobility_dist.log_prob(raw).sum(dim=-1)
    mobility = _mobility_action_from_raw(env, obs, cfg, uav_idx, mode_idx, raw.squeeze(0).detach().cpu().tolist())
    return mobility, log_prob, uav_idx, mode_idx, raw


def _mobility_log_prob(
    uav_logits: Any,
    mode_logits: Any,
    mobility_mean: Any,
    mobility_log_std: Any,
    uav_idx: int,
    mode_idx: int,
    raw: Any,
) -> Any:
    uav_action = torch.as_tensor([int(uav_idx)], dtype=torch.int64, device=uav_logits.device)
    mode_action = torch.as_tensor([int(mode_idx)], dtype=torch.int64, device=mode_logits.device)
    raw = raw.to(mobility_mean.device) if hasattr(raw, "to") else torch.as_tensor(raw, dtype=torch.float32, device=mobility_mean.device)
    return (
        Categorical(logits=uav_logits).log_prob(uav_action)
        + Categorical(logits=mode_logits).log_prob(mode_action)
        + Normal(mobility_mean, mobility_log_std.exp()).log_prob(raw).sum(dim=-1)
    )


def _mobility_action_from_raw(
    env: V19LUTResourceEnv,
    obs: dict[str, Any],
    cfg: PPOTrainConfig,
    uav_idx: int,
    mode_idx: int,
    raw_values: list[float],
) -> dict[str, Any]:
    feasible_uavs = _feasible_uav_ids(obs)
    uav_assignment = feasible_uavs[int(uav_idx) % len(feasible_uavs)] if feasible_uavs else int(uav_idx)
    mode = MOBILITY_MODES[int(mode_idx) % len(MOBILITY_MODES)]
    dx_unit = math.tanh(float(raw_values[0])) if len(raw_values) > 0 else 0.0
    dy_unit = math.tanh(float(raw_values[1])) if len(raw_values) > 1 else 0.0
    dz_unit = math.tanh(float(raw_values[2])) if len(raw_values) > 2 else 0.0
    action = {
        "uav_assignment": int(uav_assignment),
        "mobility_mode": mode,
        "waypoint_delta": [dx_unit * float(cfg.waypoint_delta_max_m), dy_unit * float(cfg.waypoint_delta_max_m)],
        "altitude_delta": dz_unit * float(cfg.altitude_delta_max_m),
    }
    return _project_mobility_action(env, obs, action, cfg)


def _default_mobility_action(env: V19LUTResourceEnv, obs: dict[str, Any], service_level: int | None) -> dict[str, Any]:
    level = int(service_level if service_level is not None else 1)
    mode = "stay" if level == 0 else "serve_task"
    return {"uav_assignment": _select_uav_assignment(obs, PPOTrainConfig()), "mobility_mode": mode, "waypoint_delta": [0.0, 0.0], "altitude_delta": 0.0}


def _project_mobility_action(env: V19LUTResourceEnv, obs: dict[str, Any], action: dict[str, Any], cfg: PPOTrainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PPOTrainConfig()
    mask = obs.get("action_mask", {}) if isinstance(obs.get("action_mask", {}), dict) else {}
    allowed = mask.get("mobility_mode_allowed", {})
    mode = str(action.get("mobility_mode", "serve_task"))
    if allowed and not bool(allowed.get(mode, allowed.get(str(mode), True))):
        mode = "serve_task" if bool(allowed.get("serve_task", True)) else "stay"
    if str(obs.get("operational_intent_state", "accepted")) in {"nonconforming", "contingent"}:
        mode = "avoid_conflict"
    if _prefer_nearest_uav_mobility(obs, cfg):
        if mode in {"reposition", "return_base"}:
            mode = "serve_task" if bool(allowed.get("serve_task", True)) else "stay"
        action["uav_assignment"] = _select_uav_assignment(obs, cfg)
        action["waypoint_delta"] = [0.0, 0.0]
        action["altitude_delta"] = 0.0
    action["mobility_mode"] = mode
    action["uav_assignment"] = _select_uav_assignment(obs, cfg) if action.get("uav_assignment") is None else int(action["uav_assignment"])
    return action


def _prefer_nearest_uav_mobility(obs: dict[str, Any], cfg: PPOTrainConfig) -> bool:
    if bool(cfg.nearest_uav_mobility):
        return True
    if not bool(cfg.critical_burst_nearest_uav):
        return False
    scenario = str(obs.get("scenario", obs.get("formal_scenario", "")))
    risk = str(obs.get("risk_level", "normal"))
    queue_len = len(list(obs.get("task_queue", []) or []))
    return scenario == "disaster_hotspot" and (risk == "critical" or queue_len >= 2)


def _apply_uav_mask(logits: Any, obs: dict[str, Any]) -> Any:
    feasible = set(_feasible_uav_ids(obs))
    if not feasible:
        return logits
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for uid in feasible:
        if 0 <= int(uid) < mask.shape[-1]:
            mask[..., int(uid)] = True
    if not bool(mask.any()):
        return logits
    return logits.masked_fill(~mask, -1.0e9)


def _apply_mobility_mode_mask(logits: Any, obs: dict[str, Any]) -> Any:
    mask_data = obs.get("action_mask", {}).get("mobility_mode_allowed", {}) if isinstance(obs.get("action_mask", {}), dict) else {}
    if not mask_data:
        return logits
    mask_values = [bool(mask_data.get(mode, True)) for mode in MOBILITY_MODES]
    if not any(mask_values):
        return logits
    mask = torch.as_tensor(mask_values, dtype=torch.bool, device=logits.device).unsqueeze(0)
    return logits.masked_fill(~mask, -1.0e9)


def _feasible_uav_ids(obs: dict[str, Any]) -> list[int]:
    mask = obs.get("action_mask", {}).get("uav_battery_ok", {}) if isinstance(obs.get("action_mask", {}), dict) else {}
    feasible = [int(uid) for uid, ok in mask.items() if bool(ok)]
    if feasible:
        return feasible
    if obs.get("feasible_uavs"):
        return [int(uid) for uid in obs.get("feasible_uavs", [])]
    uavs = list(obs.get("uav_state", []) or [])
    return [int(uav.get("uav_id", idx)) for idx, uav in enumerate(uavs)] or [0]


def _mobility_reward_adjustment(info: dict[str, Any], cfg: PPOTrainConfig) -> float:
    flight_energy = float(info.get("mobility_energy_j", 0.0))
    arrival_delay = float(info.get("arrival_delay_s", 0.0))
    coverage_gain = float(info.get("coverage_gain", 0.0))
    edge_queue = float(info.get("edge_queue_delay_s", info.get("edge_queue_s", 0.0)))
    return float(
        cfg.coverage_gain_weight * coverage_gain
        - cfg.flight_energy_cost_weight * flight_energy / max(1.0, float(cfg.energy_budget_j))
        - cfg.arrival_delay_cost_weight * arrival_delay / 10.0
        - cfg.edge_queue_cost_weight * edge_queue
    )


def _build_hybrid_action(
    env: V19LUTResourceEnv,
    obs: dict[str, Any],
    service_level: int,
    resources: list[float] | tuple[float, ...],
    cfg: PPOTrainConfig,
) -> dict[str, Any]:
    action = env.candidate_action(service_level, obs)
    if int(service_level) == 0:
        action.update({"bandwidth": 0.0, "power": 0.0, "cpu_share": 0.0, "gpu_share": 0.0})
    elif cfg.hybrid_actions:
        bandwidth_floor = _resource_floor(cfg, service_level, "bandwidth")
        power_floor = _resource_floor(cfg, service_level, "power")
        cpu_floor = _resource_floor(cfg, service_level, "cpu_share")
        gpu_floor = _resource_floor(cfg, service_level, "gpu_share")
        action.update(
            {
                "bandwidth": _scaled_unit(resources[0], bandwidth_floor, 1.0),
                "power": _scaled_unit(resources[1], max(cfg.power_min_w, power_floor), cfg.power_max_w),
                "cpu_share": _scaled_unit(resources[2], cpu_floor, 1.0),
                "gpu_share": _scaled_unit(resources[3], gpu_floor, 1.0),
            }
        )
    action["service_level"] = int(service_level)
    action["sensing_decision"] = "reuse_cache" if int(service_level) == 0 else "observe"
    action["uav_assignment"] = _select_uav_assignment(obs, cfg)
    action.setdefault("waypoint", None)
    parsed = env.parse_action(action)
    return _project_safe_action(env, obs, parsed, cfg) if cfg.safety_layer else parsed


def _project_safe_action(env: V19LUTResourceEnv, obs: dict[str, Any], action: dict[str, Any], cfg: PPOTrainConfig) -> dict[str, Any]:
    info = env.evaluate_action(action, obs)
    if not _has_hard_safety_violation(info):
        if cfg.semantic_projection or cfg.resource_projection:
            return _project_semantic_feasible_action(env, obs, action, cfg, current_info=info)
        return env.parse_action(action)
    current = int(action["service_level"])
    ordered_levels = [current] + [level for level in env.service_levels if level != current]
    best_candidate: dict[str, Any] | None = None
    best_score = float("inf")
    for level in ordered_levels:
        if not _service_allowed(obs, level):
            continue
        candidate = _resource_floor_candidate(env, obs, level, cfg)
        candidate["uav_assignment"] = _select_uav_assignment(obs, cfg)
        evaluated = env.evaluate_action(candidate, obs)
        if not _has_hard_safety_violation(evaluated):
            score = _semantic_safety_score(evaluated, obs, cfg)
            if score < best_score:
                best_score = score
                best_candidate = candidate
    if best_candidate is not None:
        return env.parse_action(best_candidate)
    fallback = env.candidate_action(env.service_levels[0], obs)
    fallback["uav_assignment"] = _select_uav_assignment(obs, cfg)
    return env.parse_action(fallback)


def _project_semantic_feasible_action(
    env: V19LUTResourceEnv,
    obs: dict[str, Any],
    action: dict[str, Any],
    cfg: PPOTrainConfig,
    current_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = env.parse_action(action)
    current_info = current_info or env.evaluate_action(current, obs)
    best_action = current
    best_score = _semantic_safety_score(current_info, obs, cfg)
    cache_override = _needs_cache_override(current, current_info, obs, cfg)
    best_gap = _semantic_gap(current_info, obs)
    best_projection_score = _semantic_projection_score(current_info, obs, cfg)
    if best_score <= 0.0:
        return best_action
    for level in env.service_levels:
        if not _service_allowed(obs, level):
            continue
        candidate = _resource_floor_candidate(env, obs, level, cfg)
        candidate["uav_assignment"] = _select_uav_assignment(obs, cfg)
        evaluated = env.evaluate_action(candidate, obs)
        if _has_hard_safety_violation(evaluated):
            continue
        if cache_override and int(level) > 0:
            candidate_gap = _semantic_gap(evaluated, obs)
            improves_gap = (best_gap - candidate_gap) >= float(cfg.cache_override_min_improvement)
            semantic_success = bool(evaluated.get("semantic_success", evaluated.get("success", False))) and not _evidence_guard_blocks(evaluated, obs, cfg)
            projection_score = _semantic_projection_score(evaluated, obs, cfg)
            if improves_gap or semantic_success or projection_score < best_projection_score:
                best_gap = candidate_gap
                best_action = candidate
                best_projection_score = projection_score
                best_score = _semantic_safety_score(evaluated, obs, cfg)
                continue
        score = _semantic_safety_score(evaluated, obs, cfg)
        if score < best_score:
            best_score = score
            best_action = candidate
    return env.parse_action(best_action)


def _resource_floor_candidate(env: V19LUTResourceEnv, obs: dict[str, Any], service_level: int, cfg: PPOTrainConfig) -> dict[str, Any]:
    candidate = env.candidate_action(service_level, obs)
    if int(service_level) == 0:
        candidate.update({"bandwidth": 0.0, "power": 0.0, "cpu_share": 0.0, "gpu_share": 0.0})
    elif cfg.hybrid_actions:
        candidate["bandwidth"] = max(float(candidate.get("bandwidth", 0.0)), _resource_floor(cfg, service_level, "bandwidth"))
        candidate["power"] = max(float(candidate.get("power", cfg.power_min_w)), _resource_floor(cfg, service_level, "power"), cfg.power_min_w)
        candidate["cpu_share"] = max(float(candidate.get("cpu_share", cfg.share_floor)), _resource_floor(cfg, service_level, "cpu_share"))
        candidate["gpu_share"] = max(float(candidate.get("gpu_share", cfg.share_floor)), _resource_floor(cfg, service_level, "gpu_share"))
    candidate["service_level"] = int(service_level)
    candidate["sensing_decision"] = "reuse_cache" if int(service_level) == 0 else "observe"
    return candidate


def _semantic_safety_score(info: dict[str, Any], obs: dict[str, Any], cfg: PPOTrainConfig) -> float:
    risk = str(info.get("risk_level", obs.get("risk_level", "normal")))
    risk_weight = 1.6 if risk == "critical" else 1.0
    epsilon = float(obs.get("epsilon_k", 0.0))
    accuracy = float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0)))
    deadline = max(1e-6, float(obs.get("deadline_s", obs.get("tau_k", 5.0))))
    delay = float(info.get("delay_s", info.get("total_delay_s", 0.0)))
    score = 0.0
    score += 100.0 * float(_has_hard_safety_violation(info))
    score += 24.0 * risk_weight * float(bool(info.get("quality_violation", False)))
    score += 16.0 * risk_weight * float(bool(info.get("deadline_violation", False)))
    score += 8.0 * float(bool(info.get("utm_constraint_violation", False) or info.get("utm_conflict_violation", False)))
    score += 4.0 * float(not bool(info.get("dss_available", True)))
    score += cfg.off_nominal_cost_weight * float(info.get("off_nominal_planning_penalty", 0.0))
    score += 10.0 * risk_weight * max(0.0, epsilon - accuracy)
    if _evidence_guard_blocks(info, obs, cfg):
        score += 64.0 * risk_weight
    if int(info.get("service_level", 0)) == 0:
        cache_gap = _semantic_gap(info, obs)
        if cache_gap >= float(cfg.cache_override_gap_threshold):
            score += 36.0 * _cache_risk_multiplier(info, obs, cfg) * cache_gap
    elif int(info.get("service_level", 0)) == 1 and _semantic_gap(info, obs) <= float(cfg.token_near_success_gap):
        score -= float(cfg.token_projection_bonus)
    score += 4.0 * max(0.0, delay / deadline - 1.0)
    score += 0.25 * float(info.get("semantic_uncertainty", 0.0))
    score += 0.02 * float(info.get("payload_kb", 0.0)) / 100.0
    score += 0.01 * float(info.get("energy_j", 0.0)) / 100.0
    return float(score)


def _needs_cache_override(action: dict[str, Any], info: dict[str, Any], obs: dict[str, Any], cfg: PPOTrainConfig) -> bool:
    if int(action.get("service_level", info.get("service_level", 0))) != 0:
        return False
    gap = _semantic_gap(info, obs)
    epsilon = float(obs.get("epsilon_k", 0.0))
    risk = str(info.get("risk_level", obs.get("risk_level", "normal")))
    freshness = str(info.get("freshness_bin", obs.get("freshness_bin", "fresh")))
    state = str(info.get("operational_intent_state", obs.get("operational_intent_state", "accepted")))
    high_priority = (
        risk == "critical"
        or epsilon >= float(cfg.high_epsilon_cache_threshold)
        or freshness in {"stale", "expired"}
        or state in {"nonconforming", "contingent"}
        or bool(info.get("utm_constraint_violation", False) or info.get("utm_conflict_violation", False))
    )
    return gap >= float(cfg.cache_override_gap_threshold) or (high_priority and gap > 0.0)


def _semantic_gap(info: dict[str, Any], obs: dict[str, Any]) -> float:
    epsilon = float(obs.get("epsilon_k", 0.0))
    accuracy = float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0)))
    return max(0.0, epsilon - accuracy)


def _evidence_guard_blocks(info: dict[str, Any], obs: dict[str, Any], cfg: PPOTrainConfig) -> bool:
    return _deadline_guard_blocks(info, obs, cfg) or _payload_delay_guard_blocks(info, obs, cfg)


def _deadline_guard_blocks(info: dict[str, Any], obs: dict[str, Any], cfg: PPOTrainConfig) -> bool:
    if not bool(cfg.deadline_aware_evidence_guard):
        return False
    service_level = int(info.get("service_level", 0))
    if service_level <= 0:
        return False
    deadline = max(1e-6, float(obs.get("deadline_s", obs.get("tau_k", info.get("deadline_s", 5.0)))))
    evidence_delay = _evidence_delay_s(info)
    return evidence_delay > max(1e-6, float(cfg.deadline_guard_slack) * deadline)


def _payload_delay_guard_blocks(info: dict[str, Any], obs: dict[str, Any], cfg: PPOTrainConfig) -> bool:
    service_level = int(info.get("service_level", 0))
    if service_level != 2:
        return False
    if bool(cfg.no_image_under_low_snr) and _is_low_snr(obs, cfg):
        return True
    if not bool(cfg.payload_delay_aware_projection):
        return False
    payload_kb = float(info.get("semantic_payload_kb", info.get("payload_kb", 0.0)))
    deadline = float(obs.get("deadline_s", obs.get("tau_k", info.get("deadline_s", 5.0))))
    return (
        _is_low_snr(obs, cfg)
        and payload_kb >= float(cfg.high_payload_guard_kb)
        and deadline <= float(cfg.short_deadline_guard_s)
    )


def _evidence_delay_s(info: dict[str, Any]) -> float:
    tx = float(info.get("tx_delay_s", info.get("transmission_delay_s", 0.0)))
    queue = float(info.get("queue_delay_s", info.get("edge_queue_delay_s", info.get("edge_queue_s", 0.0))))
    infer = float(info.get("infer_delay_s", info.get("inference_delay_s", 0.0)))
    total = tx + queue + infer
    if total <= 0.0:
        total = float(info.get("delay_s", info.get("total_delay_s", 0.0)))
    return float(total)


def _is_low_snr(obs: dict[str, Any], cfg: PPOTrainConfig) -> bool:
    snr_value = obs.get("sensed_snr_db", obs.get("snr_db", None))
    try:
        if snr_value is not None and float(snr_value) <= float(cfg.low_snr_guard_threshold_db):
            return True
    except (TypeError, ValueError):
        pass
    snr_bin = str(obs.get("snr_bin", "")).lower()
    scenario = str(obs.get("scenario", obs.get("formal_scenario", ""))).lower()
    return "low" in snr_bin or "blockage" in scenario or "low_snr" in scenario


def _cache_risk_multiplier(info: dict[str, Any], obs: dict[str, Any], cfg: PPOTrainConfig) -> float:
    risk = str(info.get("risk_level", obs.get("risk_level", "normal")))
    freshness = str(info.get("freshness_bin", obs.get("freshness_bin", "fresh")))
    state = str(info.get("operational_intent_state", obs.get("operational_intent_state", "accepted")))
    epsilon = float(obs.get("epsilon_k", info.get("epsilon_k", 0.0)))
    multiplier = 1.0
    if risk == "critical":
        multiplier += 1.2
    if epsilon >= float(cfg.high_epsilon_cache_threshold):
        multiplier += 0.8 + max(0.0, epsilon - float(cfg.high_epsilon_cache_threshold))
    if freshness == "stale":
        multiplier += 0.4
    elif freshness == "expired":
        multiplier += 0.9
    if state in {"nonconforming", "contingent"}:
        multiplier += 0.8
    if bool(info.get("utm_constraint_violation", False) or info.get("utm_conflict_violation", False)):
        multiplier += 0.8
    return float(multiplier)


def _semantic_projection_score(info: dict[str, Any], obs: dict[str, Any], cfg: PPOTrainConfig) -> float:
    service_level = int(info.get("service_level", 0))
    deadline = max(1e-6, float(obs.get("deadline_s", obs.get("tau_k", info.get("deadline_s", 5.0)))))
    delay = float(info.get("delay_s", info.get("total_delay_s", 0.0)))
    gap = _semantic_gap(info, obs)
    deadline_overrun = max(0.0, delay / deadline - 1.0)
    score = 90.0 * gap + 14.0 * deadline_overrun
    score += 8.0 * float(bool(info.get("utm_constraint_violation", False) or info.get("utm_conflict_violation", False)))
    score += 4.0 * float(info.get("off_nominal_planning_penalty", 0.0))
    score += 0.02 * float(info.get("energy_j", 0.0)) / 10.0
    score += 0.01 * float(info.get("payload_kb", 0.0))
    if _deadline_guard_blocks(info, obs, cfg):
        deadline = max(1e-6, float(obs.get("deadline_s", obs.get("tau_k", info.get("deadline_s", 5.0)))))
        score += 80.0 * max(1.0, _evidence_delay_s(info) / deadline)
    if _payload_delay_guard_blocks(info, obs, cfg):
        score += 120.0 if bool(cfg.no_image_under_low_snr) else 48.0
    if service_level == 0 and gap > 0.0:
        score += 25.0 * _cache_risk_multiplier(info, obs, cfg) * gap
    if service_level == 1:
        score -= float(cfg.token_projection_bonus)
        if gap <= float(cfg.token_near_success_gap):
            score -= float(cfg.token_projection_bonus)
    if service_level == 2 and deadline_overrun > 0.0:
        score += 4.0
    return float(score)


def _has_hard_safety_violation(info: dict[str, Any]) -> bool:
    return bool(info.get("battery_violation", False) or info.get("resource_violation", False))


def _select_uav_assignment(obs: dict[str, Any], cfg: PPOTrainConfig) -> int:
    uavs = list(obs.get("uav_state", []) or [])
    if cfg.safety_layer:
        mask = obs.get("action_mask", {}).get("uav_battery_ok", {})
        feasible = [int(uid) for uid, ok in mask.items() if bool(ok)]
        if not feasible:
            feasible = [int(item) for item in obs.get("feasible_uavs", [])]
    else:
        feasible = [int(uav.get("uav_id", idx)) for idx, uav in enumerate(uavs)]
    if not feasible:
        feasible = [int(uav.get("uav_id", idx)) for idx, uav in enumerate(uavs)] or [0]
    task_xy = _front_task_xy(obs)
    if task_xy is None:
        return int(feasible[0])
    best = int(feasible[0])
    best_dist = float("inf")
    for idx, uav in enumerate(uavs):
        uid = int(uav.get("uav_id", idx))
        if uid not in feasible:
            continue
        dist = (float(uav.get("x_m", 0.0)) - task_xy[0]) ** 2 + (float(uav.get("y_m", 0.0)) - task_xy[1]) ** 2
        if dist < best_dist:
            best = uid
            best_dist = dist
    return best


def _front_task_xy(obs: dict[str, Any]) -> tuple[float, float] | None:
    task_id = str(obs.get("task_id", ""))
    queue = list(obs.get("task_queue", []) or [])
    candidates = [item for item in queue if str(item.get("task_id", "")) == task_id] or queue[:1]
    if not candidates:
        return None
    area = candidates[0].get("area4d", {})
    if not isinstance(area, dict):
        return None
    return float(area.get("center_x_m", 0.0)), float(area.get("center_y_m", 0.0))


def _service_mask_tensor(obs: dict[str, Any], service_levels: list[int], device: Any, cfg: PPOTrainConfig) -> Any:
    if not cfg.safety_layer:
        return torch.ones((1, len(service_levels)), dtype=torch.bool, device=device)
    mask = [_service_allowed(obs, level) for level in service_levels]
    if not any(mask):
        mask = [True for _ in service_levels]
    return torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)


def _service_allowed(obs: dict[str, Any], level: int) -> bool:
    mask_data = obs.get("action_mask", {}).get("service_level_allowed", {})
    return bool(mask_data.get(level, mask_data.get(str(level), True)))


def _apply_service_mask(logits: Any, mask: Any) -> Any:
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and logits.shape[0] != 1:
        mask = mask.expand_as(logits)
    return logits.masked_fill(~mask, -1.0e9)


def _semantic_controller_reward(obs: dict[str, Any], info: dict[str, Any], raw_reward: float, cfg: PPOTrainConfig) -> float:
    if cfg.semantic_reward_mode == "env":
        return float(raw_reward)
    risk_weight = 1.6 if str(info.get("risk_level", obs.get("risk_level", "normal"))) == "critical" else 1.0
    accuracy = _control_accuracy(info, cfg)
    accuracy_mean = float(info.get("semantic_accuracy_mean", accuracy))
    uncertainty = float(info.get("semantic_uncertainty", 0.0))
    epsilon = float(obs.get("epsilon_k", 0.0))
    semantic_success = float(bool(info.get("semantic_success", info.get("success", False))))
    task_success = float(bool(info.get("success", False)))
    service_level = int(info.get("service_level", 0))
    delay_norm = float(info.get("delay_s", 0.0)) / max(0.5, float(obs.get("deadline_s", obs.get("tau_k", 5.0))))
    energy_norm = float(info.get("energy_j", 0.0)) / max(1.0, float(info.get("energy_budget_j", cfg.energy_budget_j)))
    payload_norm = float(info.get("payload_kb", 0.0)) / 200.0
    flight_energy_norm = float(info.get("mobility_energy_j", 0.0)) / max(1.0, float(info.get("energy_budget_j", cfg.energy_budget_j)))
    arrival_delay_norm = float(info.get("arrival_delay_s", 0.0)) / max(0.5, float(obs.get("deadline_s", obs.get("tau_k", 5.0))))
    edge_queue_norm = float(info.get("edge_queue_delay_s", info.get("edge_queue_s", 0.0))) / max(0.5, float(obs.get("deadline_s", obs.get("tau_k", 5.0))))
    coverage_gain = float(info.get("coverage_gain", 0.0))
    dss_delay_norm = (
        float(info.get("dss_delay_s", 0.0)) + float(info.get("subscription_notification_delay_s", 0.0))
    ) / max(0.5, float(obs.get("deadline_s", obs.get("tau_k", 5.0))))
    safety_cost = (
        cfg.conflict_cost_weight * float(bool(info.get("airspace_conflict", False)))
        + cfg.battery_cost_weight * float(bool(info.get("battery_violation", False)))
        + cfg.gpu_cost_weight * float(bool(info.get("resource_violation", False)))
        + cfg.utm_conflict_cost_weight * float(bool(info.get("utm_constraint_violation", False)))
        + cfg.dss_delay_cost_weight * dss_delay_norm
        + cfg.off_nominal_cost_weight * float(info.get("off_nominal_planning_penalty", 0.0))
    )
    mobility_cost = (
        cfg.flight_energy_cost_weight * flight_energy_norm
        + cfg.arrival_delay_cost_weight * arrival_delay_norm
        + cfg.edge_queue_cost_weight * edge_queue_norm
        - cfg.coverage_gain_weight * coverage_gain
    )
    resource_cost = (
        cfg.delay_cost_weight * delay_norm
        + cfg.energy_cost_weight * energy_norm
        + cfg.payload_cost_weight * payload_norm
        + mobility_cost
        + safety_cost
    )
    if cfg.semantic_reward_mode == "no_semantic_utility":
        snr_scaled = _snr_scaled(obs)
        return float(snr_scaled - resource_cost)
    margin = accuracy - epsilon
    if cfg.semantic_reward_mode == "accuracy_only":
        return float(risk_weight * accuracy_mean - 0.1 * resource_cost)
    semantic_gap = max(0.0, -margin)
    utility = (
        cfg.semantic_success_weight * risk_weight * semantic_success
        + cfg.semantic_accuracy_weight * risk_weight * accuracy
        + cfg.semantic_margin_weight * risk_weight * max(0.0, margin)
        - cfg.semantic_gap_weight * risk_weight * semantic_gap
        + 0.5 * cfg.semantic_success_weight * risk_weight * task_success
        - resource_cost
    )
    if service_level == 0 and semantic_gap > 0.0:
        cache_multiplier = _cache_risk_multiplier(info, obs, cfg)
        utility -= cfg.cache_shortfall_penalty_weight * cache_multiplier * (1.0 + epsilon) * semantic_gap
        if risk_weight > 1.0 or epsilon >= float(cfg.high_epsilon_cache_threshold):
            utility -= cfg.high_risk_cache_penalty_weight * cache_multiplier
        if str(info.get("freshness_bin", obs.get("freshness_bin", "fresh"))) in {"stale", "expired"}:
            utility -= cfg.cache_stale_penalty_weight * cache_multiplier * semantic_gap
        if str(info.get("operational_intent_state", obs.get("operational_intent_state", "accepted"))) in {"nonconforming", "contingent"}:
            utility -= cfg.cache_utm_penalty_weight * cache_multiplier * semantic_gap
    elif service_level > 0 and semantic_success > 0.0:
        utility += cfg.non_cache_semantic_bonus * risk_weight
    if service_level == 1 and semantic_gap > 0.0:
        utility += cfg.semantic_token_exploration_bonus * risk_weight
    if cfg.semantic_reward_mode == "uncertainty_aware":
        utility += 0.5 * risk_weight * max(0.0, margin) - 0.4 * uncertainty
    if cfg.lyapunov_reward and cfg.use_lyapunov_queues:
        queue_penalty = (
            cfg.queue_quality_weight * float(info.get("q_quality", 0.0)) * max(0.0, -margin)
            + cfg.queue_deadline_weight * float(info.get("q_deadline", 0.0)) * max(0.0, delay_norm - 1.0)
            + cfg.queue_energy_weight * float(info.get("q_energy", 0.0)) / max(1.0, float(cfg.energy_budget_j))
            + cfg.queue_risk_weight * float(info.get("q_risk", 0.0))
            + cfg.queue_utm_weight * float(info.get("q_utm", 0.0))
        )
        utility -= queue_penalty
    if cfg.lyapunov_reward:
        utility -= cfg.uncertainty_cost_weight * risk_weight * uncertainty
    return float(utility)


def _control_accuracy(info: dict[str, Any], cfg: PPOTrainConfig) -> float:
    if cfg.use_semantic_lcb:
        return float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0)))
    return float(
        info.get(
            "answer_accuracy_raw",
            info.get("semantic_accuracy_mean", info.get("answer_accuracy_est", 0.0)),
        )
    )


def _snr_scaled(obs: dict[str, Any]) -> float:
    vector = obs.get("vector", [])
    return float(vector[11]) if len(vector) > 11 else 0.0


def _update_duals(dual: DualState, rollout: dict[str, Any], cfg: PPOTrainConfig) -> None:
    if not cfg.constrained:
        return
    if cfg.risk_aware_constraints:
        dual.quality_normal = _dual_update(dual.quality_normal, _mean(rollout["quality_costs_normal"]), cfg.quality_cost_limit_normal, cfg)
        dual.quality_critical = _dual_update(dual.quality_critical, _mean(rollout["quality_costs_critical"]), cfg.quality_cost_limit_critical, cfg)
        dual.deadline_normal = _dual_update(dual.deadline_normal, _mean(rollout["deadline_costs_normal"]), cfg.deadline_cost_limit_normal, cfg)
        dual.deadline_critical = _dual_update(dual.deadline_critical, _mean(rollout["deadline_costs_critical"]), cfg.deadline_cost_limit_critical, cfg)
    else:
        quality = _dual_update((dual.quality_normal + dual.quality_critical) / 2.0, _mean(rollout["quality_costs"]), cfg.quality_cost_limit, cfg)
        deadline = _dual_update((dual.deadline_normal + dual.deadline_critical) / 2.0, _mean(rollout["deadline_costs"]), cfg.deadline_cost_limit, cfg)
        dual.quality_normal = quality
        dual.quality_critical = quality
        dual.deadline_normal = deadline
        dual.deadline_critical = deadline
    dual.conflict = _dual_update(dual.conflict, _mean(rollout["conflict_costs"]), cfg.conflict_cost_limit, cfg)
    dual.battery = _dual_update(dual.battery, _mean(rollout["battery_costs"]), cfg.battery_cost_limit, cfg)
    dual.gpu = _dual_update(dual.gpu, _mean(rollout["gpu_costs"]), cfg.gpu_cost_limit, cfg)


def _collect_demonstrations(env: V19LUTResourceEnv, cfg: PPOTrainConfig, seed: int) -> list[dict[str, Any]]:
    demos: list[dict[str, Any]] = []
    for episode in range(max(1, int(cfg.demo_episodes))):
        obs = env.reset(seed=seed + 10_000 + episode, options={"policy_name": f"demo_{cfg.demo_policy}"})
        done = False
        while not done:
            action = _demo_action(env, obs, cfg.demo_policy)
            parsed = env.parse_action(action)
            demos.append(_demo_sample(env, obs, parsed, cfg))
            obs, _reward, done, _info = env.step(parsed)
    return demos


def _demo_action(env: V19LUTResourceEnv, obs: dict[str, Any], policy: str) -> dict[str, Any]:
    if policy in {"greedy_min_sufficient_evidence", "semantic_greedy"}:
        epsilon = float(obs["epsilon_k"])
        for level in env.service_levels:
            if env.candidate_metrics(level, obs)["accuracy"] >= epsilon:
                return env.candidate_action(level, obs)
        return env.candidate_action(env.service_levels[-1], obs)
    if policy == "oracle_best_feasible_evidence":
        candidates = []
        for level in env.service_levels:
            action = env.candidate_action(level, obs)
            info = env.evaluate_action(action, obs)
            if bool(info.get("success", False)):
                candidates.append((float(info.get("energy_j", 0.0)) + 0.2 * float(info.get("payload_kb", 0.0)), level))
        if candidates:
            return env.candidate_action(min(candidates)[1], obs)
        best_level = max(env.service_levels, key=lambda level: env.candidate_metrics(level, obs)["accuracy"])
        return env.candidate_action(best_level, obs)
    raise ValueError(f"unknown demo policy: {policy}")


def _demo_sample(env: V19LUTResourceEnv, obs: dict[str, Any], action: dict[str, Any], cfg: PPOTrainConfig) -> dict[str, Any]:
    level = int(action["service_level"])
    service_idx = env.service_levels.index(level if level in env.service_levels else env.service_levels[0])
    bandwidth = float(action.get("bandwidth", action.get("bandwidth_hz", 1.0)))
    bandwidth_share = bandwidth / max(1.0, env.base_bandwidth_hz) if bandwidth > 1.0 else bandwidth
    bandwidth_floor = _service_resource_floor(cfg, level, "bandwidth") if level > 0 else cfg.bandwidth_floor
    power_floor = _service_resource_floor(cfg, level, "power") if level > 0 else cfg.power_min_w
    cpu_floor = _service_resource_floor(cfg, level, "cpu_share") if level > 0 else cfg.share_floor
    gpu_floor = _service_resource_floor(cfg, level, "gpu_share") if level > 0 else cfg.share_floor
    resource_target = [
        _unit_from_scaled(bandwidth_share, bandwidth_floor, 1.0),
        _unit_from_scaled(float(action.get("power", action.get("power_w", cfg.power_min_w))), max(cfg.power_min_w, power_floor), cfg.power_max_w),
        _unit_from_scaled(float(action.get("cpu_share", cfg.share_floor)), cpu_floor, 1.0),
        _unit_from_scaled(float(action.get("gpu_share", cfg.share_floor)), gpu_floor, 1.0),
    ]
    return {
        "observation": list(obs["vector"]),
        "service_action": service_idx,
        "service_mask": _plain_service_mask(obs, env.service_levels, cfg),
        "resource_target": resource_target,
    }


def _behavior_clone(model: HybridActorCritic, optimizer: Any, demos: list[dict[str, Any]], cfg: PPOTrainConfig) -> float:
    if not demos:
        return 0.0
    losses = []
    for _ in range(max(1, int(cfg.bc_epochs))):
        loss = _bc_loss(model, demos, cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sanitize_model_parameters(model)
        losses.append(float(loss.detach().cpu().item()))
    return float(losses[-1] if losses else 0.0)


def _behavior_clone_two_timescale(
    model: TwoTimescaleMobilitySemanticActorCritic,
    optimizer: Any,
    demos: list[dict[str, Any]],
    cfg: PPOTrainConfig,
) -> float:
    if not demos:
        return 0.0
    losses = []
    for _ in range(max(1, int(cfg.bc_epochs))):
        loss = _bc_loss_two_timescale(model, demos, cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sanitize_model_parameters(model)
        losses.append(float(loss.detach().cpu().item()))
    return float(losses[-1] if losses else 0.0)


def _bc_loss(model: HybridActorCritic, demos: list[dict[str, Any]], cfg: PPOTrainConfig) -> Any:
    device = _model_device(model)
    batch = demos
    if len(demos) > int(cfg.bc_batch_size):
        idx = torch.randint(0, len(demos), (int(cfg.bc_batch_size),))
        batch = [demos[int(i)] for i in idx]
    obs = torch.as_tensor([item["observation"] for item in batch], dtype=torch.float32, device=device)
    service_actions = torch.as_tensor([item["service_action"] for item in batch], dtype=torch.int64, device=device)
    service_masks = torch.as_tensor([item["service_mask"] for item in batch], dtype=torch.bool, device=device)
    resource_targets = torch.as_tensor([item["resource_target"] for item in batch], dtype=torch.float32, device=device)
    logits, resource_mean, _log_std, _value = model(obs)
    logits = _apply_service_mask(logits, service_masks)
    loss = nn.functional.cross_entropy(logits, service_actions)
    if cfg.hybrid_actions:
        loss = loss + cfg.bc_weight * nn.functional.mse_loss(torch.sigmoid(resource_mean), resource_targets)
    return loss


def _bc_loss_two_timescale(
    model: TwoTimescaleMobilitySemanticActorCritic,
    demos: list[dict[str, Any]],
    cfg: PPOTrainConfig,
) -> Any:
    device = _model_device(model)
    batch = demos
    if len(demos) > int(cfg.bc_batch_size):
        idx = torch.randint(0, len(demos), (int(cfg.bc_batch_size),))
        batch = [demos[int(i)] for i in idx]
    obs = torch.as_tensor([item["observation"] for item in batch], dtype=torch.float32, device=device)
    service_actions = torch.as_tensor([item["service_action"] for item in batch], dtype=torch.int64, device=device)
    service_masks = torch.as_tensor([item["service_mask"] for item in batch], dtype=torch.bool, device=device)
    resource_targets = torch.as_tensor([item["resource_target"] for item in batch], dtype=torch.float32, device=device)
    (
        uav_logits,
        mode_logits,
        _mobility_mean,
        _mobility_log_std,
        service_logits,
        resource_mean,
        _resource_log_std,
        _value,
    ) = model(obs)
    service_logits = _apply_service_mask(service_logits, service_masks)
    loss = nn.functional.cross_entropy(service_logits, service_actions)
    if cfg.hybrid_actions:
        loss = loss + cfg.bc_weight * nn.functional.mse_loss(torch.sigmoid(resource_mean), resource_targets)
    if not cfg.no_mobility_actor:
        target_uav = torch.zeros((len(batch),), dtype=torch.int64, device=device)
        target_mode = torch.as_tensor(
            [0 if int(item["service_action"]) == 0 else 1 for item in batch],
            dtype=torch.int64,
            device=device,
        )
        loss = loss + 0.1 * nn.functional.cross_entropy(uav_logits, target_uav)
        loss = loss + 0.1 * nn.functional.cross_entropy(mode_logits, target_mode)
    return loss


def _sanitize_model_parameters(model: Any) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.nan_to_num(param.data, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)


def _plain_service_mask(obs: dict[str, Any], service_levels: list[int], cfg: PPOTrainConfig) -> list[bool]:
    if not cfg.safety_layer:
        return [True for _ in service_levels]
    out = [_service_allowed(obs, level) for level in service_levels]
    return out if any(out) else [True for _ in service_levels]


def _scheduled_entropy_weight(cfg: PPOTrainConfig, episode: int) -> float:
    if int(cfg.entropy_decay_episodes) <= 0:
        return float(cfg.entropy_weight)
    progress = max(0.0, min(1.0, float(episode) / max(1.0, float(cfg.entropy_decay_episodes))))
    return float(cfg.entropy_weight_start) + progress * (float(cfg.entropy_weight_end) - float(cfg.entropy_weight_start))


def _scheduled_service_prior_weight(cfg: PPOTrainConfig, episode: int) -> float:
    if int(cfg.service_prior_decay_episodes) <= 0:
        return 0.0
    progress = max(0.0, min(1.0, float(episode) / max(1.0, float(cfg.service_prior_decay_episodes))))
    return float(cfg.service_prior_weight) * (1.0 - progress)


def _service_resource_floor(cfg: PPOTrainConfig, service_level: int, key: str) -> float:
    level = int(service_level)
    if key == "bandwidth":
        return {
            1: float(cfg.semantic_token_bandwidth_floor),
            2: float(cfg.image_bandwidth_floor),
            3: float(cfg.roi_bandwidth_floor),
        }.get(level, float(cfg.bandwidth_floor))
    if key == "power":
        return {
            1: float(cfg.semantic_token_power_floor),
            2: float(cfg.image_power_floor),
            3: float(cfg.roi_power_floor),
        }.get(level, float(cfg.power_min_w))
    if key == "cpu_share":
        return {
            1: float(cfg.semantic_token_cpu_floor),
            2: float(cfg.image_cpu_floor),
            3: float(cfg.roi_cpu_floor),
        }.get(level, float(cfg.share_floor))
    if key == "gpu_share":
        return {
            1: float(cfg.semantic_token_gpu_floor),
            2: float(cfg.image_gpu_floor),
            3: float(cfg.roi_gpu_floor),
        }.get(level, float(cfg.share_floor))
    raise KeyError(key)


def _resource_floor(cfg: PPOTrainConfig, service_level: int, key: str) -> float:
    if not cfg.resource_projection:
        if key == "bandwidth":
            return float(cfg.bandwidth_floor)
        if key == "power":
            return float(cfg.power_min_w)
        if key in {"cpu_share", "gpu_share"}:
            return float(cfg.share_floor)
        raise KeyError(key)
    return _service_resource_floor(cfg, service_level, key)


def _scaled_unit(value: float, low: float, high: float) -> float:
    unit = max(0.0, min(1.0, float(value)))
    return float(low) + unit * (float(high) - float(low))


def _unit_from_scaled(value: float, low: float, high: float) -> float:
    if math.isclose(float(high), float(low)):
        return 0.0
    return max(0.0, min(1.0, (float(value) - float(low)) / (float(high) - float(low))))


def _default_resource_values() -> list[float]:
    return [1.0, 1.0, 0.5, 0.5]


def _dual_update(current: float, observed_cost: float, limit: float, cfg: PPOTrainConfig) -> float:
    updated = float(current) + float(cfg.lambda_lr) * (float(observed_cost) - float(limit))
    return max(0.0, min(float(cfg.lambda_max), updated))


def _discounted_returns(rewards: list[float], dones: list[bool], gamma: float) -> list[float]:
    out = []
    running = 0.0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        running = float(reward) + gamma * running * float(not done)
        out.append(running)
    out.reverse()
    return out


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _std(values: list[float], mean: float | None = None) -> float:
    if not values:
        return 0.0
    avg = _mean(values) if mean is None else float(mean)
    return float((sum((float(value) - avg) ** 2 for value in values) / len(values)) ** 0.5)
