from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from vqa_semcom.config import resolve_path
from vqa_semcom.semantic.utility import SemanticUtilityModel
from vqa_semcom.sim.multi_uav_env import MultiUAVVQAEnv
from vqa_semcom.sim.resource_env import LUTEntry


@dataclass(frozen=True)
class V19StepRecord:
    episode: int
    episode_step: int
    task_id: str
    policy: str
    question_type: str
    risk_level: str
    view_quality_bin: str
    freshness_bin: str
    epsilon_k: float
    sensed_snr_db: float
    snr_bin: str
    service_level: int
    bandwidth_hz: float
    power_w: float
    cpu_share: float
    gpu_share: float
    uav_assignment: int
    mobility_mode: str
    waypoint_x: float
    waypoint_y: float
    altitude_m: float
    fly_distance_m: float
    coverage_gain: float
    mobility_energy_j: float
    arrival_delay_s: float
    utm_conflict_risk: float
    answer_accuracy_est: float
    semantic_accuracy_mean: float
    semantic_accuracy_lcb: float
    semantic_uncertainty: float
    semantic_sample_count: int
    semantic_payload_kb: float
    semantic_quality_gap: float
    semantic_success: bool
    q_quality: float
    q_deadline: float
    q_energy: float
    q_risk: float
    q_utm: float
    success: bool
    delay_s: float
    energy_j: float
    payload_kb: float
    quality_violation: bool
    deadline_violation: bool
    battery_violation: bool
    resource_violation: bool
    airspace_conflict: bool
    gpu_memory_ok: bool
    battery_remaining_j: float
    utm_constraint_violation: bool
    utm_conflict_violation: bool
    dss_available: bool
    dss_delay_s: float
    subscription_notification_delay_s: float
    utm_dss_delay_s: float
    utm_notification_delay_s: float
    operational_intent_state: str
    off_nominal_planning_penalty: float
    reward: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class V19LUTResourceEnv:
    """Algorithm-facing V1.9 wrapper around the canonical multi-UAV simulator.

    The environment dynamics live in :class:`MultiUAVVQAEnv`.  This class keeps
    the vector observation, candidate action helpers, and ``V19StepRecord``
    shape expected by the V1.9 baseline/PPO scripts.
    """

    def __init__(
        self,
        tasks: list[dict[str, str]],
        lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
        cfg: dict[str, Any],
        seed: int = 0,
        snr_bins_db: list[float] | None = None,
        tasks_per_episode: int | None = None,
        service_levels: list[int] | None = None,
        formal_scenario: str | None = None,
        policy_name: str = "policy",
        state_version: str | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("V19LUTResourceEnv needs at least one task")
        if not lut:
            raise ValueError("V19LUTResourceEnv needs a non-empty LUT")
        self.tasks = list(tasks)
        self.lut = lut
        self.cfg = cfg
        self.seed_value = int(seed)
        self.policy_name = policy_name
        self.formal_scenario = str(formal_scenario or "")
        rl_cfg = cfg.get("rl", {}) if isinstance(cfg, dict) else {}
        self.state_version = str(state_version or rl_cfg.get("state_version", "v1")).lower()
        if self.state_version not in {"v1", "v2"}:
            raise ValueError(f"unknown state_version: {self.state_version}")
        sim_cfg = cfg.get("simulation", {})
        self.tasks_per_episode = int(tasks_per_episode or sim_cfg.get("tasks_per_episode", 40))
        self._env_cfg = self._cfg_with_overrides(cfg, snr_bins_db, service_levels)
        env_override = dict(self._env_cfg.get("multi_uav_env", {}))
        env_override["tasks_per_episode"] = self.tasks_per_episode
        env_override["episode_steps"] = self.tasks_per_episode
        self._env_cfg["multi_uav_env"] = env_override
        self._env = MultiUAVVQAEnv(self.tasks, self.lut, self._env_cfg, seed=self.seed_value)
        self.service_levels = self._env.service_levels()
        self.base_bandwidth_hz = float(self._env.env_cfg["bandwidth_hz"])
        self.episode = -1
        self.episode_step = 0
        self.current_context: dict[str, Any] | None = None
        self._last_obs: dict[str, Any] | None = None
        self._last_record: V19StepRecord | None = None
        self._semantic_utility = self._load_semantic_utility_model(cfg)
        env_cfg = self._env_cfg.get("multi_uav_env", {}) if isinstance(self._env_cfg, dict) else {}
        self.energy_budget_j = float(rl_cfg.get("energy_budget_j", env_cfg.get("energy_budget_j", 500.0)))
        self._state_v2_service_levels = self._canonical_state_v2_service_levels(cfg)
        self._state_v2_max_uavs = max(
            int(rl_cfg.get("state_v2_max_uavs", 8)),
            int(env_cfg.get("num_uavs", 1)),
            int(self.action_spec().get("num_uavs", 1)),
        )
        self._state_v2_mobility_modes = ("stay", "serve_task", "reposition", "avoid_conflict", "return_base")
        self._state_v2_delay_norm_s = max(1.0, float(rl_cfg.get("state_v2_delay_norm_s", 60.0)))
        self._state_v2_energy_norm_j = max(1.0, float(rl_cfg.get("state_v2_energy_norm_j", self.energy_budget_j)))
        self._queue_state = self._empty_queue_state()

    @property
    def obs_dim(self) -> int:
        if self._last_obs is None:
            self.reset(seed=self.seed_value)
        assert self._last_obs is not None
        return int(len(self._last_obs["vector"]))

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed_value = int(seed)
        options = dict(options or {})
        self.policy_name = str(options.get("policy_name", self.policy_name))
        options.setdefault("tasks_per_episode", self.tasks_per_episode)
        if self.formal_scenario:
            options.setdefault("formal_scenario", self.formal_scenario)
        options["policy_name"] = self.policy_name
        self._queue_state = self._empty_queue_state()
        obs = self._env.reset(seed=self.seed_value, options=options)
        self.service_levels = self._env.service_levels()
        self.base_bandwidth_hz = float(self._env.env_cfg["bandwidth_hz"])
        self.episode = int(self._env.episode)
        self.episode_step = int(self._env.step_count)
        self.current_context = self._obs_context(obs)
        self._last_obs = self._normalize_obs(obs)
        return self._last_obs

    def step(self, action: dict[str, Any] | int | Integral) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        parsed = self.parse_action(action)
        prev_obs = self._last_obs
        obs, reward, done, info = self._env.step(parsed)
        info = self._enrich_semantic_info(dict(info), prev_obs)
        self._update_virtual_queues(info)
        record = self._record_from_info(info, float(reward))
        self._last_record = record
        info.update(
            {
                "answer_accuracy_est": record.answer_accuracy_est,
                "semantic_accuracy_mean": record.semantic_accuracy_mean,
                "semantic_accuracy_lcb": record.semantic_accuracy_lcb,
                "semantic_uncertainty": record.semantic_uncertainty,
                "semantic_sample_count": record.semantic_sample_count,
                "semantic_payload_kb": record.semantic_payload_kb,
                "semantic_quality_gap": record.semantic_quality_gap,
                "semantic_success": record.semantic_success,
                "q_quality": record.q_quality,
                "q_deadline": record.q_deadline,
                "q_energy": record.q_energy,
                "q_risk": record.q_risk,
                "q_utm": record.q_utm,
                "success": record.success,
                "delay_s": record.delay_s,
                "energy_j": record.energy_j,
                "payload_kb": record.payload_kb,
                "quality_violation": record.quality_violation,
                "deadline_violation": record.deadline_violation,
                "battery_violation": record.battery_violation,
                "resource_violation": record.resource_violation,
                "airspace_conflict": record.airspace_conflict,
                "gpu_memory_ok": record.gpu_memory_ok,
                "battery_remaining_j": record.battery_remaining_j,
                "utm_constraint_violation": record.utm_constraint_violation,
                "utm_conflict_violation": record.utm_conflict_violation,
                "dss_available": record.dss_available,
                "dss_delay_s": record.dss_delay_s,
                "subscription_notification_delay_s": record.subscription_notification_delay_s,
                "utm_dss_delay_s": record.utm_dss_delay_s,
                "utm_notification_delay_s": record.utm_notification_delay_s,
                "operational_intent_state": record.operational_intent_state,
                "off_nominal_planning_penalty": record.off_nominal_planning_penalty,
                "snr_bin": record.snr_bin,
                "service_level": record.service_level,
                "mobility_mode": record.mobility_mode,
                "waypoint_x": record.waypoint_x,
                "waypoint_y": record.waypoint_y,
                "altitude_m": record.altitude_m,
                "fly_distance_m": record.fly_distance_m,
                "coverage_gain": record.coverage_gain,
                "mobility_energy_j": record.mobility_energy_j,
                "arrival_delay_s": record.arrival_delay_s,
                "utm_conflict_risk": record.utm_conflict_risk,
                "bandwidth_unit": "Hz",
                "power_unit": "W",
                "record": record.to_row(),
            }
        )
        self.episode = int(self._env.episode)
        self.episode_step = int(self._env.step_count)
        self.current_context = self._obs_context(obs)
        self._last_obs = self._normalize_obs(obs)
        return self._last_obs, float(reward), bool(done), info

    def parse_action(self, action: dict[str, Any] | int | Integral) -> dict[str, Any]:
        if isinstance(action, Integral):
            action = {"service_level": int(action)}
        parsed = self._env.parse_action(dict(action))
        parsed["service_level"] = self._nearest_service_level(int(parsed["service_level"]))
        return parsed

    def default_action(self, service_level: int) -> dict[str, Any]:
        return self._env.default_action(self._nearest_service_level(service_level))

    def candidate_action(self, service_level: int, obs: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._env.candidate_action(self._nearest_service_level(service_level), obs)

    def candidate_metrics(self, service_level: int, obs: dict[str, Any] | None = None) -> dict[str, float]:
        info = self.evaluate_action(self.candidate_action(service_level, obs), obs)
        return {
            "accuracy": float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0))),
            "accuracy_mean": float(info.get("semantic_accuracy_mean", info.get("answer_accuracy_est", 0.0))),
            "uncertainty": float(info.get("semantic_uncertainty", 0.0)),
            "delay_s": float(info.get("delay_s", info.get("total_delay_s", 0.0))),
            "energy_j": float(info.get("energy_j", info.get("total_energy_j", 0.0))),
            "payload_kb": float(info.get("payload_kb", 0.0)),
            "success": float(bool(info.get("success", False))),
        }

    def action_spec(self) -> dict[str, Any]:
        return self._env.action_spec()

    def action_mask(self) -> dict[str, Any]:
        return self._env.action_mask()

    def evaluate_action(self, action: dict[str, Any], obs: dict[str, Any] | None = None) -> dict[str, Any]:
        parsed = self.parse_action(action)
        task_id = str(obs.get("task_id")) if obs and obs.get("task_id") else None
        info = self._env.evaluate_action(parsed, task_id=task_id, obs=obs, mutate=False)
        return self._enrich_semantic_info(dict(info), obs)

    def _record_from_info(self, info: dict[str, Any], reward: float) -> V19StepRecord:
        return V19StepRecord(
            episode=int(info.get("episode", self._env.episode)),
            episode_step=int(info.get("episode_step", max(0, self._env.step_count - 1))),
            task_id=str(info.get("task_id", "")),
            policy=self.policy_name,
            question_type=str(info.get("question_type", info.get("task_type", ""))),
            risk_level=str(info.get("risk_level", "normal")),
            view_quality_bin=str(info.get("view_quality_bin", "medium")),
            freshness_bin=str(info.get("freshness_bin", "fresh")),
            epsilon_k=float(info.get("epsilon_k", 0.0)),
            sensed_snr_db=float(info.get("sensed_snr_db", 0.0)),
            snr_bin=str(info.get("snr_bin", "")),
            service_level=int(info.get("service_level", self.service_levels[0])),
            bandwidth_hz=float(info.get("bandwidth_hz", 0.0)),
            power_w=float(info.get("power_w", 0.0)),
            cpu_share=float(info.get("cpu_share", 0.0)),
            gpu_share=float(info.get("gpu_share", 0.0)),
            uav_assignment=int(info.get("uav_assignment", 0)),
            mobility_mode=str(info.get("mobility_mode", "stay")),
            waypoint_x=float(info.get("waypoint_x", 0.0)),
            waypoint_y=float(info.get("waypoint_y", 0.0)),
            altitude_m=float(info.get("altitude_m", 0.0)),
            fly_distance_m=float(info.get("fly_distance_m", 0.0)),
            coverage_gain=float(info.get("coverage_gain", 0.0)),
            mobility_energy_j=float(info.get("mobility_energy_j", 0.0)),
            arrival_delay_s=float(info.get("arrival_delay_s", 0.0)),
            utm_conflict_risk=float(info.get("utm_conflict_risk", 0.0)),
            answer_accuracy_est=float(info.get("answer_accuracy_est", 0.0)),
            semantic_accuracy_mean=float(info.get("semantic_accuracy_mean", info.get("answer_accuracy_est", 0.0))),
            semantic_accuracy_lcb=float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0))),
            semantic_uncertainty=float(info.get("semantic_uncertainty", 0.0)),
            semantic_sample_count=int(info.get("semantic_sample_count", 0)),
            semantic_payload_kb=float(info.get("semantic_payload_kb", info.get("payload_kb", 0.0))),
            semantic_quality_gap=float(info.get("semantic_quality_gap", 0.0)),
            semantic_success=bool(info.get("semantic_success", False)),
            q_quality=float(info.get("q_quality", 0.0)),
            q_deadline=float(info.get("q_deadline", 0.0)),
            q_energy=float(info.get("q_energy", 0.0)),
            q_risk=float(info.get("q_risk", 0.0)),
            q_utm=float(info.get("q_utm", 0.0)),
            success=bool(info.get("success", False)),
            delay_s=float(info.get("delay_s", info.get("total_delay_s", 0.0))),
            energy_j=float(info.get("energy_j", info.get("total_energy_j", 0.0))),
            payload_kb=float(info.get("payload_kb", 0.0)),
            quality_violation=bool(info.get("quality_violation", False)),
            deadline_violation=bool(info.get("deadline_violation", False)),
            battery_violation=bool(info.get("battery_violation", False)),
            resource_violation=bool(info.get("resource_violation", False)),
            airspace_conflict=bool(info.get("airspace_conflict", False)),
            gpu_memory_ok=bool(info.get("gpu_memory_ok", True)),
            battery_remaining_j=float(info.get("battery_remaining_j", 0.0)),
            utm_constraint_violation=bool(info.get("utm_constraint_violation", False)),
            utm_conflict_violation=bool(info.get("utm_conflict_violation", info.get("utm_constraint_violation", False))),
            dss_available=bool(info.get("dss_available", True)),
            dss_delay_s=float(info.get("dss_delay_s", 0.0)),
            subscription_notification_delay_s=float(info.get("subscription_notification_delay_s", 0.0)),
            utm_dss_delay_s=float(info.get("utm_dss_delay_s", info.get("dss_delay_s", 0.0))),
            utm_notification_delay_s=float(
                info.get("utm_notification_delay_s", info.get("subscription_notification_delay_s", 0.0))
            ),
            operational_intent_state=str(info.get("operational_intent_state", "accepted")),
            off_nominal_planning_penalty=float(info.get("off_nominal_planning_penalty", 0.0)),
            reward=float(info.get("reward", reward)),
        )

    def _enrich_semantic_info(self, info: dict[str, Any], obs: dict[str, Any] | None) -> dict[str, Any]:
        """Expose the calibrated semantic-utility API to controllers and CSVs.

        The simulator still produces the raw LUT accuracy estimate.  RL control
        uses the conservative LCB so sparse VQA cells do not look artificially
        attractive during exploration.
        """

        raw_accuracy = float(info.get("answer_accuracy_est", 0.0))
        info.setdefault("semantic_accuracy_mean", raw_accuracy)
        info.setdefault("semantic_accuracy_lcb", raw_accuracy)
        info.setdefault("semantic_uncertainty", 0.0)
        info.setdefault("semantic_sample_count", 0)
        model = self._semantic_utility
        if model is not None:
            query = self._semantic_query(info, obs)
            estimate = model.U_sem(
                query["question_type"],
                int(query["service_level"]),
                query["snr_bin"],
                query["view_quality_bin"],
                query["freshness_bin"],
                query["risk_level"],
            )
            info["semantic_accuracy_mean"] = float(estimate.accuracy_mean)
            info["semantic_accuracy_lcb"] = float(estimate.accuracy_lcb)
            info["semantic_uncertainty"] = float(estimate.uncertainty)
            info["semantic_sample_count"] = int(estimate.sample_count)
            info["answer_accuracy_raw"] = raw_accuracy
            info["answer_accuracy_est"] = float(estimate.accuracy_lcb)
            info["payload_kb"] = float(estimate.payload_kb)
            info["semantic_payload_kb"] = float(estimate.payload_kb)

        epsilon = self._epsilon_from_obs_or_info(info, obs)
        accuracy_lcb = float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0)))
        delay_s = float(info.get("delay_s", info.get("total_delay_s", 0.0)))
        deadline_s = self._deadline_from_obs_or_info(info, obs)
        energy_j = float(info.get("energy_j", info.get("total_energy_j", 0.0)))
        energy_budget_j = self._energy_budget_from_obs_or_info(info, obs)
        info.setdefault("semantic_payload_kb", float(info.get("payload_kb", 0.0)))
        info["epsilon_k"] = float(epsilon)
        info["semantic_quality_gap"] = max(0.0, epsilon - accuracy_lcb)
        info["semantic_success"] = bool(accuracy_lcb >= epsilon)
        info["deadline_s"] = float(deadline_s)
        info["energy_budget_j"] = float(energy_budget_j)
        info["q_quality_increment"] = max(0.0, epsilon - accuracy_lcb)
        info["q_deadline_increment"] = max(0.0, delay_s - deadline_s)
        info["q_energy_increment"] = max(0.0, energy_j - energy_budget_j)
        info["quality_violation"] = bool(float(info.get("answer_accuracy_est", 0.0)) < epsilon)
        info.setdefault("utm_constraint_violation", False)
        info.setdefault("utm_conflict_violation", bool(info.get("airspace_conflict", False) or info.get("utm_constraint_violation", False)))
        info.setdefault("dss_available", True)
        info.setdefault("dss_delay_s", 0.0)
        info.setdefault("subscription_notification_delay_s", 0.0)
        info.setdefault("utm_dss_delay_s", info.get("dss_delay_s", 0.0))
        info.setdefault("utm_notification_delay_s", info.get("subscription_notification_delay_s", 0.0))
        info.setdefault("mobility_mode", "stay" if int(info.get("service_level", 0)) == 0 else "serve_task")
        info.setdefault("waypoint_x", 0.0)
        info.setdefault("waypoint_y", 0.0)
        info.setdefault("altitude_m", 0.0)
        info.setdefault("fly_distance_m", 0.0)
        info.setdefault("coverage_gain", 0.0)
        info.setdefault("mobility_energy_j", 0.0)
        info.setdefault("arrival_delay_s", 0.0)
        info.setdefault("utm_conflict_risk", 0.0)
        state = str(info.get("operational_intent_state", "accepted"))
        info["operational_intent_state"] = state
        info["off_nominal_planning_penalty"] = 1.0 if state in {"nonconforming", "contingent"} else 0.0
        info["q_risk_increment"] = self._risk_increment(info)
        info["q_utm_increment"] = float(bool(info.get("utm_constraint_violation", False)))
        info["success"] = not (
            bool(info.get("quality_violation", False))
            or bool(info.get("deadline_violation", False))
            or bool(info.get("battery_violation", False))
            or bool(info.get("resource_violation", False))
            or bool(info.get("airspace_conflict", False))
            or bool(info.get("utm_constraint_violation", False))
        )
        self._attach_queue_state(info)
        return info

    def _semantic_query(self, info: dict[str, Any], obs: dict[str, Any] | None) -> dict[str, Any]:
        obs = obs or self._last_obs or self.current_context or {}
        return {
            "question_type": str(info.get("question_type", obs.get("question_type", obs.get("task_type", "")))),
            "service_level": int(info.get("service_level", self.service_levels[0])),
            "snr_bin": str(info.get("snr_bin", obs.get("snr_bin", ""))),
            "view_quality_bin": str(info.get("view_quality_bin", obs.get("view_quality_bin", "medium"))),
            "freshness_bin": str(info.get("freshness_bin", obs.get("freshness_bin", "fresh"))),
            "risk_level": str(info.get("risk_level", obs.get("risk_level", "normal"))),
        }

    @staticmethod
    def _epsilon_from_obs_or_info(info: dict[str, Any], obs: dict[str, Any] | None) -> float:
        if "epsilon_k" in info:
            return float(info["epsilon_k"])
        if obs and "epsilon_k" in obs:
            return float(obs["epsilon_k"])
        return 0.0

    @staticmethod
    def _deadline_from_obs_or_info(info: dict[str, Any], obs: dict[str, Any] | None) -> float:
        if "deadline_s" in info:
            return float(info["deadline_s"])
        if "tau_k" in info:
            return float(info["tau_k"])
        if obs and "deadline_s" in obs:
            return float(obs["deadline_s"])
        if obs and "tau_k" in obs:
            return float(obs["tau_k"])
        return 5.0

    def _energy_budget_from_obs_or_info(self, info: dict[str, Any], obs: dict[str, Any] | None) -> float:
        if "energy_budget_j" in info:
            return float(info["energy_budget_j"])
        if obs and "energy_budget_j" in obs:
            return float(obs["energy_budget_j"])
        return float(self.energy_budget_j)

    @staticmethod
    def _load_semantic_utility_model(cfg: dict[str, Any]) -> SemanticUtilityModel | None:
        paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
        candidates = [
            paths.get("semantic_utility_csv"),
            Path("outputs") / "lut" / "v1_9_semantic_utility_with_ci.csv",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = resolve_path(candidate)
            if path.exists():
                return SemanticUtilityModel.from_csv(path)
        return None

    def _nearest_service_level(self, level: int) -> int:
        if int(level) in self.service_levels:
            return int(level)
        return min(self.service_levels, key=lambda item: abs(item - int(level)))

    def _normalize_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        out = dict(obs)
        vector = [float(value) for value in out.get("vector", [])] + self._queue_vector()
        if self.state_version == "v2":
            vector += self._state_v2_vector(out)
        out["lyapunov_queues"] = dict(self._queue_state)
        out["state_version"] = self.state_version
        out["vector"] = vector
        return out

    def _state_v2_vector(self, obs: dict[str, Any]) -> list[float]:
        features: list[float] = []
        deadline = max(1e-6, float(obs.get("deadline_s", obs.get("tau_k", 5.0))))
        epsilon = self._clip01(float(obs.get("epsilon_k", 0.0)))
        for level in self._state_v2_service_levels:
            if int(level) not in {int(item) for item in self.service_levels}:
                features.extend([0.0, 0.0, epsilon, 0.0, 0.0, 0.0, -1.0])
                continue
            action = self.candidate_action(level, obs)
            info = self.evaluate_action(action, obs)
            accuracy_lcb = self._clip01(float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0))))
            uncertainty = self._clip01(float(info.get("semantic_uncertainty", 0.0)))
            gap = self._clip01(max(0.0, epsilon - accuracy_lcb))
            estimated_delay = max(0.0, float(info.get("delay_s", info.get("total_delay_s", info.get("estimated_delay_s", 0.0)))))
            semantic_feasible = float(gap <= 1e-9)
            deadline_feasible = float(estimated_delay <= deadline)
            joint_feasible = float(bool(info.get("success", False)) or (semantic_feasible > 0.0 and deadline_feasible > 0.0))
            slack_ratio = self._clip((deadline - estimated_delay) / deadline, -1.0, 1.0)
            features.extend(
                [
                    accuracy_lcb,
                    uncertainty,
                    gap,
                    semantic_feasible,
                    deadline_feasible,
                    joint_feasible,
                    slack_ratio,
                ]
            )
        features += self._uav_mobility_feature_vector(obs)
        features += self._mask_feature_vector(obs)
        return features

    def _uav_mobility_feature_vector(self, obs: dict[str, Any]) -> list[float]:
        task_xy = self._front_task_xy(obs)
        features: list[float] = []
        for idx in range(self._state_v2_max_uavs):
            uav = self._uav_by_index_or_id(obs, idx)
            if uav is None:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue
            predicted_delay = max(0.0, float(uav.get("predicted_fly_delay_s", uav.get("arrival_delay_s", 0.0))))
            predicted_energy = max(0.0, float(uav.get("predicted_fly_energy_j", uav.get("mobility_energy_j", 0.0))))
            if task_xy is not None and predicted_delay <= 0.0:
                dx = float(uav.get("x_m", 0.0)) - task_xy[0]
                dy = float(uav.get("y_m", 0.0)) - task_xy[1]
                distance = (dx * dx + dy * dy) ** 0.5
                speed = max(1.0, float(uav.get("speed_mps", uav.get("max_speed_mps", 15.0))))
                predicted_delay = distance / speed
                predicted_energy = predicted_energy if predicted_energy > 0.0 else distance * 0.5
            battery = max(0.0, float(uav.get("battery_remaining_j", uav.get("battery_j", self.energy_budget_j))))
            features.extend(
                [
                    1.0,
                    self._clip01(predicted_delay / self._state_v2_delay_norm_s),
                    self._clip01(predicted_energy / self._state_v2_energy_norm_j),
                    self._clip01(battery / max(1.0, float(self.energy_budget_j))),
                ]
            )
        return features

    def _mask_feature_vector(self, obs: dict[str, Any]) -> list[float]:
        mask = obs.get("action_mask", {}) if isinstance(obs.get("action_mask", {}), dict) else {}
        service_allowed = mask.get("service_level_allowed", {}) if isinstance(mask.get("service_level_allowed", {}), dict) else {}
        service_values = [
            float(bool(service_allowed.get(level, service_allowed.get(str(level), int(level) in self.service_levels))))
            for level in self._state_v2_service_levels
        ]
        mode_allowed = mask.get("mobility_mode_allowed", {}) if isinstance(mask.get("mobility_mode_allowed", {}), dict) else {}
        mobility_values = [float(bool(mode_allowed.get(mode, True))) for mode in self._state_v2_mobility_modes]
        uav_ok = mask.get("uav_battery_ok", {}) if isinstance(mask.get("uav_battery_ok", {}), dict) else {}
        uav_values: list[float] = []
        for uid in range(self._state_v2_max_uavs):
            active = self._uav_by_index_or_id(obs, uid) is not None
            uav_values.append(float(active and bool(uav_ok.get(uid, uav_ok.get(str(uid), True)))))
        return service_values + mobility_values + uav_values

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(float(low), min(float(high), float(value)))

    @classmethod
    def _clip01(cls, value: float) -> float:
        return cls._clip(value, 0.0, 1.0)

    @staticmethod
    def _canonical_state_v2_service_levels(cfg: dict[str, Any]) -> tuple[int, ...]:
        bins = cfg.get("bins", {}) if isinstance(cfg, dict) else {}
        env_cfg = cfg.get("multi_uav_env", {}) if isinstance(cfg, dict) else {}
        levels = bins.get("service_levels") or env_cfg.get("enabled_service_levels") or [0, 1, 2]
        return tuple(int(level) for level in levels)

    @staticmethod
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

    @staticmethod
    def _uav_by_index_or_id(obs: dict[str, Any], uid: int) -> dict[str, Any] | None:
        uavs = list(obs.get("uav_state", []) or [])
        for idx, uav in enumerate(uavs):
            if int(uav.get("uav_id", idx)) == int(uid):
                return uav
        if 0 <= int(uid) < len(uavs):
            return uavs[int(uid)]
        return None

    @staticmethod
    def _empty_queue_state() -> dict[str, float]:
        return {"quality": 0.0, "deadline": 0.0, "energy": 0.0, "risk": 0.0, "utm": 0.0}

    def _queue_vector(self) -> list[float]:
        return [
            float(self._queue_state["quality"]),
            float(self._queue_state["deadline"]),
            float(self._queue_state["energy"]) / max(1.0, float(self.energy_budget_j)),
            float(self._queue_state["risk"]),
            float(self._queue_state["utm"]),
        ]

    def _attach_queue_state(self, info: dict[str, Any]) -> None:
        info["q_quality"] = float(self._queue_state["quality"])
        info["q_deadline"] = float(self._queue_state["deadline"])
        info["q_energy"] = float(self._queue_state["energy"])
        info["q_risk"] = float(self._queue_state["risk"])
        info["q_utm"] = float(self._queue_state["utm"])

    def _update_virtual_queues(self, info: dict[str, Any]) -> None:
        self._queue_state["quality"] = max(0.0, self._queue_state["quality"] + float(info.get("q_quality_increment", 0.0)))
        self._queue_state["deadline"] = max(0.0, self._queue_state["deadline"] + float(info.get("q_deadline_increment", 0.0)))
        self._queue_state["energy"] = max(0.0, self._queue_state["energy"] + float(info.get("q_energy_increment", 0.0)))
        self._queue_state["risk"] = max(0.0, self._queue_state["risk"] + float(info.get("q_risk_increment", 0.0)))
        self._queue_state["utm"] = max(0.0, self._queue_state["utm"] + float(info.get("q_utm_increment", 0.0)))
        self._attach_queue_state(info)

    @staticmethod
    def _risk_increment(info: dict[str, Any]) -> float:
        return float(
            bool(info.get("airspace_conflict", False))
            or bool(info.get("battery_violation", False))
            or bool(info.get("resource_violation", False))
            or bool(info.get("utm_constraint_violation", False))
            or float(info.get("off_nominal_planning_penalty", 0.0)) > 0.0
        )

    @staticmethod
    def _obs_context(obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": obs.get("task_id", ""),
            "episode_step": int(obs.get("episode_step", 0)),
            "question_type": obs.get("question_type", obs.get("task_type", "")),
            "risk_level": obs.get("risk_level", "normal"),
            "view_quality_bin": obs.get("view_quality_bin", "medium"),
            "freshness_bin": obs.get("freshness_bin", "fresh"),
            "sensed_snr_db": float(obs.get("sensed_snr_db", 0.0)),
            "snr_bin": obs.get("snr_bin", ""),
            "epsilon_k": float(obs.get("epsilon_k", 0.0)),
            "tau_k": float(obs.get("tau_k", obs.get("deadline_s", 0.0))),
        }

    @staticmethod
    def _cfg_with_overrides(
        cfg: dict[str, Any],
        snr_bins_db: list[float] | None,
        service_levels: list[int] | None,
    ) -> dict[str, Any]:
        out = dict(cfg)
        if snr_bins_db is not None or service_levels is not None:
            bins = dict(out.get("bins", {}))
            if snr_bins_db is not None:
                bins["snr_db"] = [float(value) for value in snr_bins_db]
            if service_levels is not None:
                bins["service_levels"] = [int(value) for value in service_levels]
            out["bins"] = bins
        if service_levels is not None:
            env_cfg = dict(out.get("multi_uav_env", {}))
            env_cfg["enabled_service_levels"] = [int(value) for value in service_levels]
            env_cfg["enable_service_level_3"] = 3 in {int(value) for value in service_levels}
            out["multi_uav_env"] = env_cfg
        return out
