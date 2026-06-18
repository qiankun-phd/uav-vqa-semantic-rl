from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any

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
    sensed_snr_db: float
    snr_bin: str
    service_level: int
    bandwidth_hz: float
    power_w: float
    cpu_share: float
    gpu_share: float
    uav_assignment: int
    answer_accuracy_est: float
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
        policy_name: str = "policy",
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
        options["policy_name"] = self.policy_name
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
        obs, reward, done, info = self._env.step(parsed)
        record = self._record_from_info(info, float(reward))
        self._last_record = record
        info = dict(info)
        info.update(
            {
                "answer_accuracy_est": record.answer_accuracy_est,
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
                "snr_bin": record.snr_bin,
                "service_level": record.service_level,
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
        return self._env.candidate_metrics(self._nearest_service_level(service_level), obs)

    def action_spec(self) -> dict[str, Any]:
        return self._env.action_spec()

    def action_mask(self) -> dict[str, Any]:
        return self._env.action_mask()

    def evaluate_action(self, action: dict[str, Any], obs: dict[str, Any] | None = None) -> dict[str, Any]:
        parsed = self.parse_action(action)
        task_id = str(obs.get("task_id")) if obs and obs.get("task_id") else None
        return self._env.evaluate_action(parsed, task_id=task_id, obs=obs, mutate=False)

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
            sensed_snr_db=float(info.get("sensed_snr_db", 0.0)),
            snr_bin=str(info.get("snr_bin", "")),
            service_level=int(info.get("service_level", self.service_levels[0])),
            bandwidth_hz=float(info.get("bandwidth_hz", 0.0)),
            power_w=float(info.get("power_w", 0.0)),
            cpu_share=float(info.get("cpu_share", 0.0)),
            gpu_share=float(info.get("gpu_share", 0.0)),
            uav_assignment=int(info.get("uav_assignment", 0)),
            answer_accuracy_est=float(info.get("answer_accuracy_est", 0.0)),
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
            reward=float(info.get("reward", reward)),
        )

    def _nearest_service_level(self, level: int) -> int:
        if int(level) in self.service_levels:
            return int(level)
        return min(self.service_levels, key=lambda item: abs(item - int(level)))

    @staticmethod
    def _normalize_obs(obs: dict[str, Any]) -> dict[str, Any]:
        out = dict(obs)
        out["vector"] = [float(value) for value in out.get("vector", [])]
        return out

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
