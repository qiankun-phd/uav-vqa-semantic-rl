from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from vqa_semcom.config import resolve_path
from vqa_semcom.quality.persample_predictor import PersamplePredictor
from vqa_semcom.semantic.utility import SemanticUtilityModel
from vqa_semcom.sim.multi_uav_env import MultiUAVVQAEnv, count_bucket_v5
from vqa_semcom.sim.resource_env import LUTEntry
from vqa_semcom.snr import channel_bin_from_snr, snr_db_from_label

QUALITY_BACKENDS = ("lut", "persample", "lut_v5")
DEFAULT_PERSAMPLE_MODEL = Path("outputs") / "models" / "persample_predictor_v1.json"
DEFAULT_LUT_V5 = Path("outputs") / "lut" / "v5_unified_lut.csv"


class LutV5Table:
    """v5 unified semantic-quality table (task #28 v5, EPSILON_RECAL_V5.md change 2).

    Single source of truth read identically by the offline calibrator and the RL
    record layer.  Keyed by

        (question_type, service_level, channel_bin, snr_bin, view_quality_bin,
         count_bucket)

    channel_bin is derived at lookup time from the SNR label via
    channel_bin_from_snr (bad/medium/good) so the runtime query needs only the
    task's snr_bin.  Sparse cells already inherited their pooled parent Wilson
    interval offline; this loader adds runtime fallbacks (pool over
    view->count_bucket->parent) so an unseen key never crashes control.
    """

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._exact: dict[tuple, tuple[float, float, float, float]] = {}
        # parent pools (weighted by sample_count) for graceful fallback
        self._parent: dict[tuple, list[float]] = {}   # (qt,svc,ch,snr) -> [w*lcb, w*mean, w*pay, w]
        self._qsvc: dict[tuple, list[float]] = {}     # (qt,svc) -> [w*lcb, w*mean, w*pay, w]
        for r in rows:
            try:
                svc = int(r["service_level"])
                n = max(0, int(float(r.get("sample_count", 0) or 0)))
            except (KeyError, ValueError):
                continue
            qt = str(r.get("question_type", ""))
            ch = str(r.get("channel_bin", ""))
            snr = str(r.get("snr_bin", ""))
            view = str(r.get("view_quality_bin", ""))
            bucket = str(r.get("count_bucket", "na"))
            mean = float(r.get("expected_accuracy", 0.0) or 0.0)
            lcb = float(r.get("wilson_low", 0.0) or 0.0)
            pay_kb = float(r.get("avg_payload_bytes", 0.0) or 0.0) / 1024.0
            unc = max(0.0, mean - lcb)
            self._exact[(qt, svc, ch, snr, view, bucket)] = (mean, lcb, unc, pay_kb)
            w = float(max(n, 1))
            for store, key in ((self._parent, (qt, svc, ch, snr)), (self._qsvc, (qt, svc))):
                acc = store.setdefault(key, [0.0, 0.0, 0.0, 0.0])
                acc[0] += w * lcb
                acc[1] += w * mean
                acc[2] += w * pay_kb
                acc[3] += w

    @classmethod
    def from_csv(cls, path: Path) -> "LutV5Table":
        with Path(path).open(newline="", encoding="utf-8") as f:
            return cls(list(csv.DictReader(f)))

    @staticmethod
    def _pool(acc: list[float]) -> tuple[float, float, float, float] | None:
        w = acc[3]
        if w <= 0:
            return None
        lcb, mean, pay = acc[0] / w, acc[1] / w, acc[2] / w
        return mean, lcb, max(0.0, mean - lcb), pay

    def lookup(
        self,
        question_type: str,
        service_level: int,
        snr_bin: str,
        view_quality_bin: str,
        count_bucket: str,
    ) -> tuple[float, float, float, float] | None:
        try:
            snr_db = snr_db_from_label(snr_bin)
            channel = channel_bin_from_snr(snr_db)
        except (ValueError, TypeError):
            channel = "medium"
        qt, svc = str(question_type), int(service_level)
        view, bucket = str(view_quality_bin), str(count_bucket)
        hit = self._exact.get((qt, svc, channel, snr_bin, view, bucket))
        if hit is not None:
            return hit
        # fallback 1: pool over view for this (qt,svc,ch,snr,bucket)
        cand = [v for k, v in self._exact.items()
                if k[0] == qt and k[1] == svc and k[2] == channel and k[3] == snr_bin and k[5] == bucket]
        if cand:
            mean = sum(c[0] for c in cand) / len(cand)
            lcb = sum(c[1] for c in cand) / len(cand)
            pay = sum(c[3] for c in cand) / len(cand)
            return mean, lcb, max(0.0, mean - lcb), pay
        # fallback 2: parent pool (qt,svc,ch,snr) over view x bucket
        pooled = self._pool(self._parent.get((qt, svc, channel, snr_bin), [0, 0, 0, 0]))
        if pooled is not None:
            return pooled
        # fallback 3: (qt,svc) pool
        return self._pool(self._qsvc.get((qt, svc), [0, 0, 0, 0]))


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
    semantic_path: str
    task_status: str
    remaining_deadline_s: float
    defer_count: int
    cache_eligible: bool
    cache_quality_lcb: float
    cache_age: float
    selected_path_joint_feasible: bool
    selected_path_deadline_feasible: bool
    selected_path_utm_feasible: bool
    selected_path_deadline_slack_s: float
    selected_path_bottleneck_type: str
    expert_semantic_path: str
    expert_path_agreement: bool
    expert_mobility_mode: str
    expert_mobility_agreement: bool
    oracle_path_joint_feasible: bool
    oracle_mobility_joint_feasible: bool
    selected_mobility_joint_feasible: bool
    selected_mobility_deadline_feasible: bool
    selected_mobility_utm_feasible: bool
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
    rejected: bool
    reject_feasible: bool
    reject_reason: str
    expected_saved_energy_j: float
    expected_saved_delay_s: float
    avoided_utm_violation: bool
    avoided_deadline_violation: bool
    reject_penalty: float
    correct_reject: bool
    wrong_reject: bool
    q_quality: float
    q_deadline: float
    q_energy: float
    q_risk: float
    q_utm: float
    q_defer: float
    q_cache_stale: float
    success: bool
    delay_s: float
    fly_delay_s: float
    sense_delay_s: float
    tx_delay_s: float
    queue_delay_s: float
    infer_delay_s: float
    load_delay_s: float
    deadline_token_cache_fallback: bool
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
        num_uavs: int | None = None,
        quality_backend: str | None = None,
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
        # Optional UAV-count override for scalability/zero-shot sweeps.  It is
        # injected via reset options so it lands in the canonical env's
        # scalability layer, which is merged AFTER the scenario preset env
        # overrides (scenario presets pin their own num_uavs otherwise).
        self.num_uavs_override = int(num_uavs) if num_uavs else None
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
        # W3c/E4: pluggable quality source for the transmit services (1/2).
        # "lut" keeps the calibrated cell-table source; "persample" swaps the
        # accuracy/LCB/uncertainty fields for the calibrated per-sample
        # predictor (LCB = p - uncertainty), leaving payload estimates and the
        # cache/defer machinery untouched.
        self.quality_backend = str(quality_backend or rl_cfg.get("quality_backend", "lut")).lower()
        if self.quality_backend not in QUALITY_BACKENDS:
            raise ValueError(f"unknown quality_backend: {self.quality_backend} (expected {QUALITY_BACKENDS})")
        self._persample: PersamplePredictor | None = None
        self._persample_cache: dict[tuple[str, str, str, str, str], tuple[float, float]] = {}
        self._lut_v5: LutV5Table | None = None
        if self.quality_backend == "lut_v5":
            lut_v5_path = resolve_path(
                cfg.get("paths", {}).get("lut_v5_csv")
                or rl_cfg.get("lut_v5_path")
                or DEFAULT_LUT_V5
            )
            if not Path(lut_v5_path).exists():
                raise FileNotFoundError(
                    f"lut_v5 quality backend requested but table not found: {lut_v5_path} "
                    "(run scripts/build_lut_v5.py first)"
                )
            self._lut_v5 = LutV5Table.from_csv(lut_v5_path)
        if self.quality_backend == "persample":
            model_path = resolve_path(
                cfg.get("paths", {}).get("persample_model_json")
                or rl_cfg.get("persample_model_path")
                or DEFAULT_PERSAMPLE_MODEL
            )
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"persample quality backend requested but model not found: {model_path} "
                    "(run scripts/train_persample_predictor.py first)"
                )
            self._persample = PersamplePredictor.load(model_path)
        env_cfg = self._env_cfg.get("multi_uav_env", {}) if isinstance(self._env_cfg, dict) else {}
        self.energy_budget_j = float(rl_cfg.get("energy_budget_j", env_cfg.get("energy_budget_j", 500.0)))
        self._state_v2_service_levels = self._canonical_state_v2_service_levels(cfg)
        self._state_v2_max_uavs = max(
            int(rl_cfg.get("state_v2_max_uavs", 8)),
            int(env_cfg.get("num_uavs", 1)),
            int(self.num_uavs_override or 0),
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
        if self.num_uavs_override:
            options.setdefault("num_uavs", self.num_uavs_override)
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
        raw_concurrent = parsed.get("concurrent_actions", [])
        fallback_marker = str(parsed.get("sensing_decision", "")) == "deadline_token_cache_fallback" or any(
            isinstance(item, dict) and item.get("event") == "deadline_token_cache_fallback" for item in raw_concurrent
        )
        info = self._enrich_semantic_info(dict(info), prev_obs)
        info = self._attach_selected_path_metrics(info, prev_obs)
        info["deadline_token_cache_fallback"] = bool(fallback_marker or info.get("deadline_token_cache_fallback", False))
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
                "q_defer": record.q_defer,
                "q_cache_stale": record.q_cache_stale,
                "semantic_path": record.semantic_path,
                "task_status": record.task_status,
                "remaining_deadline_s": record.remaining_deadline_s,
                "defer_count": record.defer_count,
                "cache_eligible": record.cache_eligible,
                "cache_quality_lcb": record.cache_quality_lcb,
                "cache_age": record.cache_age,
                "success": record.success,
                "delay_s": record.delay_s,
                "fly_delay_s": record.fly_delay_s,
                "sense_delay_s": record.sense_delay_s,
                "tx_delay_s": record.tx_delay_s,
                "queue_delay_s": record.queue_delay_s,
                "infer_delay_s": record.infer_delay_s,
                "load_delay_s": record.load_delay_s,
                "deadline_token_cache_fallback": record.deadline_token_cache_fallback,
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

    def _critical_cache_compliance_forbidden(self) -> bool:
        """True iff the underlying env forbids s0 cache-only compliance for
        critical/high tasks (task #28 v3, method (c)).  Default "allowed" is a
        no-op, keeping legacy/v1/v2 record-layer output bit-identical."""
        try:
            mode = str(self._env.env_cfg.get("critical_cache_compliance", "allowed") or "allowed").lower()
        except Exception:
            mode = "allowed"
        return mode == "forbidden"

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
            semantic_path=str(info.get("semantic_path", "cache" if int(info.get("service_level", 0)) == 0 else "token")),
            task_status=str(info.get("task_status", "served" if bool(info.get("success", False)) else "pending")),
            remaining_deadline_s=float(info.get("remaining_deadline_s", info.get("deadline_s", 0.0))),
            defer_count=int(info.get("defer_count", 0)),
            cache_eligible=bool(info.get("cache_eligible", False)),
            cache_quality_lcb=float(info.get("cache_quality_lcb", 0.0)),
            cache_age=float(info.get("cache_age", 0.0)),
            selected_path_joint_feasible=bool(info.get("selected_path_joint_feasible", True)),
            selected_path_deadline_feasible=bool(info.get("selected_path_deadline_feasible", True)),
            selected_path_utm_feasible=bool(info.get("selected_path_utm_feasible", True)),
            selected_path_deadline_slack_s=float(info.get("selected_path_deadline_slack_s", 0.0)),
            selected_path_bottleneck_type=str(info.get("selected_path_bottleneck_type", "unknown")),
            expert_semantic_path=str(info.get("expert_semantic_path", "")),
            expert_path_agreement=bool(info.get("expert_path_agreement", False)),
            expert_mobility_mode=str(info.get("expert_mobility_mode", "")),
            expert_mobility_agreement=bool(info.get("expert_mobility_agreement", False)),
            oracle_path_joint_feasible=bool(info.get("oracle_path_joint_feasible", False)),
            oracle_mobility_joint_feasible=bool(info.get("oracle_mobility_joint_feasible", False)),
            selected_mobility_joint_feasible=bool(info.get("selected_mobility_joint_feasible", info.get("selected_path_joint_feasible", True))),
            selected_mobility_deadline_feasible=bool(info.get("selected_mobility_deadline_feasible", info.get("selected_path_deadline_feasible", True))),
            selected_mobility_utm_feasible=bool(info.get("selected_mobility_utm_feasible", info.get("selected_path_utm_feasible", True))),
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
            rejected=bool(info.get("rejected", False)),
            reject_feasible=bool(info.get("reject_feasible", False)),
            reject_reason=str(info.get("reject_reason", "")),
            expected_saved_energy_j=float(info.get("expected_saved_energy_j", 0.0)),
            expected_saved_delay_s=float(info.get("expected_saved_delay_s", 0.0)),
            avoided_utm_violation=bool(info.get("avoided_utm_violation", False)),
            avoided_deadline_violation=bool(info.get("avoided_deadline_violation", False)),
            reject_penalty=float(info.get("reject_penalty", 0.0)),
            correct_reject=bool(info.get("rejected", False)) and bool(info.get("reject_feasible", False)),
            wrong_reject=bool(info.get("rejected", False)) and not bool(info.get("reject_feasible", False)),
            q_quality=float(info.get("q_quality", 0.0)),
            q_deadline=float(info.get("q_deadline", 0.0)),
            q_energy=float(info.get("q_energy", 0.0)),
            q_risk=float(info.get("q_risk", 0.0)),
            q_utm=float(info.get("q_utm", 0.0)),
            q_defer=float(info.get("q_defer", 0.0)),
            q_cache_stale=float(info.get("q_cache_stale", 0.0)),
            success=bool(info.get("success", False)),
            delay_s=float(info.get("delay_s", info.get("total_delay_s", 0.0))),
            fly_delay_s=float(info.get("fly_delay_s", info.get("arrival_delay_s", 0.0))),
            sense_delay_s=float(info.get("sense_delay_s", 0.0)),
            tx_delay_s=float(info.get("tx_delay_s", info.get("transmission_delay_s", 0.0))),
            queue_delay_s=float(info.get("queue_delay_s", info.get("edge_queue_delay_s", info.get("edge_queue_s", 0.0)))),
            infer_delay_s=float(info.get("infer_delay_s", info.get("inference_delay_s", 0.0))),
            load_delay_s=float(info.get("load_delay_s", 0.0)),
            deadline_token_cache_fallback=bool(info.get("deadline_token_cache_fallback", False)),
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
        semantic_path = str(info.get("semantic_path", "cache" if int(info.get("service_level", 0)) == 0 else "token"))
        info["semantic_path"] = semantic_path
        model = self._semantic_utility
        if semantic_path == "defer":
            info["semantic_accuracy_mean"] = 0.0
            info["semantic_accuracy_lcb"] = 0.0
            info["semantic_uncertainty"] = 0.0
            info["semantic_sample_count"] = 0
            info["answer_accuracy_est"] = 0.0
            info["payload_kb"] = 0.0
            info["semantic_payload_kb"] = 0.0
        elif model is not None:
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
        if self._persample is not None and semantic_path != "defer":
            self._apply_persample_quality(info, obs, raw_accuracy)
        if self._lut_v5 is not None and semantic_path != "defer":
            self._apply_lut_v5_quality(info, obs, raw_accuracy)

        epsilon = self._epsilon_from_obs_or_info(info, obs)
        accuracy_lcb = float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0)))
        delay_s = float(info.get("delay_s", info.get("total_delay_s", 0.0)))
        deadline_s = self._deadline_from_obs_or_info(info, obs)
        energy_j = float(info.get("energy_j", info.get("total_energy_j", 0.0)))
        energy_budget_j = self._energy_budget_from_obs_or_info(info, obs)
        info.setdefault("semantic_payload_kb", float(info.get("payload_kb", 0.0)))
        info.setdefault("task_status", "served" if bool(info.get("success", False)) else "pending")
        info.setdefault("remaining_deadline_s", float(deadline_s))
        info.setdefault("defer_count", 0)
        info.setdefault("cache_eligible", False)
        info.setdefault("cache_quality_lcb", 0.0)
        info.setdefault("cache_age", 0.0)
        info["epsilon_k"] = float(epsilon)
        info["semantic_quality_gap"] = max(0.0, epsilon - accuracy_lcb)
        info["semantic_success"] = bool(accuracy_lcb >= epsilon)
        info["deadline_s"] = float(deadline_s)
        info["energy_budget_j"] = float(energy_budget_j)
        info["q_quality_increment"] = max(0.0, epsilon - accuracy_lcb)
        info["q_deadline_increment"] = max(0.0, delay_s - deadline_s)
        info["q_energy_increment"] = max(0.0, energy_j - energy_budget_j)
        info["quality_violation"] = bool(float(info.get("answer_accuracy_est", 0.0)) < epsilon)
        # Structural cache-compliance ban (task #28 v3, method (c)).  This is the
        # authoritative RL record layer: it recomputes semantic_success /
        # quality_violation from the calibrated LUT/persample estimate and
        # therefore CLOBBERS the env-level compliance override.  Re-apply the ban
        # here so a critical/high task served by the s0 cache-only path is never
        # counted as quality-compliant -- this propagates uniformly to the CSV
        # metrics, info["success"] below, the binary Lagrangian quality cost, and
        # the oracle_best_feasible_evidence feasibility search (which calls this
        # via evaluate_action).  The reward penalty and the env-side candidate
        # metrics get the same override from the env.  Default "allowed" no-op.
        if (
            str(info.get("semantic_path", "")) == "cache"
            and str(info.get("risk_level", "")) in ("critical", "high")
            and self._critical_cache_compliance_forbidden()
        ):
            info["semantic_success"] = False
            info["quality_violation"] = True
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
        info["q_defer_increment"] = float(str(info.get("semantic_path", "")) == "defer")
        cache_stale = str(info.get("semantic_path", "")) == "cache" and (not bool(info.get("cache_eligible", False)) or str(info.get("cache_freshness_bin", info.get("freshness_bin", "fresh"))) in {"stale", "expired"})
        info["q_cache_stale_increment"] = float(cache_stale) * max(1.0, float(info.get("semantic_quality_gap", 0.0)))
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

    def _attach_selected_path_metrics(self, info: dict[str, Any], obs: dict[str, Any] | None) -> dict[str, Any]:
        obs = obs or self._last_obs or {}
        path = str(info.get("semantic_path", "cache" if int(info.get("service_level", 0)) == 0 else "token"))
        metrics = obs.get("candidate_path_metrics", {}) if isinstance(obs.get("candidate_path_metrics", {}), dict) else {}
        data = metrics.get(path, {}) if isinstance(metrics.get(path, {}), dict) else {}
        if data:
            info["selected_path_joint_feasible"] = bool(data.get("joint_feasible", data.get("feasible", True)))
            info["selected_path_deadline_feasible"] = bool(data.get("deadline_feasible", not bool(info.get("deadline_violation", False))))
            info["selected_path_utm_feasible"] = bool(data.get("utm_feasible", not bool(info.get("utm_constraint_violation", False) or info.get("utm_conflict_violation", False))))
            info["selected_path_deadline_slack_s"] = float(data.get("deadline_slack_s", float(info.get("deadline_s", 0.0)) - float(info.get("delay_s", 0.0))))
            info["selected_path_bottleneck_type"] = str(data.get("bottleneck_type", data.get("bottleneck", "unknown")))
        else:
            info["selected_path_joint_feasible"] = not bool(
                info.get("quality_violation", False)
                or info.get("deadline_violation", False)
                or info.get("battery_violation", False)
                or info.get("resource_violation", False)
                or info.get("utm_constraint_violation", False)
                or info.get("utm_conflict_violation", False)
            )
            info["selected_path_deadline_feasible"] = not bool(info.get("deadline_violation", False))
            info["selected_path_utm_feasible"] = not bool(info.get("utm_constraint_violation", False) or info.get("utm_conflict_violation", False))
            info["selected_path_deadline_slack_s"] = float(info.get("deadline_s", 0.0)) - float(info.get("delay_s", 0.0))
            info["selected_path_bottleneck_type"] = "unknown"
        return info

    def _semantic_query(self, info: dict[str, Any], obs: dict[str, Any] | None) -> dict[str, Any]:
        obs = obs or self._last_obs or self.current_context or {}
        qtype = str(info.get("question_type", obs.get("question_type", obs.get("task_type", ""))))
        # v5 count bucket: prefer the explicit key; else derive from object_count.
        bucket = info.get("count_bucket", obs.get("count_bucket"))
        if bucket is None:
            if qtype == "counting":
                oc = info.get("object_count", obs.get("object_count", -1))
                try:
                    bucket = count_bucket_v5(int(oc))
                except (ValueError, TypeError):
                    bucket = "na"
            else:
                bucket = "na"
        return {
            "question_type": qtype,
            "service_level": int(info.get("service_level", self.service_levels[0])),
            "snr_bin": str(info.get("snr_bin", obs.get("snr_bin", ""))),
            "view_quality_bin": str(info.get("view_quality_bin", obs.get("view_quality_bin", "medium"))),
            "freshness_bin": str(info.get("freshness_bin", obs.get("freshness_bin", "fresh"))),
            "risk_level": str(info.get("risk_level", obs.get("risk_level", "normal"))),
            "count_bucket": str(bucket),
        }

    def _apply_lut_v5_quality(self, info: dict[str, Any], obs: dict[str, Any] | None,
                              raw_accuracy: float) -> None:
        """v5 quality backend: authoritative accuracy for transmit services (1/2).

        Reads the v5 unified LUT (qtype x service x channel x snr x view x count
        bucket) and overrides the accuracy/LCB/uncertainty/payload fields.  Cache
        (service 0) and defer keep their existing sources -- s0 runtime quality is
        governed by cache_quality=entry_v2 (change 3), not this table (its s0 rows
        are simulator_derived).  The LCB is what RL control consumes.
        """
        assert self._lut_v5 is not None
        service = int(info.get("service_level", 0))
        if service not in (1, 2):
            return
        query = self._semantic_query(info, obs)
        hit = self._lut_v5.lookup(
            query["question_type"], service, query["snr_bin"],
            query["view_quality_bin"], query["count_bucket"],
        )
        if hit is None:
            return
        mean, lcb, unc, payload_kb = hit
        info["semantic_accuracy_mean"] = float(mean)
        info["semantic_accuracy_lcb"] = float(lcb)
        info["semantic_uncertainty"] = float(unc)
        info["answer_accuracy_raw"] = raw_accuracy
        info["answer_accuracy_est"] = float(lcb)
        info["quality_backend"] = self.quality_backend
        if payload_kb > 0.0:
            info["payload_kb"] = float(payload_kb)
            info["semantic_payload_kb"] = float(payload_kb)

    def _apply_persample_quality(self, info: dict[str, Any], obs: dict[str, Any] | None,
                                 raw_accuracy: float) -> None:
        """E4: per-sample calibrated quality source for transmit services.

        Replaces the LUT/utility-model accuracy fields with the calibrated
        per-sample probability; the reliability-table uncertainty stands in
        for the Wilson-LCB gap (LCB = p - u).  Cache (service 0) and defer
        keep their existing sources so the cache-hit and risk-constraint
        machinery is untouched; payload estimates are also left as-is.
        """
        assert self._persample is not None
        service = str(int(info.get("service_level", 0)))
        if service not in self._persample.heads:
            return
        query = self._semantic_query(info, obs)
        snr_bin = str(query["snr_bin"])
        sensed = float(info.get("sensed_snr_db", (obs or {}).get("sensed_snr_db", 0.0)) or 0.0)
        snr_key = snr_bin if snr_bin else f"{round(sensed, 1)}snr"
        key = (query["question_type"], service, snr_key,
               query["view_quality_bin"], query["risk_level"])
        cached = self._persample_cache.get(key)
        if cached is None:
            record = {
                "question_type": query["question_type"],
                "view_quality_bin": query["view_quality_bin"],
                "risk_level": query["risk_level"],
                "snr_bin": snr_bin,
                "sensed_snr_db": sensed,
            }
            proba = float(self._persample.predict_proba([record], service)[0])
            uncertainty = float(self._persample.uncertainty([record], service)[0])
            cached = (proba, uncertainty)
            self._persample_cache[key] = cached
        proba, uncertainty = cached
        lcb = max(0.0, min(1.0, proba - uncertainty))
        info["semantic_accuracy_mean"] = proba
        info["semantic_accuracy_lcb"] = lcb
        info["semantic_uncertainty"] = uncertainty
        info["answer_accuracy_raw"] = raw_accuracy
        info["answer_accuracy_est"] = lcb
        info["quality_backend"] = self.quality_backend

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
        features += self._semantic_path_feature_vector(obs)
        features += self._uav_mobility_feature_vector(obs)
        features += self._mask_feature_vector(obs)
        return features


    def _semantic_path_feature_vector(self, obs: dict[str, Any]) -> list[float]:
        paths = ("cache", "token", "image", "defer", "cache_update")
        metrics = obs.get("candidate_path_metrics", {}) if isinstance(obs.get("candidate_path_metrics", {}), dict) else {}
        deadline = max(1e-6, float(obs.get("remaining_deadline_s", obs.get("deadline_s", obs.get("tau_k", 5.0)))))
        features: list[float] = []
        for path in paths:
            data = metrics.get(path, {}) if isinstance(metrics.get(path, {}), dict) else {}
            accuracy_lcb = self._clip01(float(data.get("accuracy_lcb", 0.0)))
            accuracy_mean = self._clip01(float(data.get("accuracy_mean", accuracy_lcb)))
            gap = self._clip01(float(data.get("quality_gap", max(0.0, float(obs.get("epsilon_k", 0.0)) - accuracy_lcb))))
            delay_s = max(0.0, float(data.get("delay_s", 0.0)))
            energy_j = max(0.0, float(data.get("energy_j", 0.0)))
            payload_kb = max(0.0, float(data.get("payload_kb", 0.0)))
            semantic_feasible = float(gap <= 1e-9)
            deadline_feasible = float(delay_s <= deadline)
            feasible = float(bool(data.get("feasible", semantic_feasible > 0.0 and deadline_feasible > 0.0)))
            slack_ratio = self._clip(float(data.get("deadline_slack_s", deadline - delay_s)) / deadline, -1.0, 1.0)
            features.extend(
                [
                    feasible,
                    accuracy_lcb,
                    accuracy_mean,
                    gap,
                    semantic_feasible,
                    deadline_feasible,
                    float(semantic_feasible > 0.0 and deadline_feasible > 0.0),
                    slack_ratio,
                    self._clip01(payload_kb / 200.0),
                    self._clip01(energy_j / self._state_v2_energy_norm_j),
                    float(bool(data.get("cache_eligible", False))),
                    float(bool(data.get("utm_constraint_violation", False))),
                ]
            )
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
        path_allowed = mask.get("semantic_path_allowed", {}) if isinstance(mask.get("semantic_path_allowed", {}), dict) else {}
        path_values = [float(bool(path_allowed.get(path, True))) for path in ("cache", "token", "image", "defer", "cache_update")]
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
        return path_values + service_values + mobility_values + uav_values

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
        return {"quality": 0.0, "deadline": 0.0, "energy": 0.0, "risk": 0.0, "utm": 0.0, "defer": 0.0, "cache_stale": 0.0}

    def _queue_vector(self) -> list[float]:
        return [
            float(self._queue_state["quality"]),
            float(self._queue_state["deadline"]),
            float(self._queue_state["energy"]) / max(1.0, float(self.energy_budget_j)),
            float(self._queue_state["risk"]),
            float(self._queue_state["utm"]),
            float(self._queue_state["defer"]),
            float(self._queue_state["cache_stale"]),
        ]

    def _attach_queue_state(self, info: dict[str, Any]) -> None:
        info["q_quality"] = float(self._queue_state["quality"])
        info["q_deadline"] = float(self._queue_state["deadline"])
        info["q_energy"] = float(self._queue_state["energy"])
        info["q_risk"] = float(self._queue_state["risk"])
        info["q_utm"] = float(self._queue_state["utm"])
        info["q_defer"] = float(self._queue_state["defer"])
        info["q_cache_stale"] = float(self._queue_state["cache_stale"])

    def _update_virtual_queues(self, info: dict[str, Any]) -> None:
        self._queue_state["quality"] = max(0.0, self._queue_state["quality"] + float(info.get("q_quality_increment", 0.0)))
        self._queue_state["deadline"] = max(0.0, self._queue_state["deadline"] + float(info.get("q_deadline_increment", 0.0)))
        self._queue_state["energy"] = max(0.0, self._queue_state["energy"] + float(info.get("q_energy_increment", 0.0)))
        self._queue_state["risk"] = max(0.0, self._queue_state["risk"] + float(info.get("q_risk_increment", 0.0)))
        self._queue_state["utm"] = max(0.0, self._queue_state["utm"] + float(info.get("q_utm_increment", 0.0)))
        self._queue_state["defer"] = max(0.0, self._queue_state["defer"] + float(info.get("q_defer_increment", 0.0)))
        self._queue_state["cache_stale"] = max(0.0, self._queue_state["cache_stale"] + float(info.get("q_cache_stale_increment", 0.0)))
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
