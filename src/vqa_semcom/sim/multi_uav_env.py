from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.sim import bubbles_separation
from vqa_semcom.semantic.utility import SemanticUtilityEstimate, SemanticUtilityModel
from vqa_semcom.sim.resource_env import LUTEntry, load_lut, read_csv
from vqa_semcom.snr import channel_bin_from_snr, snr_bins_from_config, snr_db_from_label, snr_db_to_bin_label


# v5 count-bucket dimension (task #28 v5, EPSILON_RECAL_V5.md change 2).  MUST
# match scripts/build_lut_v5.py::COUNT_BUCKETS bit-for-bit so the runtime
# lut_v5 quality lookup and the offline LUT builder agree on the counting
# key.  Non-counting question types carry the sentinel "na".
COUNT_BUCKETS_V5 = [(1, 4, "1-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, 49, "20-49"), (50, 10 ** 9, "50+")]


def count_bucket_v5(gt: int) -> str:
    for lo, hi, label in COUNT_BUCKETS_V5:
        if lo <= gt <= hi:
            return label
    if gt <= 0:
        return "0"
    return "50+"


def _safe_int(text: Any, default: int = -1) -> int:
    try:
        s = str(text).strip().lower()
        if not s or s in ("unknown", "none", "nan"):
            return default
        return int(float(s))
    except (ValueError, TypeError):
        return default


@dataclass(frozen=True)
class Area4D:
    center_x_m: float
    center_y_m: float
    radius_m: float
    altitude_min_m: float
    altitude_max_m: float
    start_step: int
    end_step: int

    def distance_to(self, x_m: float, y_m: float) -> float:
        return math.hypot(self.center_x_m - x_m, self.center_y_m - y_m)

    def overlaps(self, other: "Area4D") -> bool:
        spatial = self.distance_to(other.center_x_m, other.center_y_m) <= self.radius_m + other.radius_m
        altitude = self.altitude_min_m <= other.altitude_max_m and other.altitude_min_m <= self.altitude_max_m
        temporal = self.start_step <= other.end_step and other.start_step <= self.end_step
        return spatial and altitude and temporal


@dataclass
class UAVNode:
    uav_id: int
    x_m: float
    y_m: float
    altitude_m: float
    battery_j: float
    speed_mps: float
    base_x_m: float = 0.0
    base_y_m: float = 0.0
    base_altitude_m: float = 90.0
    current_task_id: str = ""
    camera_state: str = "idle"
    total_flight_m: float = 0.0
    utilization: float = 0.0


@dataclass
class EdgeNode:
    edge_id: int
    load: float
    gpu_load: float
    cached_service_levels: tuple[int, ...]
    model_cache_capacity: int
    gpu_memory_capacity_mb: float
    gpu_memory_used_mb: float


@dataclass(frozen=True)
class SemanticCacheEntry:
    task_id: str
    task_type: str
    risk_level: str
    priority: float
    x_m: float
    y_m: float
    cache_age: int
    updated_step: int
    area_id: int = -1
    question_type: str = ""
    quality_lcb: float = 0.0
    uncertainty: float = 0.0
    reuse_count: int = 0


@dataclass
class EnvTask:
    task_id: str
    task_type: str
    question: str
    risk_level: str
    epsilon_k: float
    tau_k: float
    priority: float
    view_quality_bin: str
    freshness_bin: str
    generation_time: int
    area_id: int
    area4d: Area4D
    object_count: int = -1
    spec_attainable: bool = False
    escalated: bool = False
    completed: bool = False
    task_status: str = "pending"
    defer_count: int = 0
    expired: bool = False
    rejected: bool = False
    cache_age: int = 0
    last_sensed_snr_db: float = 0.0
    last_sinr_db: float = 0.0
    last_snr_bin: str = ""
    operational_intent_id: str = ""
    operational_intent_state: str = "accepted"
    operational_priority: float = 1.0

    @property
    def x_m(self) -> float:
        return self.area4d.center_x_m

    @property
    def y_m(self) -> float:
        return self.area4d.center_y_m


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# Attainability-anchored semantic-quality thresholds (task #28). See
# docs_spec/EPSILON_RECAL.md: epsilon_critical = 0.90 x peak-condition oracle
# LCB ceiling (0.6835) => 0.615; epsilon_normal = 0.75 x nominal-condition
# oracle LCB ceiling (0.2209) => 0.166. Selected via
# env_cfg["epsilon_calibration"] == "attainability_v1"; default "legacy"
# preserves the original 0.82/0.65 constants bit-for-bit.
ATTAINABILITY_V1_EPSILON: dict[str, float] = {
    "critical": 0.615,
    "normal": 0.166,
    "high": 0.615,
}


# Attainability recalibration iteration 2 (task #28). See
# docs/EPSILON_RECAL_V2.md.  Rule (1) quantile anchoring: eps_critical =
# P10 of the peak (all-critical) best-feasible-service LCB distribution
# (=0.504); eps_normal = P25 of the nominal normal-risk distribution
# (=0.297).  Rule (2) cache-ceiling guardrail: eps_critical =
# max(P10 anchor, cache_accuracy_P90 + 0.05) = max(0.504, 0.633) = 0.633
# so the cache cannot become a critical-task compliance shortcut.
# NOTE: the guardrail floor (0.633) binds ABOVE the attainability anchor
# (0.504) -- flagged in the doc as a rule conflict for the counting-heavy
# peak mix; the constants below are the literal output of the v2 rule.
# Selected via env_cfg["epsilon_calibration"] == "attainability_v2".
ATTAINABILITY_V2_EPSILON: dict[str, float] = {
    "critical": 0.633,
    "normal": 0.297,
    "high": 0.633,
}


# Attainability recalibration iteration 3 (task #28, method (c): structural
# cache-compliance ban).  See docs/EPSILON_RECAL_V3.md.  v2 proved rule (1)
# quantile anchoring and rule (2) cache-ceiling guardrail mutually
# contradictory for the counting-heavy peak mix: any eps_critical that defeats
# the cache shortcut (>0.633) is unattainable, any attainable one (<=0.504) is
# below the cache ceiling.  v3 DROPS the guardrail and instead forbids cache-
# ONLY compliance for critical/high tasks *structurally* -- gated by
# env_cfg["critical_cache_compliance"] == "forbidden" (default "allowed"
# preserves legacy/v1/v2 bit-for-bit).  With the cache shortcut removed at the
# compliance-judgment layer (not just the reward), eps_critical returns to the
# pure attainability anchor P10=0.504, and eps_normal is restored to the v1
# anchor 0.166 (v2's 0.297 caused a pure nominal regression 0.897 -> 0.846).
# Selected via env_cfg["epsilon_calibration"] == "attainability_v3".
ATTAINABILITY_V3_EPSILON: dict[str, float] = {
    "critical": 0.504,
    "normal": 0.166,
    "high": 0.504,
}


# Attainability recalibration iteration 4 (task #28): TRANSMISSION-ONLY anchor.
# See docs/EPSILON_RECAL_V4.md and scripts/calibrate_epsilon_v4.py.  v3's 0.504
# was the v2 P10 of the oracle realised best-*feasible*-service LCB, a
# distribution that STILL contained the s0 cache in the feasible set.  Once cache
# is compliance-banned (v3), ~44% of the peak all-critical mix (counting) has no
# transmission path reaching 0.504, so the peak oracle collapses (semSucc 0.556)
# and every learning arm collapses (semSucc 0.008, lambda pinned).  v4 re-anchors
# on the PURE-TRANSMISSION feasible set: eps_critical = floor_3dp(P10) of the
# per-task best-transmission LCB max(token, image) over the peak all-critical mix
# with s0 cache excluded = 0.355 (P10 = 0.355225, counting cluster; rounded DOWN
# so the P10 cluster stays >= threshold).  eps_normal held at the v1/v3
# attainability anchor 0.166 (nominal normal unaffected by the cache ban).  Pure-
# tx anchor 0.355 <= cache-inclusive v3 anchor 0.504 (invariant).  Pair with
# env_cfg["critical_cache_compliance"] == "forbidden".
# Selected via env_cfg["epsilon_calibration"] == "attainability_v4".
ATTAINABILITY_V4_EPSILON: dict[str, float] = {
    "critical": 0.355,
    "normal": 0.166,
    "high": 0.355,
}


# Attainability recalibration iteration 5 (task #28 v5): ATTAINABLE EPSILON on the
# v5 UNIFIED LUT + escalation layer.  See docs/EPSILON_RECAL_V5.md and
# scripts/calibrate_epsilon_v5.py.  v4's single pooled P10 (0.355) flattened the
# two structurally different critical clusters (hard wide-tolerance counting vs
# easy yes/no presence) into one bar, and could not separate the quality axis
# from the deadline axis (v4 collapsed: 99.5% of peak critical charged as quality
# violations that were actually physically-infeasible expired tasks).  v5:
#   1. Splits the critical bar by qtype into a TWO-KEY (risk x qtype) table.
#   2. Anchors each key on the v5 LUT (lut_v5 backend, GT>=10 counting re-judged
#      +-20%) via eps = floor_3dp(P25(best_tx_lcb) - 0.05):
#        (critical, counting GT>=10) -> P25 0.5144 -> 0.464   [quality-axis
#          unreachable only 0.083 -> genuinely attainable]
#        (critical, presence/other)  -> P25 0.7462 -> 0.696
#        (normal, *)                 -> P25 0.5796 -> 0.529   (all-qtype pool)
#   3. Pairs with the escalation layer (change 5): critical reject/expired tasks
#      that are spec-UNattainable (no tx service both clears eps AND is deadline-
#      feasible against full tau_k) are ESCALATED, not charged as quality
#      violations -- this un-pins lambda_critical.
# Keyed by (risk, qtype_class); qtype_class = "counting" iff counting with a
# GT>=10 count bucket, else "presence".  normal uses one bar for all qtypes.
# Selected via env_cfg["epsilon_calibration"] == "attainability_v5".  Pair with
# --critical-cache-compliance forbidden and --quality-backend lut_v5.
ATTAINABILITY_V5_GE10_BUCKETS = ("10-19", "20-49", "50+")
ATTAINABILITY_V5_EPSILON: dict[tuple[str, str], float] = {
    ("critical", "counting"): 0.464,
    ("critical", "presence"): 0.696,
    ("high", "counting"): 0.464,
    ("high", "presence"): 0.696,
    ("normal", "counting"): 0.529,
    ("normal", "presence"): 0.529,
}
# Escalation dual-channel budgets delta_esc (change 5/escalation channel): the
# measured spec-UNattainable fraction + 0.05, condition-adaptive.
#
# v6 UPDATE (task #33/#34): under the LEGACY full-flight deadline these values
# reflected a UAV-flight-deadline CLIFF -- peak spec-unattainable ~1.0 because a
# critical tau_k of 2.55 s (3.0 x tau_scale 0.85) can never fit the ~13 s flight
# to the task area.  That cliff is a scale ARTEFACT, not physics: BUBBLES puts
# flight/positioning in the tasking layer and reserves the T4 tactical-comm
# window (separation-communication N(1.8, sigma 1.0), Table G-2) for the
# decision loop.  Under deadline_semantics="comm_window" the flight term is
# removed from the deadline clock, so the escalation budget is re-derived by
# scripts/calibrate_epsilon_v6.py and SHIPPED IN THE CALIBRATION JSON (single
# estimator, task #34-(i)).  The dict below is only the legacy fallback used
# when no calibration JSON is supplied; do NOT hand-tune it.
ESCALATION_DELTA_V5: dict[str, float] = {"peak": 0.90, "nominal": 0.50}

# v6 comm-window deadline anchors (task #33-B).  tau_k under
# deadline_semantics="comm_window" is the tactical COMMUNICATION-DECISION window
# only (sense + tx + queue + infer + load); flight/positioning is charged to the
# tasking layer, not the deadline clock.  Anchored on BUBBLES D2.1 Table G-2
# separation-communication delay N(mean 1.8 s, sigma 1.0 s):
#   * tau_critical = 1.8 + 1*sigma = 2.8 s  (1-sigma / 0.841 confidence)
#   * tau_normal   = 1.8 + 2*sigma = 3.8 s  (2-sigma / 0.977 confidence)
# Overridable from configs/*.yaml thresholds {tau_critical_comm, tau_normal_comm}.
# tau_scale is NOT applied under comm_window (the anchor already IS the window).
COMM_WINDOW_TAU: dict[str, float] = {"critical": 2.8, "normal": 3.8, "high": 2.8}


def epsilon_v5_class(qtype: str, count_bucket: str) -> str:
    """(risk, qtype)-table column: counting-critical bucket vs presence-like."""
    if str(qtype) == "counting" and str(count_bucket) in ATTAINABILITY_V5_GE10_BUCKETS:
        return "counting"
    return "presence"


SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "nominal": {
        "description": "Default mixed task queue with calibrated physical constants.",
        "env": {},
        "task_layout": {},
    },
    "conflict-heavy": {
        "description": "Burst tasks share overlapping Area4D operational volumes to stress airspace conflict checks.",
        "env": {
            "num_uavs": 3,
            "num_areas": 1,
            "tasks_per_episode": 18,
            "episode_steps": 12,
            "area_spacing_m": 80.0,
            "area_radius_m": 130.0,
            "area_altitude_min_m": 45.0,
            "area_altitude_max_m": 115.0,
            "reward_conflict": 2.0,
        },
        "task_layout": {
            "generation_mode": "burst",
            "force_same_area": True,
            "jitter_ratio": 0.03,
            "risk_cycle": ["critical", "normal", "normal"],
            "freshness_cycle": ["stale", "fresh", "expired"],
            "tau_scale": 0.9,
        },
    },
    "interference-heavy": {
        "description": "Concurrent same-band uploads with stronger interference overlap and fast fading.",
        "env": {
            "num_uavs": 4,
            "num_areas": 2,
            "tasks_per_episode": 18,
            "episode_steps": 12,
            "area_spacing_m": 300.0,
            "area_radius_m": 65.0,
            "bandwidth_hz": 700_000.0,
            "a2g": {
                "fading_mode": "fast_fading",
                "fast_fading_std_db": 6.0,
                "interference_enabled": True,
                "interference_floor_dbm": -116.0,
                "interference_overlap_scale": 0.32,
            },
        },
        "task_layout": {
            "generation_mode": "burst",
            "jitter_ratio": 0.10,
            "freshness_cycle": ["stale", "fresh"],
            "tau_scale": 1.0,
        },
    },
    "cache-heavy": {
        "description": "Spatially repeated fresh/stale tasks with seeded semantic cache entries.",
        "env": {
            "num_uavs": 2,
            "num_areas": 2,
            "tasks_per_episode": 16,
            "episode_steps": 12,
            "area_spacing_m": 120.0,
            "area_radius_m": 95.0,
            "semantic_cache_capacity": 96,
            "semantic_cache_radius_m": 220.0,
            "semantic_cache_reuse_boost": 0.26,
            "cache_hit_probability": {"fresh": 0.98, "stale": 0.78, "expired": 0.36},
        },
        "task_layout": {
            "generation_mode": "wave",
            "jitter_ratio": 0.05,
            "freshness_cycle": ["fresh", "fresh", "stale", "expired"],
            "view_quality_cycle": ["good", "medium", "good", "medium"],
        },
        "semantic_cache_seed": {
            "enabled": True,
            "entries_per_area": 2,
            "cache_age": 0,
        },
    },
    "mobility-stress": {
        "description": "Sparse disaster areas, lower speed, tighter battery reserve, and longer flight legs.",
        "env": {
            "num_uavs": 2,
            "num_areas": 6,
            "tasks_per_episode": 18,
            "episode_steps": 16,
            "area_spacing_m": 650.0,
            "area_radius_m": 80.0,
            "uav_speed_mps": 12.0,
            "initial_battery_j": 9_000.0,
            "return_energy_reserve_j": 1_200.0,
            "flight_energy_j_per_m": 10.5,
            "hover_power_w": 88.0,
        },
        "task_layout": {
            "generation_mode": "wave",
            "jitter_ratio": 0.18,
            "risk_cycle": ["normal", "critical", "normal"],
            "freshness_cycle": ["stale", "expired", "fresh"],
            "tau_scale": 1.25,
        },
    },
    "nominal_patrol": {
        "description": "Paper preset: routine multi-UAV patrol with medium SNR, low UTM pressure, and medium semantic QoS targets.",
        "env": {
            "num_uavs": 3,
            "num_edges": 1,
            "num_areas": 4,
            "tasks_per_episode": 20,
            "episode_steps": 12,
            "area_spacing_m": 260.0,
            "area_radius_m": 70.0,
            "bandwidth_hz": 1_000_000.0,
            "edge_load_range": [0.18, 0.36],
            "gpu_load_range": [0.14, 0.30],
            "semantic_threshold_by_risk": {"normal": 0.58, "critical": 0.76, "high": 0.76},
            "a2g": {
                "fading_mode": "slow_fading",
                "slow_fading_std_db": 1.5,
                "excess_loss_db": 4.0,
                "los_excess_loss_db": 1.0,
                "nlos_excess_loss_db": 10.0,
                "interference_overlap_scale": 0.02,
            },
            "utm": {
                "enabled": True,
                "mode": "nominal_planning",
                "dss_available": True,
                "dss_delay_s": 0.04,
                "subscription_notification_delay_s": 0.06,
                "spatial_buffer_m": 15.0,
                "altitude_buffer_m": 0.0,
                "temporal_buffer_steps": 0,
            },
        },
        "task_layout": {
            "generation_mode": "staggered",
            "jitter_ratio": 0.14,
            "risk_cycle": ["normal", "normal", "critical", "normal"],
            "freshness_cycle": ["fresh", "stale", "fresh", "stale"],
            "view_quality_cycle": ["medium", "good", "medium", "good"],
            "tau_scale": 1.0,
        },
    },
    "disaster_hotspot": {
        "description": "Paper preset: clustered high-risk VQA burst with stricter epsilon_k and tighter deadlines.",
        "env": {
            "num_uavs": 4,
            "num_edges": 1,
            "num_areas": 2,
            "tasks_per_episode": 24,
            "episode_steps": 10,
            "area_spacing_m": 90.0,
            "area_radius_m": 120.0,
            "edge_load_range": [0.32, 0.56],
            "gpu_load_range": [0.26, 0.50],
            "semantic_threshold_by_risk": {"normal": 0.62, "critical": 0.84, "high": 0.84},
            "reward_violation": 1.8,
            "reward_conflict": 1.5,
            "utm": {
                "enabled": True,
                "mode": "nominal_planning",
                "dss_available": True,
                "dss_delay_s": 0.08,
                "subscription_notification_delay_s": 0.16,
                "spatial_buffer_m": 30.0,
                "temporal_buffer_steps": 1,
            },
        },
        "task_layout": {
            "generation_mode": "burst",
            "force_same_area": True,
            "jitter_ratio": 0.04,
            "risk_cycle": ["critical", "critical", "normal", "critical", "critical"],
            "freshness_cycle": ["stale", "expired", "fresh", "stale"],
            "view_quality_cycle": ["medium", "poor", "medium", "good"],
            "tau_scale": 0.58,
        },
    },
    "low_snr_blockage": {
        "description": "Paper preset: weak A2G links and blockage stress where cache/tokens should be robust against image payload delay.",
        "env": {
            "num_uavs": 3,
            "num_edges": 1,
            "num_areas": 5,
            "tasks_per_episode": 20,
            "episode_steps": 12,
            "area_spacing_m": 520.0,
            "area_radius_m": 80.0,
            "uav_altitude_m": 70.0,
            "bandwidth_hz": 650_000.0,
            "semantic_cache_capacity": 80,
            "semantic_cache_radius_m": 240.0,
            "semantic_cache_reuse_boost": 0.22,
            "cache_hit_probability": {"fresh": 0.96, "stale": 0.72, "expired": 0.30},
            "semantic_threshold_by_risk": {"normal": 0.56, "critical": 0.78, "high": 0.78},
            "a2g": {
                "path_loss_exponent": 2.9,
                "excess_loss_db": 18.0,
                "los_excess_loss_db": 5.0,
                "nlos_excess_loss_db": 28.0,
                "fading_mode": "slow_fading",
                "slow_fading_std_db": 4.5,
                "fading_correlation": 0.70,
                "interference_overlap_scale": 0.06,
            },
        },
        "task_layout": {
            "generation_mode": "wave",
            "jitter_ratio": 0.16,
            "risk_cycle": ["normal", "critical", "normal", "normal"],
            "freshness_cycle": ["fresh", "stale", "expired", "fresh"],
            "view_quality_cycle": ["medium", "poor", "medium", "good"],
            "tau_scale": 1.1,
        },
        "semantic_cache_seed": {
            "enabled": True,
            "entries_per_area": 1,
            "cache_age": 0,
        },
    },
    "low_snr_soft": {
        "description": "Paper preset: moderate weak-link scenario calibrated so path-greedy/oracle policies retain a non-trivial feasible region.",
        "env": {
            "num_uavs": 3,
            "num_edges": 1,
            "num_areas": 4,
            "tasks_per_episode": 20,
            "episode_steps": 12,
            "area_spacing_m": 380.0,
            "area_radius_m": 75.0,
            "uav_altitude_m": 80.0,
            "uav_speed_mps": 34.0,
            "bandwidth_hz": 1_000_000.0,
            "semantic_cache_capacity": 80,
            "semantic_cache_radius_m": 220.0,
            "semantic_cache_reuse_boost": 0.20,
            "cache_hit_probability": {"fresh": 0.96, "stale": 0.72, "expired": 0.30},
            "semantic_threshold_by_risk": {"normal": 0.48, "critical": 0.70, "high": 0.70},
            "a2g": {
                "path_loss_exponent": 2.65,
                "excess_loss_db": 12.0,
                "los_excess_loss_db": 4.0,
                "nlos_excess_loss_db": 22.0,
                "fading_mode": "slow_fading",
                "slow_fading_std_db": 3.0,
                "fading_correlation": 0.75,
                "interference_overlap_scale": 0.04,
            },
        },
        "task_layout": {
            "generation_mode": "wave",
            "jitter_ratio": 0.14,
            "risk_cycle": ["normal", "critical", "normal", "normal"],
            "freshness_cycle": ["fresh", "stale", "fresh", "expired"],
            "view_quality_cycle": ["medium", "medium", "good", "medium"],
            "tau_scale": 1.55,
            "tau_floor_s": 8.5,
            "epsilon_scale": 0.85,
            "epsilon_cap_by_risk": {"normal": 0.52, "critical": 0.70, "high": 0.70},
        },
        "semantic_cache_seed": {
            "enabled": True,
            "entries_per_area": 1,
            "cache_age": 0,
        },
    },
    "edge_overload": {
        "description": "Paper preset: high edge CPU/GPU load and model-cache pressure for queue/resource-projection tests.",
        "env": {
            "num_uavs": 4,
            "num_edges": 1,
            "num_areas": 4,
            "tasks_per_episode": 24,
            "episode_steps": 12,
            "area_spacing_m": 170.0,
            "area_radius_m": 75.0,
            "uav_speed_mps": 24.0,
            "edge_load_range": [0.56, 0.76],
            "gpu_load_range": [0.52, 0.72],
            "queue_delay_scale_s": 0.62,
            "gpu_queue_delay_scale_s": 0.48,
            "model_load_delay_s": 0.55,
            "model_cache_hit_delay_s": 0.08,
            "model_cache_capacity": 1,
            "gpu_memory_capacity_mb": 4096.0,
            "gpu_memory_load": 0.48,
            "semantic_threshold_by_risk": {"normal": 0.54, "critical": 0.64, "high": 0.64},
            "semantic_threshold_cap_by_risk": {"normal": 0.62, "critical": 0.64, "high": 0.64},
        },
        "task_layout": {
            "generation_mode": "wave",
            "jitter_ratio": 0.12,
            "risk_cycle": ["normal", "normal", "normal", "critical", "normal", "normal"],
            "freshness_cycle": ["fresh", "fresh", "fresh", "stale"],
            "view_quality_cycle": ["good", "good", "medium", "good"],
            "tau_scale": 1.55,
        },
    },
    "edge_overload_soft": {
        "description": "Calibrated soft edge-overload preset with a non-zero feasible region while preserving edge pressure.",
        "env": {
            "num_uavs": 4,
            "num_edges": 1,
            "num_areas": 4,
            "tasks_per_episode": 22,
            "episode_steps": 12,
            "area_spacing_m": 150.0,
            "area_radius_m": 75.0,
            "uav_speed_mps": 26.0,
            "edge_load_range": [0.38, 0.58],
            "gpu_load_range": [0.34, 0.54],
            "queue_delay_scale_s": 0.42,
            "gpu_queue_delay_scale_s": 0.30,
            "model_load_delay_s": 0.42,
            "model_cache_hit_delay_s": 0.06,
            "model_cache_capacity": 2,
            "gpu_memory_capacity_mb": 6144.0,
            "gpu_memory_load": 0.34,
            "semantic_threshold_by_risk": {"normal": 0.50, "critical": 0.60, "high": 0.60},
            "semantic_threshold_cap_by_risk": {"normal": 0.58, "critical": 0.60, "high": 0.60},
        },
        "task_layout": {
            "generation_mode": "wave",
            "jitter_ratio": 0.10,
            "risk_cycle": ["normal", "normal", "normal", "critical", "normal"],
            "freshness_cycle": ["fresh", "fresh", "stale", "fresh"],
            "view_quality_cycle": ["good", "medium", "good", "good"],
            "tau_scale": 1.45,
            "tau_floor_s": 6.5,
            "epsilon_scale": 0.92,
            "epsilon_cap_by_risk": {"normal": 0.56, "critical": 0.60, "high": 0.60},
        },
    },
    "utm_conflict": {
        "description": "Paper preset: UTM strategic conflict with DSS/subscription delay and observable intent states.",
        "env": {
            "num_uavs": 4,
            "num_edges": 1,
            "num_areas": 3,
            "tasks_per_episode": 20,
            "episode_steps": 10,
            "area_spacing_m": 260.0,
            "area_radius_m": 90.0,
            "area_altitude_min_m": 45.0,
            "area_altitude_max_m": 115.0,
            "reward_conflict": 2.4,
            "semantic_threshold_by_risk": {"normal": 0.60, "critical": 0.82, "high": 0.82},
            "utm": {
                "enabled": True,
                "mode": "flight_intent_validation",
                "background_operational_intents": True,
                "background_operational_intent_density": 0.30,
                "dss_available": True,
                "dss_delay_s": 0.12,
                "subscription_notification_delay_s": 0.55,
                "spatial_buffer_m": 28.0,
                "altitude_buffer_m": 8.0,
                "temporal_buffer_steps": 1,
            },
        },
        "task_layout": {
            "generation_mode": "burst",
            "force_same_area": False,
            "jitter_ratio": 0.04,
            "risk_cycle": ["critical", "normal", "critical", "normal"],
            "freshness_cycle": ["stale", "fresh", "expired", "stale"],
            "view_quality_cycle": ["medium", "good", "medium", "poor"],
            "operational_state_cycle": ["accepted", "activated", "nonconforming", "contingent"],
            "tau_scale": 0.85,
        },
    },
    "utm_conflict_soft": {
        "description": "Calibrated soft UTM preset with lower background conflict density and observable safe service routes.",
        "env": {
            "num_uavs": 4,
            "num_edges": 1,
            "num_areas": 4,
            "tasks_per_episode": 20,
            "episode_steps": 12,
            "area_spacing_m": 280.0,
            "area_radius_m": 70.0,
            "area_altitude_min_m": 45.0,
            "area_altitude_max_m": 120.0,
            "reward_conflict": 2.0,
            "semantic_threshold_by_risk": {"normal": 0.52, "critical": 0.70, "high": 0.70},
            "semantic_threshold_cap_by_risk": {"normal": 0.58, "critical": 0.70, "high": 0.70},
            "utm": {
                "enabled": True,
                "mode": "flight_intent_validation",
                "background_operational_intents": True,
                "background_operational_intent_density": 0.12,
                "dss_available": True,
                "dss_delay_s": 0.08,
                "subscription_notification_delay_s": 0.22,
                "spatial_buffer_m": 16.0,
                "altitude_buffer_m": 4.0,
                "temporal_buffer_steps": 0,
            },
        },
        "task_layout": {
            "generation_mode": "wave",
            "force_same_area": False,
            "jitter_ratio": 0.08,
            "risk_cycle": ["normal", "normal", "critical", "normal"],
            "freshness_cycle": ["fresh", "fresh", "stale", "fresh"],
            "view_quality_cycle": ["good", "medium", "good", "medium"],
            "operational_state_cycle": ["accepted", "activated", "accepted", "activated"],
            "tau_scale": 1.35,
            "tau_floor_s": 7.0,
            "epsilon_scale": 0.90,
            "epsilon_cap_by_risk": {"normal": 0.56, "critical": 0.70, "high": 0.70},
        },
        "semantic_cache_seed": {
            "enabled": True,
            "entries_per_area": 1,
            "cache_age": 0,
        },
    },
}


SEMANTIC_SCENARIO_PRESET_NAMES = (
    "nominal_patrol",
    "disaster_hotspot",
    "low_snr_soft",
    "low_snr_blockage",
    "edge_overload",
    "edge_overload_soft",
    "utm_conflict",
    "utm_conflict_soft",
)


SEMANTIC_NETWORK_LAYERS: dict[str, list[str]] = {
    "task_layer": ["VQA task", "risk level", "deadline", "semantic accuracy requirement"],
    "semantic_service_layer": ["cache answer", "semantic tokens", "raw image evidence"],
    "semantic_utility_layer": ["LUT answer accuracy", "payload-aware utility", "constraint-aware reward"],
    "network_layer": ["UAV mobility", "A2G SINR/rate", "bandwidth", "power", "edge CPU/GPU"],
    "cognitive_control_layer": ["RL controller", "heuristic baseline", "hybrid action projection"],
}


SERVICE_LEVELS: dict[int, dict[str, Any]] = {
    0: {
        "name": "cache_answer",
        "evidence_type": "cached semantic answer",
        "requires_uav": False,
        "requires_edge_model": False,
    },
    1: {
        "name": "semantic_tokens",
        "evidence_type": "detector tags/boxes/tokens",
        "requires_uav": True,
        "requires_edge_model": True,
    },
    2: {
        "name": "raw_image_evidence",
        "evidence_type": "full image evidence for VQA/VLM",
        "requires_uav": True,
        "requires_edge_model": True,
    },
    3: {
        "name": "roi_crop_image",
        "evidence_type": "reserved ROI/crop image evidence",
        "requires_uav": True,
        "requires_edge_model": True,
        "enabled_by_default": False,
    },
}

OPERATIONAL_INTENT_STATES = ("accepted", "activated", "nonconforming", "contingent")
MOBILITY_MODES = ("stay", "serve_task", "reposition", "avoid_conflict", "return_base")
SEMANTIC_PATHS = ("cache", "token", "image", "defer", "cache_update", "reject")
SEMANTIC_PATH_TO_SERVICE_LEVEL = {"cache": 0, "token": 1, "image": 2, "cache_update": 1}
SERVICE_LEVEL_TO_SEMANTIC_PATH = {0: "cache", 1: "token", 2: "image", 3: "image"}


FORMAL_SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "train_nominal": {
        "split": "train",
        "base_scenario": "nominal",
        "description": "Stable training scenario with calibrated nominal task arrivals.",
        "env": {"tasks_per_episode": 24, "episode_steps": 12},
        "task_layout": {"generation_mode": "staggered", "jitter_ratio": 0.16},
    },
    "train_mixed_random": {
        "split": "train",
        "base_scenario": "nominal",
        "scenario_mixture": ["nominal", "cache-heavy", "interference-heavy", "mobility-stress"],
        "description": "Training scenario that samples a known stressor per episode.",
        "env": {"tasks_per_episode": 24, "episode_steps": 12},
        "task_layout": {"generation_mode": "wave", "jitter_ratio": 0.14},
    },
    "test_conflict_heavy": {
        "split": "test",
        "base_scenario": "conflict-heavy",
        "description": "Held-out airspace conflict stress test.",
    },
    "test_interference_heavy": {
        "split": "test",
        "base_scenario": "interference-heavy",
        "description": "Held-out multi-UAV same-band interference stress test.",
    },
    "test_cache_heavy": {
        "split": "test",
        "base_scenario": "cache-heavy",
        "description": "Held-out semantic cache reuse/freshness stress test.",
    },
    "test_mobility_stress": {
        "split": "test",
        "base_scenario": "mobility-stress",
        "description": "Held-out long-range mobility and battery stress test.",
    },
    "test_unseen_mixed": {
        "split": "test",
        "base_scenario": "nominal",
        "scenario_mixture": ["conflict-heavy", "interference-heavy", "cache-heavy", "mobility-stress"],
        "description": "Unseen mixture with larger network size and heavier arrivals than training.",
        "env": {
            "num_uavs": 6,
            "tasks_per_episode": 32,
            "episode_steps": 16,
            "num_areas": 6,
            "edge_load_range": [0.35, 0.65],
            "gpu_load_range": [0.30, 0.60],
        },
        "task_layout": {
            "generation_mode": "wave",
            "jitter_ratio": 0.12,
            "risk_cycle": ["normal", "critical", "normal", "critical"],
            "freshness_cycle": ["fresh", "stale", "expired", "fresh"],
        },
    },
    "test_utm_conflict": {
        "split": "test",
        "base_scenario": "utm_conflict",
        "utm_realistic": True,
        "description": "Paper preset: UTM strategic conflict with DSS and notification delay.",
    },
    "test_utm_nominal_planning": {
        "split": "test",
        "base_scenario": "nominal",
        "utm_realistic": True,
        "description": "UTM-style nominal planning with accepted/activated operational intents and available DSS.",
        "env": {
            "tasks_per_episode": 18,
            "episode_steps": 10,
            "utm": {
                "enabled": True,
                "mode": "nominal_planning",
                "dss_available": True,
                "dss_delay_s": 0.05,
                "subscription_notification_delay_s": 0.08,
                "spatial_buffer_m": 25.0,
                "temporal_buffer_steps": 1,
            },
        },
        "task_layout": {"generation_mode": "staggered", "jitter_ratio": 0.12},
    },
    "test_utm_off_nominal_planning": {
        "split": "test",
        "base_scenario": "mobility-stress",
        "utm_realistic": True,
        "description": "UTM-style off-nominal planning with low battery/mobility pressure producing nonconforming intents.",
        "env": {
            "tasks_per_episode": 18,
            "episode_steps": 12,
            "initial_battery_j": 7_500.0,
            "return_energy_reserve_j": 1_500.0,
            "utm": {
                "enabled": True,
                "mode": "off_nominal_planning",
                "dss_available": True,
                "dss_delay_s": 0.08,
                "subscription_notification_delay_s": 0.12,
                "spatial_buffer_m": 30.0,
                "temporal_buffer_steps": 1,
            },
        },
        "task_layout": {
            "generation_mode": "wave",
            "jitter_ratio": 0.18,
            "risk_cycle": ["critical", "normal", "normal"],
            "freshness_cycle": ["stale", "expired", "fresh"],
        },
    },
    "test_utm_intent_conflict": {
        "split": "test",
        "base_scenario": "conflict-heavy",
        "utm_realistic": True,
        "description": "UTM-style strategic conflict detection with spatial/temporal buffers around overlapping intents.",
        "env": {
            "tasks_per_episode": 18,
            "episode_steps": 10,
            "utm": {
                "enabled": True,
                "mode": "flight_intent_validation",
                "dss_available": True,
                "dss_delay_s": 0.06,
                "subscription_notification_delay_s": 0.10,
                "spatial_buffer_m": 50.0,
                "altitude_buffer_m": 10.0,
                "temporal_buffer_steps": 2,
            },
        },
        "task_layout": {"generation_mode": "burst", "force_same_area": True, "jitter_ratio": 0.02},
    },
    "test_utm_dss_outage": {
        "split": "test",
        "base_scenario": "nominal",
        "utm_realistic": True,
        "description": "UTM-style DSS outage abstraction: operational intents enter contingent state and incur DSS delay.",
        "env": {
            "tasks_per_episode": 16,
            "episode_steps": 10,
            "utm": {
                "enabled": True,
                "mode": "dss_outage",
                "dss_available": False,
                "dss_delay_s": 0.75,
                "subscription_notification_delay_s": 0.20,
                "spatial_buffer_m": 25.0,
                "temporal_buffer_steps": 1,
            },
        },
        "task_layout": {"generation_mode": "wave", "jitter_ratio": 0.10},
    },
    "test_utm_notification_delay": {
        "split": "test",
        "base_scenario": "conflict-heavy",
        "utm_realistic": True,
        "description": "UTM-style subscription notification delay for delayed strategic conflict updates.",
        "env": {
            "tasks_per_episode": 18,
            "episode_steps": 10,
            "utm": {
                "enabled": True,
                "mode": "subscription_notifications",
                "dss_available": True,
                "dss_delay_s": 0.06,
                "subscription_notification_delay_s": 0.85,
                "spatial_buffer_m": 45.0,
                "altitude_buffer_m": 10.0,
                "temporal_buffer_steps": 2,
            },
        },
        "task_layout": {"generation_mode": "burst", "force_same_area": True, "jitter_ratio": 0.03},
    },
}


SCALABILITY_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "uav_count": {
        "M2": {"num_uavs": 2},
        "M4": {"num_uavs": 4},
        "M6": {"num_uavs": 6},
        "M8": {"num_uavs": 8},
    },
    "task_arrival": {
        "low": {"tasks_per_episode": 12, "episode_steps": 12},
        "medium": {"tasks_per_episode": 24, "episode_steps": 12},
        "high": {"tasks_per_episode": 40, "episode_steps": 16},
    },
    "edge_load": {
        "light": {"edge_load_range": [0.05, 0.20], "gpu_load_range": [0.03, 0.16]},
        "medium": {"edge_load_range": [0.20, 0.45], "gpu_load_range": [0.15, 0.38]},
        "heavy": {"edge_load_range": [0.45, 0.80], "gpu_load_range": [0.38, 0.72]},
    },
}


def available_scenarios() -> list[str]:
    return sorted(SCENARIO_PRESETS)


def semantic_scenario_preset_names() -> list[str]:
    return list(SEMANTIC_SCENARIO_PRESET_NAMES)


def available_formal_scenarios(include_utm: bool = False) -> list[str]:
    return [
        name
        for name, spec in FORMAL_SCENARIO_PRESETS.items()
        if include_utm or not bool(spec.get("utm_realistic", False))
    ]


def available_utm_realistic_scenarios() -> list[str]:
    return [name for name, spec in FORMAL_SCENARIO_PRESETS.items() if bool(spec.get("utm_realistic", False))]


def scalability_presets() -> dict[str, dict[str, dict[str, Any]]]:
    return SCALABILITY_PRESETS


def _normalize_scenario_name(name: str | None) -> str:
    if not name or name in {"default", "base"}:
        return "nominal"
    aliases = {
        "normal_patrol": "nominal_patrol",
    }
    name = aliases.get(name, name)
    if name not in SCENARIO_PRESETS:
        raise ValueError(f"unknown multi_uav_env scenario: {name}")
    return name


def _normalize_formal_scenario_name(name: str | None) -> str:
    if not name:
        return ""
    if name not in FORMAL_SCENARIO_PRESETS:
        raise ValueError(f"unknown formal semantic-network scenario: {name}")
    return name


def _apply_calibration(env_cfg: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    calibration = cfg.get("multi_uav_env", {}).get("calibration", {})
    if not calibration:
        return env_cfg
    out = dict(env_cfg)
    propulsion = calibration.get("propulsion", {})
    for key in ("flight_energy_j_per_m", "hover_power_w"):
        if key in propulsion:
            out[key] = propulsion[key]
    sensing = calibration.get("sensing", {})
    if "sensing_delay_s_by_level" in sensing:
        out["sensing_delay_s_by_level"] = sensing["sensing_delay_s_by_level"]
    if "processing_delay_s_by_level" in sensing:
        out["processing_delay_s_by_level"] = sensing["processing_delay_s_by_level"]
    if "cpu_workload_by_level" in sensing:
        out["cpu_workload_by_level"] = sensing["cpu_workload_by_level"]
    if "gpu_workload_by_level" in sensing:
        out["gpu_workload_by_level"] = sensing["gpu_workload_by_level"]
    if "a2g" in calibration:
        out["a2g"] = _deep_merge(out.get("a2g", {}), calibration["a2g"])
    edge = calibration.get("edge", {})
    for key in (
        "queue_delay_scale_s",
        "gpu_queue_delay_scale_s",
        "edge_load_decay",
        "model_load_delay_s",
        "model_cache_hit_delay_s",
        "model_cache_capacity",
        "gpu_memory_capacity_mb",
        "gpu_memory_load",
        "model_memory_mb_by_level",
    ):
        if key in edge:
            out[key] = edge[key]
    return out


def _env_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "num_uavs": 3,
        "num_edges": 1,
        "num_areas": 4,
        "episode_steps": 12,
        "slot_s": 1.0,
        "tasks_per_episode": 24,
        "area_spacing_m": 260.0,
        "area_radius_m": 70.0,
        "area_altitude_min_m": 30.0,
        "area_altitude_max_m": 120.0,
        "uav_altitude_m": 90.0,
        "uav_speed_mps": 18.0,
        "initial_battery_j": 20_000.0,
        "return_energy_reserve_j": 500.0,
        "enabled_service_levels": [0, 1, 2],
        "enable_service_level_3": False,
        "bandwidth_hz": 1_000_000.0,
        "queue_delay_scale_s": 0.45,
        "gpu_queue_delay_scale_s": 0.20,
        "edge_load_decay": 0.65,
        "model_load_delay_s": 0.35,
        "model_cache_hit_delay_s": 0.04,
        "model_cache_capacity": 3,
        "gpu_memory_capacity_mb": 8192.0,
        "gpu_memory_load": 0.25,
        "model_memory_mb_by_level": {"0": 128.0, "1": 768.0, "2": 3072.0, "3": 2048.0},
        "sensing_delay_s_by_level": {"0": 0.0, "1": 0.12, "2": 0.30, "3": 0.24},
        "processing_delay_s_by_level": {"0": 0.03, "1": 0.30, "2": 1.20, "3": 0.85},
        "cpu_workload_by_level": {"0": 0.02, "1": 0.25, "2": 0.65, "3": 0.50},
        "gpu_workload_by_level": {"0": 0.0, "1": 0.10, "2": 0.85, "3": 0.70},
        "flight_energy_j_per_m": 8.0,
        "hover_power_w": 70.0,
        "compute_energy_j_by_level": {"0": 0.05, "1": 1.2, "2": 5.0, "3": 4.0},
        "semantic_cache_capacity": 64,
        "semantic_cache_radius_m": 140.0,
        "semantic_cache_reuse_boost": 0.18,
        "cache_hit_probability": {"fresh": 0.96, "stale": 0.68, "expired": 0.28},
        "freshness_slots": {"fresh": 1, "stale": 3},
        "semantic_threshold_by_risk": {"normal": 0.50, "critical": 0.75, "high": 0.75},
        "a2g": {
            "carrier_mhz": 2400.0,
            "reference_distance_m": 1.0,
            "reference_gain_db": -40.0,
            "path_loss_exponent": 2.2,
            "noise_figure_db": 7.0,
            "excess_loss_db": 4.0,
            "los_excess_loss_db": 1.0,
            "nlos_excess_loss_db": 10.0,
            "los_a": 9.61,
            "los_b": 0.16,
            "fading_mode": "slow_fading",
            "slow_fading_std_db": 2.0,
            "fast_fading_std_db": 5.0,
            "fading_correlation": 0.85,
            "interference_enabled": True,
            "interference_floor_dbm": -120.0,
            "interference_overlap_scale": 0.02,
        },
        "utm": {
            "enabled": True,
            "mode": "nominal_planning",
            "spatial_buffer_m": 20.0,
            "altitude_buffer_m": 0.0,
            "temporal_buffer_steps": 0,
            "dss_available": True,
            "dss_delay_s": 0.05,
            "subscription_notification_delay_s": 0.10,
            "contingent_delay_s": 0.50,
        },
        "reward_success": 2.0,
        "reward_delay": 0.20,
        "reward_energy": 0.0004,
        "reward_payload": 0.002,
        "reward_violation": 1.0,
        "reward_conflict": 1.0,
    }
    out = _deep_merge(defaults, cfg.get("multi_uav_env", {}))
    sim_cfg = cfg.get("simulation", {})
    if "bandwidth_hz" in sim_cfg and "bandwidth_hz" not in cfg.get("multi_uav_env", {}):
        out["bandwidth_hz"] = sim_cfg["bandwidth_hz"]
    if "processing_delay_by_level" in sim_cfg and "processing_delay_s_by_level" not in cfg.get("multi_uav_env", {}):
        out["processing_delay_s_by_level"] = sim_cfg["processing_delay_by_level"]
    out = _apply_calibration(out, cfg)
    return _apply_bubbles_profile(out)


def _bubbles_profile_active(env_cfg: dict[str, Any]) -> bool:
    """True only when the config opts into the BUBBLES-conformant scenario profile.

    Gated by ``multi_uav_env.scenario_profile == "bubbles"``. Absent/None/other
    values leave every downstream code path bit-identical to the legacy default.
    """
    return str(env_cfg.get("scenario_profile") or "").strip().lower() == "bubbles"


def _apply_bubbles_profile(env_cfg: dict[str, Any]) -> dict[str, Any]:
    """Replace the default UAV performance envelope with the BUBBLES SAIL III-IV
    class (D2.1 Appendix B, Table B-2, p.99) when the profile is enabled.

    Cruise 14 m/s, RoC 5 m/s, RoD 4 m/s, horizontal size 2.0 m, vertical size
    1.0 m. Constraint limits are anchored to the mid-air-collision TLS
    ``TLS_MAC = 2.5e-7`` FAT/FH (D2.1 Appendix D, p.112): the airspace-conflict
    penalty in this profile represents a breach of the tactical-separation
    barrier that the TLS budget requires the U-space to keep below that rate.
    The mapping is declarative (no explicit FAT/FH -> reward conversion is made);
    see :data:`bubbles_separation.TLS_MAC_FAT_PER_FH`.
    """
    if not _bubbles_profile_active(env_cfg):
        return env_cfg
    out = dict(env_cfg)
    perf = bubbles_separation.TABLE_B2["SAIL_III_IV"]
    out["uav_speed_mps"] = float(perf.cruise_mps)
    out["uav_roc_mps"] = float(perf.roc_mps)
    out["uav_rod_mps"] = float(perf.rod_mps)
    out["uav_size_h_m"] = float(perf.size_h_m)
    out["uav_size_v_m"] = float(perf.size_v_m)
    # TLS anchoring (documentation only; does not alter reward magnitude).
    out.setdefault("tls_mac_fat_per_fh", bubbles_separation.TLS_MAC_FAT_PER_FH)
    out.setdefault("tls_overall_fat_per_fh", bubbles_separation.TLS_OVERALL_FAT_PER_FH)
    out.setdefault("bubbles_traffic_class", "SAIL_III_IV")
    return out


class MultiUAVVQAEnv:
    """Canonical multi-UAV VQA semantic communication environment."""

    def __init__(
        self,
        tasks: list[dict[str, str]],
        lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
        cfg: dict[str, Any],
        seed: int | None = None,
        semantic_utility_model: SemanticUtilityModel | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("MultiUAVVQAEnv needs at least one task")
        if not lut:
            raise ValueError("MultiUAVVQAEnv needs a non-empty LUT")
        self.raw_tasks = list(tasks)
        self.lut = lut
        # W3b: the per-service quality query key is (qtype, service, snr) only.
        # The 6-D table stays loaded (service-level discovery, SNR bins, and
        # the cache machinery still read it); the per-service lookup uses this
        # marginalised 3-D index so sparse view/freshness/risk cells no longer
        # fragment the quality estimate.
        self.lut3 = self._marginalize_lut_3d(lut)
        self.semantic_utility_model = semantic_utility_model
        self.cfg = cfg
        self.env_cfg = _env_cfg(cfg)
        self.default_scenario = _normalize_scenario_name(str(self.env_cfg.get("scenario", "nominal")))
        self.scenario_name = self.default_scenario
        self.scenario_cfg = SCENARIO_PRESETS[self.scenario_name]
        self.formal_scenario_name = ""
        self.formal_scenario_cfg: dict[str, Any] = {}
        self.benchmark_split = ""
        self.scalability_profile: dict[str, str] = {}
        self.snr_bins_db = self._snr_bins_from_lut() or snr_bins_from_config(cfg) or [0.0, 10.0, 20.0]
        self.base_seed = int(seed if seed is not None else cfg.get("simulation", {}).get("seed", 13))
        self.rng = random.Random(self.base_seed)
        self.episode = -1
        self.step_count = 0
        self.uavs: list[UAVNode] = []
        self.edges: list[EdgeNode] = []
        self.tasks: list[EnvTask] = []
        self.semantic_cache_entries: list[SemanticCacheEntry] = []
        self.last_info: dict[str, Any] = {}
        self.policy_name = "policy"

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        else:
            self.rng.seed(self.base_seed)
        options = options or {}
        self.policy_name = str(options.get("policy_name", self.policy_name))
        self.formal_scenario_name = _normalize_formal_scenario_name(
            str(options.get("formal_scenario", "")) if options.get("formal_scenario") else ""
        )
        self.formal_scenario_cfg = FORMAL_SCENARIO_PRESETS.get(self.formal_scenario_name, {})
        self.benchmark_split = str(self.formal_scenario_cfg.get("split", ""))
        self.scenario_name = self._select_scenario_name(options)
        base_scenario_cfg = SCENARIO_PRESETS[self.scenario_name]
        self.scenario_cfg = _deep_merge(base_scenario_cfg, self._formal_scenario_overlay())
        env_cfg = _deep_merge(_env_cfg(self.cfg), self.scenario_cfg.get("env", {}))
        env_cfg = _deep_merge(env_cfg, self._scalability_env_overrides(options))
        self.env_cfg = env_cfg
        self.episode += 1
        self.step_count = 0
        self.uavs = self._init_uavs()
        self.edges = self._init_edges()
        self.tasks = self._init_tasks(options)
        self.semantic_cache_entries = self._init_scenario_semantic_cache()
        self.last_info = {}
        return self._observation()

    def _select_scenario_name(self, options: dict[str, Any]) -> str:
        if options.get("scenario"):
            return _normalize_scenario_name(str(options["scenario"]))
        if self.formal_scenario_cfg.get("scenario_mixture"):
            mixture = [str(item) for item in self.formal_scenario_cfg["scenario_mixture"]]
            return _normalize_scenario_name(self.rng.choice(mixture))
        if self.formal_scenario_cfg.get("base_scenario"):
            return _normalize_scenario_name(str(self.formal_scenario_cfg["base_scenario"]))
        return self.default_scenario

    def _formal_scenario_overlay(self) -> dict[str, Any]:
        if not self.formal_scenario_cfg:
            return {}
        overlay: dict[str, Any] = {}
        for key in ("description", "env", "task_layout", "semantic_cache_seed"):
            if key in self.formal_scenario_cfg:
                overlay[key] = self.formal_scenario_cfg[key]
        return overlay

    def _scalability_env_overrides(self, options: dict[str, Any]) -> dict[str, Any]:
        profile: dict[str, str] = {}
        out: dict[str, Any] = {}
        if options.get("num_uavs") is not None:
            value = str(options["num_uavs"])
            key = value if value.startswith("M") else f"M{int(value)}"
            out = _deep_merge(out, SCALABILITY_PRESETS["uav_count"].get(key, {"num_uavs": int(value.replace("M", ""))}))
            profile["uav_count"] = key
        if options.get("task_arrival") is not None:
            key = str(options["task_arrival"])
            if key not in SCALABILITY_PRESETS["task_arrival"]:
                raise ValueError(f"unknown task_arrival scalability preset: {key}")
            out = _deep_merge(out, SCALABILITY_PRESETS["task_arrival"][key])
            profile["task_arrival"] = key
        if options.get("edge_load") is not None:
            key = str(options["edge_load"])
            if key not in SCALABILITY_PRESETS["edge_load"]:
                raise ValueError(f"unknown edge_load scalability preset: {key}")
            out = _deep_merge(out, SCALABILITY_PRESETS["edge_load"][key])
            profile["edge_load"] = key
        self.scalability_profile = profile
        return out

    def step(self, action: dict[str, Any] | None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action = action or {}
        task = self._select_task(action)
        if task is None:
            self.step_count += 1
            done = self.step_count >= int(self.env_cfg["episode_steps"])
            info = self._empty_info()
            return self._observation(), 0.0, done, info

        parsed = self.parse_action(action, task)
        info = self.evaluate_action(parsed, task_id=task.task_id, mutate=False)
        if str(parsed["semantic_path"]) == "reject":
            task.rejected = True
            task.task_status = "rejected"
            self._advance_cache_ages()
            self.step_count += 1
            info["task_status"] = "rejected"
            info["rejected"] = True
            info["remaining_deadline_s"] = round(float(self._remaining_deadline_s(task)), 6)
            reward = self._reject_reward(task, info)
            info["reward"] = round(reward, 6)
            self.last_info = info
            done = self.step_count >= int(self.env_cfg["episode_steps"]) or all(
                t.completed or t.expired or t.rejected for t in self.tasks
            )
            return self._observation(), round(reward, 6), done, info
        if str(parsed["semantic_path"]) == "defer":
            task.defer_count += 1
            task.task_status = "deferred"
            self._advance_cache_ages()
            self.step_count += 1
            self._expire_overdue_tasks()
            info["defer_count"] = int(task.defer_count)
            info["remaining_deadline_s"] = round(float(self._remaining_deadline_s(task)), 6)
            info["expired"] = bool(task.expired)
            info["task_status"] = task.task_status
            reward = self._reward(task, info)
            info["reward"] = round(reward, 6)
            self.last_info = info
            done = self.step_count >= int(self.env_cfg["episode_steps"]) or all(
                t.completed or t.expired or t.rejected for t in self.tasks
            )
            return self._observation(), round(reward, 6), done, info
        success = bool(info["success"])
        uav = self.uavs[int(parsed["uav_assignment"]) % len(self.uavs)]
        edge = self.edges[int(parsed["edge_id"]) % len(self.edges)]

        # Cache-hit paths do not transmit, so the cache-eligible info builder
        # omits sensed SNR/SINR (only reachable once epsilon recalibration makes
        # the cache eligible; task #28). Fall back to the task's last-known link
        # state; transmit paths always carry these keys -> behaviour unchanged.
        task.last_sensed_snr_db = float(info.get("sensed_snr_db", task.last_sensed_snr_db))
        task.last_sinr_db = float(info.get("sinr_db", task.last_sinr_db))
        task.last_snr_bin = str(info.get("snr_bin", task.last_snr_bin))
        task.operational_intent_state = str(info.get("operational_intent_state", task.operational_intent_state))
        if success:
            task.completed = True
            task.task_status = "served"
            task.cache_age = 0
            task.freshness_bin = "fresh"

        self._move_uav(uav, task, parsed)
        uav.battery_j = max(0.0, uav.battery_j - float(info["energy_j"]))
        uav.current_task_id = task.task_id
        uav.camera_state = str(parsed["mobility_mode"]) if int(parsed["service_level"]) == 0 else f"sensing:{parsed['mobility_mode']}"
        uav.utilization = min(1.0, uav.utilization + 1.0 / max(1, len(self.tasks)))

        self._update_edge(edge, parsed, info)
        self._update_semantic_cache(task, parsed, info)
        self._advance_cache_ages()

        reward = self._reward(task, info)
        info["reward"] = round(reward, 6)
        self.last_info = info
        self.step_count += 1
        self._expire_overdue_tasks()
        done = self.step_count >= int(self.env_cfg["episode_steps"]) or all(
            t.completed or t.expired or t.rejected for t in self.tasks
        )
        return self._observation(), round(reward, 6), done, info

    def parse_action(self, action: dict[str, Any], task: EnvTask | None = None) -> dict[str, Any]:
        task = task or self._front_task()
        semantic_path = self._semantic_path(action)
        service_level = self._service_level(action)
        if semantic_path in SEMANTIC_PATH_TO_SERVICE_LEVEL:
            service_level = self._nearest_service_level(SEMANTIC_PATH_TO_SERVICE_LEVEL[semantic_path])
        elif semantic_path == "reject":
            service_level = -1
        sensing_default = "reuse_cache" if service_level == 0 else "observe"
        if semantic_path == "defer":
            sensing_default = "defer"
        elif semantic_path == "reject":
            sensing_default = "reject"
        elif semantic_path == "cache_update":
            sensing_default = "observe"
        assignment = action.get("uav_assignment", action.get("assigned_uav", None))
        if isinstance(assignment, list) and assignment:
            assignment = assignment[0]
        if assignment is None and task is not None:
            assignment = self._nearest_uav(task).uav_id
        waypoint_delta_raw = action.get("waypoint_delta", action.get("waypoint_delta_xy", None))
        if waypoint_delta_raw is None and action.get("dx") is not None and action.get("dy") is not None:
            waypoint_delta_raw = [action.get("dx"), action.get("dy")]
        waypoint_delta = self._waypoint_delta(waypoint_delta_raw)
        mobility_mode = self._mobility_mode(action, service_level)
        return {
            "task_id": action.get("task_id", task.task_id if task else ""),
            "semantic_path": semantic_path,
            "service_level": service_level,
            "sensing_decision": str(action.get("sensing_decision", sensing_default)),
            "bandwidth": self._bandwidth_hz(action),
            "power": self._power_w(action),
            "cpu_share": self._share(action.get("cpu_share", 0.5)),
            "gpu_share": self._share(action.get("gpu_share", 0.5)),
            "uav_assignment": int(assignment or 0) % max(1, len(self.uavs)),
            "edge_id": int(action.get("edge_id", 0)) % max(1, len(self.edges)),
            "mobility_mode": mobility_mode,
            "waypoint": action.get("waypoint"),
            "waypoint_delta": waypoint_delta,
            "altitude_delta": float(action.get("altitude_delta", action.get("dz", 0.0))),
            "concurrent_actions": list(action.get("concurrent_actions", [])),
        }

    def default_action(self, service_level: int = 0, obs: dict[str, Any] | None = None) -> dict[str, Any]:
        presets = {
            0: {"bandwidth": 0.05 * float(self.env_cfg["bandwidth_hz"]), "power": 0.1, "cpu_share": 0.05, "gpu_share": 0.01},
            1: {"bandwidth": 0.45 * float(self.env_cfg["bandwidth_hz"]), "power": 0.5, "cpu_share": 0.25, "gpu_share": 0.05},
            2: {"bandwidth": 1.00 * float(self.env_cfg["bandwidth_hz"]), "power": 1.0, "cpu_share": 0.55, "gpu_share": 0.35},
            3: {"bandwidth": 0.70 * float(self.env_cfg["bandwidth_hz"]), "power": 0.8, "cpu_share": 0.40, "gpu_share": 0.25},
        }
        action = dict(presets.get(int(service_level), presets[max(self.service_levels())]))
        action["service_level"] = int(service_level)
        action["semantic_path"] = SERVICE_LEVEL_TO_SEMANTIC_PATH.get(int(service_level), "image")
        action["mobility_mode"] = "stay" if int(service_level) == 0 else "serve_task"
        action["waypoint"] = None
        action["waypoint_delta"] = [0.0, 0.0]
        action["altitude_delta"] = 0.0
        if obs and obs.get("task_id"):
            action["task_id"] = obs["task_id"]
        return action

    def candidate_action(self, service_level: int, obs: dict[str, Any] | None = None) -> dict[str, Any]:
        action = self.default_action(service_level, obs)
        if obs is not None and obs.get("risk_level") == "critical" and int(service_level) > 0:
            action["bandwidth"] = float(self.env_cfg["bandwidth_hz"])
            action["power"] = max(float(action["power"]), 1.0)
            action["cpu_share"] = min(1.0, float(action["cpu_share"]) + 0.15)
            action["gpu_share"] = min(1.0, float(action["gpu_share"]) + 0.10)
        return action

    def candidate_metrics(self, service_level: int, obs: dict[str, Any] | None = None) -> dict[str, float]:
        action = self.candidate_action(service_level, obs)
        info = self.evaluate_action(action, task_id=str(obs.get("task_id")) if obs else None, obs=obs, mutate=False)
        return {
            "accuracy": float(info["answer_accuracy_est"]),
            "delay_s": float(info["delay_s"]),
            "energy_j": float(info["energy_j"]),
            "payload_kb": float(info["payload_kb"]),
            "success": float(bool(info["success"])),
        }

    def candidate_path_metrics(self, task: EnvTask | None = None) -> dict[str, dict[str, Any]]:
        task = task or self._front_task()
        if task is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for path in SEMANTIC_PATHS:
            out[path] = self._candidate_path_metric(task, path)
        return out

    def candidate_mobility_metrics(self, task: EnvTask | None = None) -> dict[str, dict[str, dict[str, Any]]]:
        task = task or self._front_task()
        if task is None:
            return {}
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for path in SEMANTIC_PATHS:
            out[path] = {}
            if path in {"defer", "reject"}:
                continue
            for mode in ("stay", "serve_task", "avoid_conflict", "reposition"):
                action = self._path_action(path, task)
                action["mobility_mode"] = mode
                if mode == "reposition":
                    nearest = self._nearest_uav(task)
                    dx = task.x_m - nearest.x_m
                    dy = task.y_m - nearest.y_m
                    norm = max(1.0, math.hypot(dx, dy))
                    step = min(norm, nearest.speed_mps * float(self.env_cfg["slot_s"]))
                    action["waypoint_delta"] = [dx / norm * step, dy / norm * step]
                info = self.evaluate_action(action, task_id=task.task_id, obs={"task_id": task.task_id}, mutate=False)
                accuracy_lcb = float(info.get("semantic_accuracy_lcb", 0.0))
                semantic_feasible = bool(info.get("semantic_success", False)) and not bool(info.get("quality_violation", False))
                if path == "cache":
                    semantic_feasible = bool(info.get("cache_eligible", False)) and semantic_feasible
                deadline_feasible = not bool(info.get("deadline_violation", False))
                energy_feasible = not bool(info.get("battery_violation", False))
                utm_feasible = not bool(info.get("utm_conflict_violation", info.get("utm_constraint_violation", False)))
                resource_feasible = not bool(info.get("resource_violation", False))
                out[path][mode] = {
                    "utm_feasible": bool(utm_feasible),
                    "arrival_delay_s": float(info.get("arrival_delay_s", 0.0)),
                    "tx_delay_s": float(info.get("tx_delay_s", 0.0)),
                    "total_delay_s": float(info.get("delay_s", 0.0)),
                    "deadline_slack_s": float(info.get("remaining_deadline_s", self._remaining_deadline_s(task)))
                    - float(info.get("delay_s", 0.0)),
                    "semantic_feasible": bool(semantic_feasible),
                    "deadline_feasible": bool(deadline_feasible),
                    "energy_feasible": bool(energy_feasible),
                    "resource_feasible": bool(resource_feasible),
                    "joint_feasible": bool(
                        not task.expired
                        and semantic_feasible
                        and deadline_feasible
                        and energy_feasible
                        and utm_feasible
                        and resource_feasible
                    ),
                    "semantic_accuracy_lcb": accuracy_lcb,
                    "utm_conflict_risk": float(info.get("utm_conflict_risk", 0.0)),
                }
        return out

    def _candidate_path_metric(self, task: EnvTask, semantic_path: str) -> dict[str, Any]:
        action = self._path_action(semantic_path, task)
        info = self.evaluate_action(action, task_id=task.task_id, obs={"task_id": task.task_id}, mutate=False)
        if semantic_path == "reject":
            reject_feasible = bool(info.get("reject_feasible", False))
            return {
                "feasible": reject_feasible,
                "deadline_feasible": True,
                "semantic_feasible": False,
                "energy_feasible": True,
                "utm_feasible": True,
                "joint_feasible": reject_feasible,
                "reject_feasible": reject_feasible,
                "resource_feasible": True,
                "accuracy_lcb": 0.0,
                "accuracy_mean": 0.0,
                "quality_gap": max(0.0, float(task.epsilon_k)),
                "payload_kb": 0.0,
                "delay_s": 0.0,
                "energy_j": 0.0,
                "deadline_slack_s": self._remaining_deadline_s(task),
                "tx_delay_s": 0.0,
                "queue_delay_s": 0.0,
                "infer_delay_s": 0.0,
                "load_delay_s": 0.0,
                "arrival_delay_s": 0.0,
                "bottleneck_type": str(info.get("reject_reason", "")),
                "required_deadline_reduction_s": 0.0,
                "required_rate_mbps": 0.0,
                "required_bandwidth_hz": 0.0,
                "edge_queue_pressure": 0.0,
                "model_cache_hit": False,
                "cache_eligible": False,
                "utm_constraint_violation": False,
                "utm_conflict_violation": False,
                "service_level": -1,
                "semantic_path": "reject",
                "reject_reason": str(info.get("reject_reason", "")),
                "expected_saved_energy_j": float(info.get("expected_saved_energy_j", 0.0)),
                "expected_saved_delay_s": float(info.get("expected_saved_delay_s", 0.0)),
                "avoided_utm_violation": bool(info.get("avoided_utm_violation", False)),
                "avoided_deadline_violation": bool(info.get("avoided_deadline_violation", False)),
                "task_success": False,
                "semantic_success": False,
            }
        accuracy_lcb = float(info.get("semantic_accuracy_lcb", 0.0))
        delay_s = float(info.get("delay_s", 0.0))
        remaining = self._remaining_deadline_s(task)
        expired = bool(task.expired or remaining <= 0.0)
        semantic_feasible = bool(info.get("semantic_success", False)) and not bool(info.get("quality_violation", False))
        if semantic_path == "cache":
            semantic_feasible = bool(info.get("cache_eligible", False)) and semantic_feasible
        deadline_feasible = (delay_s <= remaining) and not bool(info.get("deadline_violation", False))
        energy_feasible = not bool(info.get("battery_violation", False))
        utm_feasible = not bool(info.get("utm_conflict_violation", info.get("utm_constraint_violation", False)))
        resource_feasible = not bool(info.get("resource_violation", False))
        if semantic_path == "defer":
            semantic_feasible = True
            deadline_feasible, energy_feasible, utm_feasible = self._defer_feasibility(task)
            resource_feasible = True
        joint_feasible = bool(
            not expired
            and semantic_feasible
            and deadline_feasible
            and energy_feasible
            and utm_feasible
            and resource_feasible
        )
        tx_delay_s = float(info.get("tx_delay_s", 0.0))
        queue_delay_s = float(info.get("queue_delay_s", 0.0))
        infer_delay_s = float(info.get("infer_delay_s", 0.0))
        load_delay_s = float(info.get("load_delay_s", 0.0))
        arrival_delay_s = float(info.get("arrival_delay_s", info.get("fly_delay_s", 0.0)))
        required = self._required_link_for_deadline(info, task)
        edge_id = int(info.get("edge_id", action.get("edge_id", 0))) % max(1, len(self.edges))
        edge = self.edges[edge_id]
        model_cache_hit = load_delay_s <= float(self.env_cfg.get("model_cache_hit_delay_s", 0.0)) + 1e-9
        return {
            "feasible": joint_feasible,
            "deadline_feasible": bool(deadline_feasible),
            "semantic_feasible": bool(semantic_feasible),
            "energy_feasible": bool(energy_feasible),
            "utm_feasible": bool(utm_feasible),
            "joint_feasible": bool(joint_feasible),
            "resource_feasible": bool(resource_feasible),
            "accuracy_lcb": accuracy_lcb,
            "accuracy_mean": float(info.get("semantic_accuracy_mean", 0.0)),
            "quality_gap": max(0.0, float(task.epsilon_k) - accuracy_lcb),
            "payload_kb": float(info.get("payload_kb", 0.0)),
            "delay_s": delay_s,
            "energy_j": float(info.get("energy_j", 0.0)),
            "deadline_slack_s": remaining - delay_s,
            "tx_delay_s": tx_delay_s,
            "queue_delay_s": queue_delay_s,
            "infer_delay_s": infer_delay_s,
            "load_delay_s": load_delay_s,
            "arrival_delay_s": arrival_delay_s,
            "bottleneck_type": self._candidate_bottleneck_type(
                expired=expired,
                semantic_feasible=semantic_feasible,
                deadline_feasible=deadline_feasible,
                energy_feasible=energy_feasible,
                utm_feasible=utm_feasible,
                resource_feasible=resource_feasible,
                tx_delay_s=tx_delay_s,
                queue_delay_s=queue_delay_s,
                infer_delay_s=infer_delay_s,
                load_delay_s=load_delay_s,
                arrival_delay_s=arrival_delay_s,
            ),
            "required_deadline_reduction_s": max(0.0, delay_s - remaining),
            "required_rate_mbps": required["required_rate_mbps"],
            "required_bandwidth_hz": required["required_bandwidth_hz"],
            "edge_queue_pressure": min(1.0, 0.5 * float(edge.load) + 0.5 * float(edge.gpu_load)),
            "model_cache_hit": bool(model_cache_hit),
            "cache_eligible": bool(info.get("cache_eligible", False)),
            "utm_constraint_violation": bool(info.get("utm_constraint_violation", False)),
            "utm_conflict_violation": bool(info.get("utm_conflict_violation", False)),
            "service_level": int(info.get("service_level", action.get("service_level", 0))),
            "semantic_path": semantic_path,
            "reject_reason": "",
            "expected_saved_energy_j": 0.0,
            "expected_saved_delay_s": 0.0,
            "avoided_utm_violation": False,
            "avoided_deadline_violation": False,
            "task_success": bool(info.get("success", False)),
        }

    @staticmethod
    def _candidate_bottleneck_type(
        *,
        expired: bool,
        semantic_feasible: bool,
        deadline_feasible: bool,
        energy_feasible: bool,
        utm_feasible: bool,
        resource_feasible: bool,
        tx_delay_s: float,
        queue_delay_s: float,
        infer_delay_s: float,
        load_delay_s: float,
        arrival_delay_s: float,
    ) -> str:
        if expired:
            return "expired"
        if not semantic_feasible:
            return "semantic_quality"
        if not utm_feasible:
            return "utm"
        if not energy_feasible:
            return "energy"
        if not resource_feasible:
            return "resource"
        if not deadline_feasible:
            parts = {
                "tx_delay": tx_delay_s,
                "queue_delay": queue_delay_s,
                "mobility": arrival_delay_s,
                "resource": infer_delay_s + load_delay_s,
            }
            return max(parts, key=parts.get)
        return "none"

    def _required_link_for_deadline(self, info: dict[str, Any], task: EnvTask) -> dict[str, float]:
        remaining = float(info.get("remaining_deadline_s", self._remaining_deadline_s(task)))
        payload_kb = float(info.get("payload_kb", 0.0))
        tx_delay_s = float(info.get("tx_delay_s", 0.0))
        delay_s = float(info.get("delay_s", 0.0))
        current_rate_mbps = max(1e-9, float(info.get("rate_mbps", 0.0)))
        current_bandwidth_hz = max(1.0, float(info.get("bandwidth_hz", self.env_cfg["bandwidth_hz"])))
        non_tx_delay_s = max(0.0, delay_s - tx_delay_s)
        tx_budget_s = max(1e-9, remaining - non_tx_delay_s)
        required_rate_mbps = (payload_kb * 8.0 / 1000.0) / tx_budget_s if payload_kb > 0.0 else 0.0
        spectral_efficiency_mbps_per_hz = current_rate_mbps / current_bandwidth_hz
        if required_rate_mbps <= current_rate_mbps:
            required_bandwidth_hz = current_bandwidth_hz
        elif spectral_efficiency_mbps_per_hz > 0.0:
            required_bandwidth_hz = required_rate_mbps / spectral_efficiency_mbps_per_hz
        else:
            required_bandwidth_hz = float("inf")
        return {
            "required_rate_mbps": required_rate_mbps,
            "required_bandwidth_hz": required_bandwidth_hz,
        }

    def _path_action(self, semantic_path: str, task: EnvTask) -> dict[str, Any]:
        if semantic_path == "reject":
            return {"task_id": task.task_id, "semantic_path": "reject", "service_level": -1, "mobility_mode": "stay"}
        if semantic_path == "defer":
            return {"task_id": task.task_id, "semantic_path": "defer", "service_level": 0, "mobility_mode": "stay"}
        service_level = SEMANTIC_PATH_TO_SERVICE_LEVEL.get(semantic_path, 0)
        action = self.candidate_action(service_level, {"task_id": task.task_id, "risk_level": task.risk_level})
        action["semantic_path"] = semantic_path
        if semantic_path == "cache_update":
            action["sensing_decision"] = "observe"
        return action

    def _defer_feasibility(self, task: EnvTask) -> tuple[bool, bool, bool]:
        remaining_after_defer = self._remaining_deadline_s(task) - float(self.env_cfg["slot_s"])
        if task.expired or remaining_after_defer <= 0.0:
            return False, True, True
        service_infos = [
            self.evaluate_action(self._path_action(path, task), task_id=task.task_id, obs={"task_id": task.task_id})
            for path in ("cache", "token", "image", "cache_update")
        ]
        if any(
            not bool(info.get("quality_violation", False))
            and not bool(info.get("deadline_violation", False))
            and not bool(info.get("battery_violation", False))
            and not bool(info.get("resource_violation", False))
            and not bool(info.get("utm_conflict_violation", False))
            for info in service_infos
        ):
            return False, True, True
        edge_decay = float(self.env_cfg.get("edge_load_decay", 1.0))
        edge_improvement = max(0.0, 1.0 - edge_decay)
        for info in service_infos:
            if bool(info.get("quality_violation", False)):
                continue
            if bool(info.get("battery_violation", False)) or bool(info.get("resource_violation", False)):
                continue
            if bool(info.get("utm_conflict_violation", False)):
                continue
            delay_s = float(info.get("delay_s", 0.0))
            fly_delay_s = float(info.get("fly_delay_s", 0.0))
            queue_delay_s = float(info.get("queue_delay_s", 0.0))
            # Task #33-A: under comm_window the deadline clock ignores flight, so
            # the whole fly term is off the deadline-charged delay; under legacy
            # keep the historical one-slot flight-progress credit.
            fly_credit = fly_delay_s if self._deadline_semantics() == "comm_window" \
                else min(fly_delay_s, float(self.env_cfg["slot_s"]))
            expected_delay_after_defer = max(
                0.0,
                delay_s
                - fly_credit
                - edge_improvement * queue_delay_s,
            )
            if expected_delay_after_defer <= remaining_after_defer:
                return True, True, True
        return False, True, True

    def _remaining_deadline_s(self, task: EnvTask) -> float:
        elapsed = max(0, self.step_count - int(task.generation_time)) * float(self.env_cfg["slot_s"])
        return max(0.0, float(task.tau_k) - elapsed)

    def _deadline_semantics(self) -> str:
        """Task #33-A: which delay components count against the deadline clock.

        "legacy"      (default) -- the FULL end-to-end delay, including the UAV
                      flight/positioning time (fly_delay), is charged against
                      tau_k.  Preserves v1-v5 behaviour bit-for-bit.
        "comm_window" -- tau_k is the tactical COMMUNICATION-DECISION window
                      only; the deadline clock counts sense + tx + queue + infer +
                      load and EXCLUDES fly_delay (flight/positioning is a tasking-
                      layer concern, per BUBBLES D2.1's separation-communication
                      T4 window).  Closes the scale contradiction where a critical
                      tau_k of 2.55 s can never fit a ~13 s flight to the task.
        """
        return str(self.env_cfg.get("deadline_semantics", "legacy") or "legacy").lower()

    def _deadline_delay_s(self, delay: dict[str, float]) -> float:
        """Task #33-A: the delay charged against the deadline clock.

        Under "comm_window" the flight/positioning term (fly_delay_s /
        arrival_delay_s) is removed; every other latency component (sense, tx,
        queue, infer, load, and the UTM DSS/notification delays already folded
        into total_delay_s) still counts.  Under "legacy" the full total is used.
        This is the SINGLE chokepoint every deadline comparison routes through, so
        the two semantics stay consistent across step/certificate/reject/defer.
        """
        total = float(delay.get("total_delay_s", 0.0))
        if self._deadline_semantics() != "comm_window":
            return total
        fly = float(delay.get("fly_delay_s", delay.get("arrival_delay_s", 0.0)))
        return max(0.0, total - fly)

    @staticmethod
    def _count_bucket_for_task(task: EnvTask) -> str:
        """v5 count-bucket key: only counting tasks carry a numeric bucket."""
        if str(task.task_type) != "counting":
            return "na"
        return count_bucket_v5(int(task.object_count))

    def _escalation_enabled(self) -> bool:
        """Change 5 (task #28 v5): the spec-attainability escalation layer.

        Default "off" preserves legacy/v1-v4 reject/expired bookkeeping
        bit-for-bit.  Under "spec_attainable" a critical/high reject-or-expired
        task is charged as a quality violation ONLY if it was spec-attainable
        (some tx service could clear eps AND fit the full tau_k); an
        un-attainable one is ESCALATED (no quality cost, routed to the escalation
        dual channel).
        """
        return str(self.env_cfg.get("escalation_mode", "off") or "off").lower() == "spec_attainable"

    def _compute_spec_attainable(self, task: EnvTask, snr_bin: str) -> bool:
        """Ground-truth feasibility certificate (change 5).

        For the transmission services {1 token, 2 image} at the task's realised
        SNR, is there one whose quality LCB clears the task's epsilon_k AND whose
        deadline-charged delay fits the FULL tau_k?  This is an ENVIRONMENT
        property (env LUT quality + physics), exposed to the policy as a state
        feature -- independent of the RL quality backend.

        Task #34-(ii): the certificate is re-computed EVERY slot from the
        remaining tau (remaining-tau reslot), so a task whose window has been
        eaten by defers/aging turns un-attainable, and the certificate is a
        non-constant distribution under nominal (verified by test).  The deadline
        comparison routes through the SAME _deadline_delay_s chokepoint as the
        step judgment, so under comm_window flight is excluded here too.
        """
        remaining = self._remaining_deadline_s(task)
        if bool(task.expired) or remaining <= 0.0:
            return False
        for level in (1, 2):
            if level not in self.service_levels():
                continue
            action = self.candidate_action(level, {"task_id": task.task_id, "snr_bin": snr_bin,
                                                    "risk_level": task.risk_level})
            info = self.evaluate_action(action, task_id=task.task_id,
                                        obs={"task_id": task.task_id, "snr_bin": snr_bin}, mutate=False)
            quality_ok = not bool(info.get("quality_violation", False))
            ddl_delay = self._deadline_delay_s({
                "total_delay_s": float(info.get("delay_s", info.get("total_delay_s", 1e9))),
                "fly_delay_s": float(info.get("fly_delay_s", info.get("arrival_delay_s", 0.0))),
            })
            deadline_ok = ddl_delay <= remaining
            if quality_ok and deadline_ok:
                return True
        return False

    def _cache_freshness_from_age(self, cache_age: int) -> str:
        return self._freshness_from_age(int(cache_age))

    def _cache_status(self, task: EnvTask) -> dict[str, Any]:
        exact_entries: list[SemanticCacheEntry] = []
        nearby_entries: list[SemanticCacheEntry] = []
        radius = float(self.env_cfg["semantic_cache_radius_m"])
        for entry in self.semantic_cache_entries:
            same_type = (entry.question_type or entry.task_type) == task.task_type
            if not same_type:
                continue
            exact = entry.task_id == task.task_id or int(entry.area_id) == int(task.area_id)
            nearby = math.hypot(entry.x_m - task.x_m, entry.y_m - task.y_m) <= radius
            if exact:
                exact_entries.append(entry)
            elif nearby:
                nearby_entries.append(entry)
        candidates = [*exact_entries, *nearby_entries]
        if candidates:
            best = max(candidates, key=lambda item: (float(item.quality_lcb), -int(item.cache_age), float(item.priority)))
            quality_lcb = float(best.quality_lcb)
            cache_age = int(best.cache_age)
            freshness = self._cache_freshness_from_age(cache_age)
            uncertainty = float(best.uncertainty)
        else:
            best = None
            quality_lcb = 0.0
            cache_age = int(task.cache_age)
            freshness = self._freshness_from_age(cache_age)
            uncertainty = 1.0
        exact_match = bool(exact_entries)
        nearby_match = bool(nearby_entries)
        eligible = bool(candidates) and freshness != "expired" and quality_lcb >= float(task.epsilon_k)
        return {
            "cache_exact_match": exact_match,
            "cache_nearby_match": nearby_match,
            "cache_eligible": eligible,
            "cache_quality_lcb": quality_lcb,
            "cache_uncertainty": uncertainty,
            "cache_age": cache_age,
            "cache_freshness_bin": freshness,
            "cache_entry_task_id": best.task_id if best else "",
        }

    def evaluate_action(
        self,
        action: dict[str, Any],
        task_id: str | None = None,
        obs: dict[str, Any] | None = None,
        mutate: bool = False,
    ) -> dict[str, Any]:
        task = self._task_by_id(task_id) if task_id else self._front_task()
        if task is None:
            return self._empty_info()
        parsed = self.parse_action(action, task)
        if str(parsed["semantic_path"]) == "reject":
            return self._reject_info(task, parsed)
        if str(parsed["semantic_path"]) == "defer":
            return self._defer_info(task, parsed)
        if task.expired or self._remaining_deadline_s(task) <= 0.0:
            return self._expired_task_info(task, parsed)
        uav = self.uavs[int(parsed["uav_assignment"]) % len(self.uavs)]
        edge = self.edges[int(parsed["edge_id"]) % len(self.edges)]
        link = self._link_budget(task, uav, parsed, interference_dbm=None)
        interference_dbm = self._interference_dbm(task, parsed, link)
        if parsed["concurrent_actions"]:
            link = self._link_budget(task, uav, parsed, interference_dbm=interference_dbm)
        snr_bin = str(obs.get("snr_bin")) if obs and obs.get("snr_bin") else snr_db_to_bin_label(float(link["sinr_db"]), self.snr_bins_db)
        cache_probability = self._semantic_cache_hit_probability(task)
        entry = self._lookup_entry(task, int(parsed["service_level"]), snr_bin)
        semantic_estimate = self._semantic_estimate(task, int(parsed["service_level"]), snr_bin, entry)
        accuracy = float(semantic_estimate.accuracy_mean)
        if int(parsed["service_level"]) == 0:
            accuracy *= 0.85 + 0.15 * cache_probability
        semantic_lcb = float(semantic_estimate.accuracy_lcb)
        cache_status = self._cache_status(task)
        if str(parsed["semantic_path"]) == "cache":
            semantic_lcb = float(cache_status["cache_quality_lcb"]) if bool(cache_status["cache_eligible"]) else 0.0
            accuracy = max(accuracy, semantic_lcb) if bool(cache_status["cache_eligible"]) else 0.0
        semantic_payload_kb = float(semantic_estimate.payload_kb)
        payload_bytes = semantic_payload_kb * 1024.0
        delay = self._delay_parts(task, uav, edge, parsed, payload_bytes, link)
        strategic_conflict_task_ids = self._strategic_conflict_task_ids(task, parsed)
        airspace_conflict = bool(strategic_conflict_task_ids)
        utm = self._utm_evaluation(task, parsed, strategic_conflict_task_ids)
        delay["utm_dss_delay_s"] = float(utm["dss_delay_s"])
        delay["utm_notification_delay_s"] = float(utm["subscription_notification_delay_s"])
        delay["total_delay_s"] += delay["utm_dss_delay_s"] + delay["utm_notification_delay_s"]
        energy = self._energy_parts(delay, parsed)
        remaining_deadline_s = self._remaining_deadline_s(task)
        semantic_quality_gap = max(0.0, task.epsilon_k - semantic_lcb)
        semantic_success = semantic_lcb >= task.epsilon_k
        quality_violation = not semantic_success
        if str(parsed["semantic_path"]) == "cache" and not bool(cache_status["cache_eligible"]):
            quality_violation = True
            semantic_success = False
        # Structural cache-compliance ban (task #28 v3, method (c)).  When the
        # env is configured with critical_cache_compliance == "forbidden", a
        # critical/high-risk task served by the s0 cache-only path is NEVER
        # counted as quality-compliant, even if the cached answer's LCB clears
        # epsilon_k.  This closes the cache shortcut at the compliance-judgment
        # layer, so it propagates uniformly to success, the reward violation
        # term, the Lagrangian quality cost, and the oracle's per-task feasible-
        # service search.  Default "allowed" keeps legacy/v1/v2 bit-identical.
        if (
            str(parsed["semantic_path"]) == "cache"
            and task.risk_level in ("critical", "high")
            and str(self.env_cfg.get("critical_cache_compliance", "allowed") or "allowed").lower()
            == "forbidden"
            # Change 5: when the escalation layer is armed the cache ban only bites
            # tasks that WERE spec-attainable (transmission was possible) -- a task
            # with no feasible transmission is not "gaming" by falling back to cache.
            and (not self._escalation_enabled() or bool(task.spec_attainable))
        ):
            quality_violation = True
            semantic_success = False
        # Task #33-A: charge only the comm-decision window against the deadline
        # under deadline_semantics="comm_window" (fly/positioning -> tasking layer);
        # legacy charges the full end-to-end delay.  Single chokepoint.
        deadline_charged_delay_s = self._deadline_delay_s(delay)
        deadline_violation = deadline_charged_delay_s > remaining_deadline_s
        battery_violation = energy["total_energy_j"] > uav.battery_j - float(self.env_cfg["return_energy_reserve_j"])
        gpu_memory_ok = self._gpu_memory_ok(edge, int(parsed["service_level"]))
        utm_constraint_violation = bool(utm["utm_constraint_violation"])
        risk_violation = bool(task.risk_level == "critical" and (quality_violation or deadline_violation or utm_constraint_violation))
        success = not (
            quality_violation
            or deadline_violation
            or battery_violation
            or airspace_conflict
            or utm_constraint_violation
            or not gpu_memory_ok
        )
        payload_kb = semantic_payload_kb
        route = self.semantic_service_route(task, parsed, entry, cache_probability)
        utility = self.semantic_utility(
            task=task,
            accuracy=accuracy,
            payload_kb=payload_kb,
            delay_s=delay["total_delay_s"],
            energy_j=energy["total_energy_j"],
            success=success,
        )
        info = {
            "episode": self.episode,
            "episode_step": self.step_count,
            "scenario": self.scenario_name,
            "formal_scenario": self.formal_scenario_name,
            "benchmark_split": self.benchmark_split,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "question_type": task.task_type,
            "risk_level": task.risk_level,
            "object_count": int(task.object_count),
            "count_bucket": self._count_bucket_for_task(task),
            "task_status": task.task_status,
            "remaining_deadline_s": round(float(remaining_deadline_s), 6),
            "defer_count": int(task.defer_count),
            "expired": bool(task.expired),
            "view_quality_bin": task.view_quality_bin,
            "freshness_bin": task.freshness_bin,
            "deadline_s": round(float(task.tau_k), 6),
            "epsilon_k": round(float(task.epsilon_k), 6),
            "priority": round(float(task.priority), 6),
            "uav_id": uav.uav_id,
            "edge_id": edge.edge_id,
            "answer_accuracy_est": round(accuracy, 6),
            "semantic_accuracy_mean": round(accuracy, 6),
            "semantic_accuracy_lcb": round(semantic_lcb, 6),
            "semantic_uncertainty": round(float(semantic_estimate.uncertainty), 6),
            "semantic_sample_count": int(semantic_estimate.sample_count),
            "semantic_payload_kb": round(semantic_payload_kb, 6),
            "semantic_quality_gap": round(semantic_quality_gap, 6),
            "semantic_success": bool(semantic_success),
            "semantic_path": str(parsed["semantic_path"]),
            "delay_s": round(delay["total_delay_s"], 6),
            "deadline_charged_delay_s": round(float(deadline_charged_delay_s), 6),
            "energy_j": round(energy["total_energy_j"], 6),
            "payload_kb": round(payload_kb, 6),
            "quality_violation": bool(quality_violation),
            "deadline_violation": bool(deadline_violation),
            # Task #34-(ii): surface the stored spec-attainability certificate on
            # the SERVED-task record too (previously only reject/expired/obs paths
            # carried it, so spec_attainable_rate read ~0 for serving policies).
            "spec_attainable": bool(task.spec_attainable),
            "battery_violation": bool(battery_violation),
            "resource_violation": bool(not gpu_memory_ok),
            "risk_violation": bool(risk_violation),
            "airspace_conflict": bool(airspace_conflict),
            "utm_constraint_violation": bool(utm_constraint_violation),
            "utm_conflict_violation": bool(airspace_conflict or utm_constraint_violation),
            "operational_intent_id": task.operational_intent_id,
            "operational_intent_state": str(utm["operational_intent_state"]),
            "airspace_state": str(utm["operational_intent_state"]),
            "operational_priority": round(float(task.operational_priority), 6),
            "strategic_conflict": bool(airspace_conflict),
            "strategic_conflict_count": int(utm["strategic_conflict_count"]),
            "strategic_conflict_task_ids": ";".join(str(item) for item in strategic_conflict_task_ids),
            "utm_spatial_buffer_m": round(float(utm["spatial_buffer_m"]), 6),
            "utm_altitude_buffer_m": round(float(utm["altitude_buffer_m"]), 6),
            "utm_temporal_buffer_steps": int(utm["temporal_buffer_steps"]),
            "dss_available": bool(utm["dss_available"]),
            "dss_delay_s": round(float(utm["dss_delay_s"]), 6),
            "utm_delay_s": round(delay["utm_dss_delay_s"] + delay["utm_notification_delay_s"], 6),
            "subscription_notification_delay_s": round(float(utm["subscription_notification_delay_s"]), 6),
            "conflict_notification_pending": bool(utm["conflict_notification_pending"]),
            "snr_bin": snr_bin,
            "sensed_snr_db": round(float(link["snr_db"]), 6),
            "sinr_db": round(float(link["sinr_db"]), 6),
            "rate_mbps": round(float(link["rate_mbps"]), 6),
            "service_level": int(parsed["service_level"]),
            "bandwidth_hz": round(float(parsed["bandwidth"]), 6),
            "power_w": round(float(parsed["power"]), 6),
            "cpu_share": round(float(parsed["cpu_share"]), 6),
            "gpu_share": round(float(parsed["gpu_share"]), 6),
            "uav_assignment": int(parsed["uav_assignment"]),
            "mobility_mode": str(parsed["mobility_mode"]),
            "waypoint_x": round(float(delay["waypoint_x"]), 6),
            "waypoint_y": round(float(delay["waypoint_y"]), 6),
            "altitude_m": round(float(delay["target_altitude_m"]), 6),
            "fly_distance_m": round(float(delay["fly_distance_m"]), 6),
            "coverage_gain": round(float(delay["coverage_gain"]), 6),
            "mobility_energy_j": round(float(energy["mobility_energy_j"]), 6),
            "arrival_delay_s": round(float(delay["arrival_delay_s"]), 6),
            "utm_conflict_risk": round(float(delay["utm_conflict_risk"]), 6),
            "semantic_service_name": route["service_name"],
            "semantic_evidence_type": route["evidence_type"],
            "semantic_utility": round(float(utility["semantic_utility"]), 6),
            "semantic_efficiency": round(float(utility["semantic_efficiency"]), 6),
            "success": bool(success),
            "distance_3d_m": round(float(link["distance_3d_m"]), 6),
            "elevation_deg": round(float(link["elevation_deg"]), 6),
            "los_probability": round(float(link["los_probability"]), 6),
            "path_loss_db": round(float(link["path_loss_db"]), 6),
            "interference_dbm": round(float(link["interference_dbm"]), 6),
            "fading_db": round(float(link["fading_db"]), 6),
            "cache_hit_probability": round(cache_probability, 6),
            **cache_status,
            "semantic_cache_hit": bool(cache_probability >= 0.5 and int(parsed["service_level"]) == 0),
            "gpu_memory_ok": bool(gpu_memory_ok),
            "gpu_memory_used_mb": round(edge.gpu_memory_used_mb, 6),
            "gpu_memory_capacity_mb": round(edge.gpu_memory_capacity_mb, 6),
            "battery_remaining_j": round(max(0.0, uav.battery_j - energy["total_energy_j"]), 6),
            "bandwidth_unit": "Hz",
            "power_unit": "W",
        }
        info.update({key: round(value, 6) for key, value in delay.items()})
        info.update({key: round(value, 6) for key, value in energy.items()})
        if mutate:
            self.last_info = info
        return info

    def _escalation_verdict(self, task: EnvTask, base_quality_violation: bool) -> tuple[bool, bool]:
        """Change 5: resolve (quality_violation, escalated) for a reject/expired
        critical/high task.  Legacy (escalation off) returns the base verdict with
        escalated=False.  Under the escalation layer a spec-attainable task that is
        rejected/expired is a genuine quality violation (the policy dropped a
        serveable task -> full price); a spec-UNattainable one is escalated (no
        quality cost, routed to the escalation channel)."""
        if not self._escalation_enabled() or str(task.risk_level) not in ("critical", "high"):
            return bool(base_quality_violation), False
        if bool(task.spec_attainable):
            return True, False
        return False, True

    def _reject_info(self, task: EnvTask, action: dict[str, Any]) -> dict[str, Any]:
        service_paths = ("cache", "token", "image", "cache_update")
        if task.expired or self._remaining_deadline_s(task) <= 0.0:
            service_infos: list[dict[str, Any]] = []
            reject_feasible = True
            reject_reason = "expired"
        else:
            service_infos = [
                self.evaluate_action(self._path_action(path, task), task_id=task.task_id, obs={"task_id": task.task_id})
                for path in service_paths
            ]
            feasible_infos = [
                info
                for info in service_infos
                if not bool(info.get("quality_violation", False))
                and not bool(info.get("deadline_violation", False))
                and not bool(info.get("battery_violation", False))
                and not bool(info.get("resource_violation", False))
                and not bool(info.get("utm_conflict_violation", False))
            ]
            reject_feasible = not feasible_infos
            reject_reason = self._reject_reason(service_infos)
        expected_saved_delay_s = min((float(info.get("delay_s", 0.0)) for info in service_infos), default=0.0)
        expected_saved_energy_j = min((float(info.get("energy_j", 0.0)) for info in service_infos), default=0.0)
        cache_status = self._cache_status(task)
        # Change 5: legacy reject quality_violation base is False; escalation layer
        # re-resolves it for critical/high tasks by spec-attainability.
        _reject_quality_violation, _reject_escalated = self._escalation_verdict(task, base_quality_violation=False)
        if _reject_escalated:
            task.escalated = True
        info = self._empty_info()
        info.update(
            {
                "episode": self.episode,
                "episode_step": self.step_count,
                "scenario": self.scenario_name,
                "formal_scenario": self.formal_scenario_name,
                "benchmark_split": self.benchmark_split,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "question_type": task.task_type,
                "risk_level": task.risk_level,
                "view_quality_bin": task.view_quality_bin,
                "freshness_bin": task.freshness_bin,
                "deadline_s": round(float(task.tau_k), 6),
                "remaining_deadline_s": round(float(self._remaining_deadline_s(task)), 6),
                "epsilon_k": round(float(task.epsilon_k), 6),
                "priority": round(float(task.priority), 6),
                "task_status": "rejected",
                "defer_count": int(task.defer_count),
                "expired": bool(task.expired),
                "rejected": True,
                "semantic_path": "reject",
                "service_level": -1,
                "sensing_decision": str(action.get("sensing_decision", "reject")),
                "answer_accuracy_est": 0.0,
                "semantic_accuracy_mean": 0.0,
                "semantic_accuracy_lcb": 0.0,
                "semantic_success": False,
                "success": False,
                "quality_violation": _reject_quality_violation,
                "escalated": _reject_escalated,
                "escalation": float(_reject_escalated),
                "spec_attainable": bool(task.spec_attainable),
                "deadline_violation": False,
                "risk_violation": False,
                "resource_violation": False,
                "utm_constraint_violation": False,
                "utm_conflict_violation": False,
                "reject_feasible": bool(reject_feasible),
                "reject_reason": reject_reason,
                "expected_saved_energy_j": round(expected_saved_energy_j, 6),
                "expected_saved_delay_s": round(expected_saved_delay_s, 6),
                "avoided_utm_violation": any(bool(item.get("utm_conflict_violation", False)) for item in service_infos),
                "avoided_deadline_violation": any(bool(item.get("deadline_violation", False)) for item in service_infos),
                "reject_penalty": round(abs(self._reject_reward(task, {"reject_feasible": reject_feasible})), 6),
                "delay_s": 0.0,
                "energy_j": 0.0,
                "payload_kb": 0.0,
                "tx_delay_s": 0.0,
                "fly_delay_s": 0.0,
                "arrival_delay_s": 0.0,
                "semantic_payload_kb": 0.0,
                "cache_hit_probability": round(self._semantic_cache_hit_probability(task), 6),
                **cache_status,
            }
        )
        return info

    @staticmethod
    def _reject_reason(service_infos: list[dict[str, Any]]) -> str:
        reasons: set[str] = set()
        for info in service_infos:
            if bool(info.get("quality_violation", False)):
                reasons.add("semantic_quality")
            if bool(info.get("deadline_violation", False)):
                reasons.add("deadline")
            if bool(info.get("utm_conflict_violation", False)):
                reasons.add("utm")
            if bool(info.get("battery_violation", False)):
                reasons.add("energy")
            if bool(info.get("resource_violation", False)):
                reasons.add("resource")
        if not reasons:
            return "mixed"
        return next(iter(reasons)) if len(reasons) == 1 else "mixed"

    def _expired_task_info(self, task: EnvTask, action: dict[str, Any]) -> dict[str, Any]:
        cache_status = self._cache_status(task)
        # Change 5: legacy expired quality_violation base is True; escalation layer
        # re-resolves it for critical/high tasks by spec-attainability.
        _exp_quality_violation, _exp_escalated = self._escalation_verdict(task, base_quality_violation=True)
        if _exp_escalated:
            task.escalated = True
        info = self._empty_info()
        info.update(
            {
                "episode": self.episode,
                "episode_step": self.step_count,
                "scenario": self.scenario_name,
                "formal_scenario": self.formal_scenario_name,
                "benchmark_split": self.benchmark_split,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "question_type": task.task_type,
                "risk_level": task.risk_level,
                "view_quality_bin": task.view_quality_bin,
                "freshness_bin": task.freshness_bin,
                "deadline_s": round(float(task.tau_k), 6),
                "remaining_deadline_s": 0.0,
                "epsilon_k": round(float(task.epsilon_k), 6),
                "priority": round(float(task.priority), 6),
                "task_status": "expired",
                "defer_count": int(task.defer_count),
                "expired": True,
                "semantic_path": str(action.get("semantic_path", SERVICE_LEVEL_TO_SEMANTIC_PATH.get(int(action.get("service_level", 0)), "cache"))),
                "service_level": int(action.get("service_level", 0)),
                "quality_violation": _exp_quality_violation,
                "escalated": _exp_escalated,
                "escalation": float(_exp_escalated),
                "spec_attainable": bool(task.spec_attainable),
                "deadline_violation": True,
                "risk_violation": bool(task.risk_level == "critical" and _exp_quality_violation),
                "success": False,
                "cache_hit_probability": round(self._semantic_cache_hit_probability(task), 6),
                **cache_status,
            }
        )
        return info

    def _defer_info(self, task: EnvTask, action: dict[str, Any]) -> dict[str, Any]:
        remaining = self._remaining_deadline_s(task)
        expired = remaining <= 0.0 or bool(task.expired)
        cache_status = self._cache_status(task)
        info = self._empty_info()
        info.update(
            {
                "episode": self.episode,
                "episode_step": self.step_count,
                "scenario": self.scenario_name,
                "formal_scenario": self.formal_scenario_name,
                "benchmark_split": self.benchmark_split,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "question_type": task.task_type,
                "risk_level": task.risk_level,
                "view_quality_bin": task.view_quality_bin,
                "freshness_bin": task.freshness_bin,
                "deadline_s": round(float(task.tau_k), 6),
                "remaining_deadline_s": round(float(remaining), 6),
                "epsilon_k": round(float(task.epsilon_k), 6),
                "priority": round(float(task.priority), 6),
                "task_status": "expired" if expired else "deferred",
                "defer_count": int(task.defer_count),
                "expired": bool(expired),
                "semantic_path": "defer",
                "service_level": 0,
                "sensing_decision": str(action.get("sensing_decision", "defer")),
                "quality_violation": False,
                "deadline_violation": bool(expired),
                "success": False,
                "cache_hit_probability": round(self._semantic_cache_hit_probability(task), 6),
                **cache_status,
            }
        )
        return info

    def semantic_service_route(
        self,
        task: EnvTask,
        action: dict[str, Any],
        entry: LUTEntry,
        cache_probability: float | None = None,
    ) -> dict[str, Any]:
        level = int(action["service_level"])
        meta = SERVICE_LEVELS.get(level, SERVICE_LEVELS[0])
        assigned_uav = int(action.get("uav_assignment", self._nearest_uav(task).uav_id))
        edge_id = int(action.get("edge_id", 0))
        cache_probability = self._semantic_cache_hit_probability(task) if cache_probability is None else float(cache_probability)
        return {
            "task_id": task.task_id,
            "service_level": level,
            "service_name": meta["name"],
            "evidence_type": meta["evidence_type"],
            "uses_cache": level == 0,
            "requires_uav": bool(meta["requires_uav"]),
            "requires_edge_model": bool(meta["requires_edge_model"]),
            "assigned_uav": assigned_uav,
            "edge_id": edge_id,
            "cache_freshness": task.freshness_bin,
            "cache_hit_probability": round(cache_probability, 6),
            "expected_accuracy": round(float(entry.accuracy), 6),
            "payload_kb": round(float(entry.payload_bytes) / 1024.0, 6),
        }

    @staticmethod
    def semantic_utility(
        task: EnvTask,
        accuracy: float,
        payload_kb: float,
        delay_s: float,
        energy_j: float,
        success: bool,
    ) -> dict[str, float]:
        semantic_gain = float(task.priority) * float(accuracy)
        deadline_factor = 1.0 if delay_s <= task.tau_k else max(0.0, task.tau_k / max(1e-9, delay_s))
        quality_shortfall = max(0.0, task.epsilon_k - float(accuracy))
        resource_cost = 0.01 * float(delay_s) + 0.0001 * float(energy_j) + 0.001 * float(payload_kb)
        utility = semantic_gain * deadline_factor - resource_cost - quality_shortfall
        if success:
            utility += 0.5 * float(task.priority)
        efficiency = semantic_gain / max(1.0, float(payload_kb))
        return {
            "semantic_gain": semantic_gain,
            "deadline_factor": deadline_factor,
            "quality_shortfall": quality_shortfall,
            "resource_cost": resource_cost,
            "semantic_utility": utility,
            "semantic_efficiency": efficiency,
        }

    @staticmethod
    def semantic_utility_schema() -> dict[str, Any]:
        return {
            "inputs": ["task priority", "expected accuracy", "payload_kb", "delay_s", "energy_j", "success"],
            "outputs": ["semantic_gain", "deadline_factor", "quality_shortfall", "resource_cost", "semantic_utility", "semantic_efficiency"],
            "note": "Accuracy and payload are LUT outputs; delay/energy are network simulator outputs.",
        }

    def service_levels(self) -> list[int]:
        measured = sorted({level for _q, level, _snr, _view, _fresh, _risk in self.lut})
        enabled = [int(level) for level in self.env_cfg.get("enabled_service_levels", measured)]
        if not bool(self.env_cfg.get("enable_service_level_3", False)):
            enabled = [level for level in enabled if level != 3]
        levels = [level for level in enabled if level in measured]
        if not levels:
            levels = [level for level in measured if level != 3] or measured
        return sorted(levels)

    def action_spec(self) -> dict[str, Any]:
        return {
            "type": "hybrid",
            "service_levels": self.service_levels(),
            "semantic_paths": list(SEMANTIC_PATHS),
            "semantic_path_to_service_level": dict(SEMANTIC_PATH_TO_SERVICE_LEVEL),
            "mobility_modes": list(MOBILITY_MODES),
            "num_uavs": len(self.uavs),
            "num_edges": len(self.edges),
            "high_level": ["task_id", "semantic_path", "uav_assignment", "edge_id", "sensing_decision", "mobility_mode"],
            "low_level_discrete": ["semantic_path", "service_level", "mobility_mode"],
            "low_level_continuous": ["bandwidth", "power", "cpu_share", "gpu_share", "waypoint_delta", "altitude_delta"],
            "constraint_mask": "action_mask",
        }

    def action_mask(self) -> dict[str, Any]:
        active = self._active_tasks()
        memory_free_by_edge = {
            edge.edge_id: max(0.0, edge.gpu_memory_capacity_mb - edge.gpu_memory_used_mb)
            for edge in self.edges
        }
        service_allowed: dict[int, bool] = {}
        for level in self.service_levels():
            if level == 0:
                service_allowed[level] = True
                continue
            memory = float(self.env_cfg["model_memory_mb_by_level"].get(str(level), 0.0))
            service_allowed[level] = any(
                level in edge.cached_service_levels or memory <= memory_free_by_edge[edge.edge_id] + 1e-9
                for edge in self.edges
            )
        return {
            "active_task_ids": [task.task_id for task in active],
            "active_task_mask": [1.0 for _ in active],
            "service_level_allowed": service_allowed,
            "semantic_path_allowed": {path: True for path in SEMANTIC_PATHS},
            "mobility_mode_allowed": {mode: True for mode in MOBILITY_MODES},
            "feasible_mobility_mask": self._feasible_mobility_mask(active),
            "uav_battery_ok": {
                uav.uav_id: uav.battery_j > float(self.env_cfg["return_energy_reserve_j"])
                for uav in self.uavs
            },
            "edge_memory_free_mb": memory_free_by_edge,
            "resource_budget_hint": {
                "bandwidth_hz": float(self.env_cfg["bandwidth_hz"]),
                "fair_bandwidth_hz": float(self.env_cfg["bandwidth_hz"]) / max(1, len(active)),
                "fair_cpu_share": 1.0 / max(1, len(active)),
                "fair_gpu_share": 1.0 / max(1, len(active)),
            },
        }

    def _init_uavs(self) -> list[UAVNode]:
        num = int(self.env_cfg["num_uavs"])
        radius = float(self.env_cfg["area_spacing_m"])
        out: list[UAVNode] = []
        for uid in range(num):
            angle = 2.0 * math.pi * uid / max(1, num)
            x_m = radius * math.cos(angle)
            y_m = radius * math.sin(angle)
            altitude_m = float(self.env_cfg["uav_altitude_m"])
            out.append(
                UAVNode(
                    uav_id=uid,
                    x_m=x_m,
                    y_m=y_m,
                    altitude_m=altitude_m,
                    battery_j=float(self.env_cfg["initial_battery_j"]),
                    speed_mps=float(self.env_cfg["uav_speed_mps"]),
                    base_x_m=x_m,
                    base_y_m=y_m,
                    base_altitude_m=altitude_m,
                )
            )
        return out

    def _init_edges(self) -> list[EdgeNode]:
        out: list[EdgeNode] = []
        capacity = float(self.env_cfg["gpu_memory_capacity_mb"])
        base_memory = capacity * float(self.env_cfg["gpu_memory_load"])
        edge_load_range = self.env_cfg.get("edge_load_range", [0.05, 0.35])
        gpu_load_range = self.env_cfg.get("gpu_load_range", [0.03, 0.25])
        for edge_id in range(int(self.env_cfg["num_edges"])):
            cached = (1,) if 1 in self.service_levels() else ()
            used = base_memory + sum(float(self.env_cfg["model_memory_mb_by_level"].get(str(level), 0.0)) for level in cached)
            out.append(
                EdgeNode(
                    edge_id=edge_id,
                    load=round(self.rng.uniform(float(edge_load_range[0]), float(edge_load_range[1])), 6),
                    gpu_load=round(self.rng.uniform(float(gpu_load_range[0]), float(gpu_load_range[1])), 6),
                    cached_service_levels=cached,
                    model_cache_capacity=int(self.env_cfg["model_cache_capacity"]),
                    gpu_memory_capacity_mb=capacity,
                    gpu_memory_used_mb=min(capacity, used),
                )
            )
        return out

    def _init_tasks(self, options: dict[str, Any]) -> list[EnvTask]:
        layout = dict(self.scenario_cfg.get("task_layout", {}))
        if _bubbles_profile_active(self.env_cfg) and "generation_mode" not in layout:
            # BUBBLES profile defaults to the Table G-13 daily demand curve
            # unless the active scenario already pins a generation mode.
            layout["generation_mode"] = "bubbles_daily"
        count = int(options.get("tasks_per_episode", self.env_cfg["tasks_per_episode"]))
        self._episode_task_count = count
        selected = [self.raw_tasks[self.rng.randrange(len(self.raw_tasks))] for _ in range(count)]
        tasks: list[EnvTask] = []
        spacing = float(self.env_cfg["area_spacing_m"])
        num_areas = int(self.env_cfg["num_areas"])
        width = max(1, int(math.ceil(math.sqrt(num_areas))))
        jitter_ratio = float(layout.get("jitter_ratio", 0.2))
        force_same_area = bool(layout.get("force_same_area", False))
        for idx, row in enumerate(selected):
            area_id = 0 if force_same_area else idx % max(1, num_areas)
            grid_x = area_id % width
            grid_y = area_id // width
            jitter_x = self.rng.uniform(-jitter_ratio, jitter_ratio) * spacing
            jitter_y = self.rng.uniform(-jitter_ratio, jitter_ratio) * spacing
            risk = self._cycled_value(layout.get("risk_cycle"), idx, row.get("risk_level", "normal"))
            generation = self._generation_step(idx, area_id, layout)
            tau = self._tau_for_task(row, risk, layout)
            if "tau_floor_s" in layout:
                tau = max(tau, float(layout["tau_floor_s"]))
            freshness = self._cycled_value(layout.get("freshness_cycle"), idx, row.get("freshness_bin") or "")
            if freshness:
                cache_age = self._cache_age_for_freshness(freshness)
            else:
                cache_age = self.rng.randrange(0, 4)
                freshness = self._freshness_from_age(cache_age)
            view_quality = self._cycled_value(layout.get("view_quality_cycle"), idx, row.get("view_quality_bin", "medium"))
            area = Area4D(
                center_x_m=(grid_x - (width - 1) / 2.0) * spacing + jitter_x,
                center_y_m=(grid_y - (width - 1) / 2.0) * spacing + jitter_y,
                radius_m=float(self.env_cfg["area_radius_m"]),
                altitude_min_m=float(self.env_cfg["area_altitude_min_m"]),
                altitude_max_m=float(self.env_cfg["area_altitude_max_m"]),
                start_step=generation,
                end_step=generation + max(1, int(math.ceil(tau))),
            )
            tasks.append(
                EnvTask(
                    task_id=str(row.get("task_id") or row.get("image_id") or f"task_{idx}") + f"_{idx}",
                    task_type=row.get("question_type", "presence"),
                    question=row.get("question", ""),
                    risk_level=risk,
                    object_count=_safe_int(row.get("object_count", -1), -1),
                    epsilon_k=self._epsilon_for_task(row, risk),
                    tau_k=tau,
                    priority=2.0 if risk == "critical" else 1.0,
                    view_quality_bin=view_quality,
                    freshness_bin=freshness,
                    generation_time=generation,
                    area_id=area_id,
                    area4d=area,
                    cache_age=cache_age,
                    operational_intent_id=f"oi_ep{self.episode}_task{idx}_area{area_id}",
                    operational_intent_state=self._initial_operational_intent_state(idx, risk),
                    operational_priority=2.0 if risk == "critical" else 1.0,
                )
            )
        return tasks

    def _tau_for_task(self, row: dict[str, str], risk: str, layout: dict[str, Any]) -> float:
        """Task #33-B: per-task deadline tau_k.

        LEGACY: tau_k = CSV tau_k (fallback 3.0 critical / 5.0 normal) x tau_scale
        -- unchanged, so v1-v5 stay bit-for-bit.

        COMM_WINDOW: tau_k is the tactical communication-decision window anchored
        on BUBBLES D2.1 Table G-2 separation-communication N(1.8, sigma 1.0):
        critical=2.8 s (1-sigma), normal=3.8 s (2-sigma), overridable from
        config thresholds {tau_critical_comm, tau_normal_comm} or from the layout
        {tau_comm_by_risk}.  tau_scale is deliberately NOT applied -- the anchor
        already IS the window (its role under comm_window is documented as待校验
        in docs/SCALE_CONSISTENCY_V6.md; layout may still override via
        tau_comm_scale if a study needs it).
        """
        if self._deadline_semantics() == "comm_window":
            # thresholds live at the TOP level of the config (self.cfg), not in
            # the multi_uav_env sub-block (self.env_cfg); read both so either
            # placement works.
            thresholds = dict(self.cfg.get("thresholds", {}) or {})
            thresholds.update(self.env_cfg.get("thresholds", {}) or {})
            by_risk = dict(layout.get("tau_comm_by_risk", {}) or {})
            default = COMM_WINDOW_TAU.get(risk, COMM_WINDOW_TAU["normal"])
            cfg_key = "tau_critical_comm" if risk in ("critical", "high") else "tau_normal_comm"
            tau = float(by_risk.get(risk, thresholds.get(cfg_key, default)))
            return tau * float(layout.get("tau_comm_scale", 1.0))
        return float(row.get("tau_k", 3.0 if risk == "critical" else 5.0)) * float(layout.get("tau_scale", 1.0))

    def _epsilon_for_task(self, row: dict[str, str], risk: str) -> float:
        mode = str(self.env_cfg.get("epsilon_calibration", "legacy") or "legacy").lower()
        if mode == "attainability_v5":
            qtype = str(row.get("question_type", "presence"))
            bucket = count_bucket_v5(_safe_int(row.get("object_count", -1), -1)) if qtype == "counting" else "na"
            klass = epsilon_v5_class(qtype, bucket)
            key = (risk, klass)
            if key not in ATTAINABILITY_V5_EPSILON:
                key = ("normal", klass) if (risk, "presence") not in ATTAINABILITY_V5_EPSILON else (risk, "presence")
            return float(ATTAINABILITY_V5_EPSILON.get(key, ATTAINABILITY_V5_EPSILON[("critical", "presence")]))
        if mode == "attainability_v4":
            table = ATTAINABILITY_V4_EPSILON
            return float(table.get(risk, table["critical"]))
        if mode == "attainability_v3":
            table = ATTAINABILITY_V3_EPSILON
            return float(table.get(risk, table["critical"]))
        if mode == "attainability_v2":
            table = ATTAINABILITY_V2_EPSILON
            return float(table.get(risk, table["critical"]))
        if mode in ("attainability_v1", "attainability"):
            table = ATTAINABILITY_V1_EPSILON
            return float(table.get(risk, table["critical"]))
        default = 0.82 if risk == "critical" else 0.65
        base = float(row.get("epsilon_k", default))
        thresholds = self.env_cfg.get("semantic_threshold_by_risk", {})
        floor = float(thresholds.get(risk, thresholds.get("normal", 0.0)))
        epsilon = max(base, floor)
        caps = self.env_cfg.get("semantic_threshold_cap_by_risk", {})
        if risk in caps:
            epsilon = min(epsilon, float(caps[risk]))
        elif "normal" in caps:
            epsilon = min(epsilon, float(caps["normal"]))
        layout = self.scenario_cfg.get("task_layout", {})
        epsilon *= float(layout.get("epsilon_scale", 1.0))
        layout_caps = layout.get("epsilon_cap_by_risk", {})
        if isinstance(layout_caps, dict):
            if risk in layout_caps:
                epsilon = min(epsilon, float(layout_caps[risk]))
            elif "normal" in layout_caps:
                epsilon = min(epsilon, float(layout_caps["normal"]))
        return epsilon

    def _initial_operational_intent_state(self, idx: int, risk: str) -> str:
        layout = self.scenario_cfg.get("task_layout", {})
        explicit = layout.get("operational_state_cycle")
        if isinstance(explicit, list) and explicit:
            state = str(explicit[idx % len(explicit)])
            return state if state in OPERATIONAL_INTENT_STATES else "accepted"
        mode = str(self.env_cfg.get("utm", {}).get("mode", "nominal_planning"))
        if mode == "dss_outage":
            return "contingent"
        if mode == "off_nominal_planning" and risk == "critical":
            return "nonconforming"
        return "accepted"

    def _generation_step(self, idx: int, area_id: int, layout: dict[str, Any]) -> int:
        mode = str(layout.get("generation_mode", "staggered"))
        if mode == "bubbles_daily":
            # D2.1 Table G-13 (p.128) daily demand curve scaled to the episode.
            count = int(getattr(self, "_episode_task_count", self.env_cfg["tasks_per_episode"]))
            return bubbles_separation.bubbles_daily_generation_step(
                idx, count, int(self.env_cfg["episode_steps"])
            )
        if mode == "burst":
            return int(layout.get("burst_start_step", 0))
        if mode == "wave":
            return int(idx // max(1, int(self.env_cfg["num_areas"]))) % max(1, int(self.env_cfg["episode_steps"]) // 2)
        return idx % max(1, int(self.env_cfg["episode_steps"]) // 2)

    @staticmethod
    def _cycled_value(values: Any, idx: int, fallback: str) -> str:
        if isinstance(values, list) and values:
            return str(values[idx % len(values)])
        return str(fallback)

    def _cache_age_for_freshness(self, freshness: str) -> int:
        thresholds = self.env_cfg["freshness_slots"]
        if freshness == "fresh":
            return 0
        if freshness == "stale":
            return int(thresholds["fresh"]) + 1
        if freshness == "expired":
            return int(thresholds["stale"]) + 1
        return 0

    def _init_scenario_semantic_cache(self) -> list[SemanticCacheEntry]:
        seed_cfg = self.scenario_cfg.get("semantic_cache_seed", {})
        if not bool(seed_cfg.get("enabled", False)):
            return []
        entries_per_area = int(seed_cfg.get("entries_per_area", 1))
        cache_age = int(seed_cfg.get("cache_age", 0))
        entries: list[SemanticCacheEntry] = []
        seen: dict[int, int] = {}
        for task in self.tasks:
            count = seen.get(task.area_id, 0)
            if count >= entries_per_area:
                continue
            seen[task.area_id] = count + 1
            entries.append(
                SemanticCacheEntry(
                    task_id=f"scenario_cache_{task.area_id}_{count}",
                    task_type=task.task_type,
                    risk_level=task.risk_level,
                    priority=task.priority,
                    x_m=task.x_m,
                    y_m=task.y_m,
                    cache_age=cache_age,
                    updated_step=-1,
                    area_id=task.area_id,
                    question_type=task.task_type,
                    quality_lcb=max(float(task.epsilon_k), 0.75),
                    uncertainty=0.05,
                )
            )
        entries.sort(key=lambda item: (-item.priority, item.cache_age, item.task_id))
        return entries[: int(self.env_cfg["semantic_cache_capacity"])]

    def _snr_bins_from_lut(self) -> list[float]:
        bins: list[float] = []
        for _qtype, _level, link_quality, _view, _fresh, _risk in self.lut:
            try:
                bins.append(snr_db_from_label(link_quality))
            except ValueError:
                continue
        return sorted(dict.fromkeys(bins))

    def _active_tasks(self) -> list[EnvTask]:
        return [
            task
            for task in self.tasks
            if not task.completed and not task.expired and not task.rejected and task.generation_time <= self.step_count
        ]

    def _front_task(self) -> EnvTask | None:
        active = self._active_tasks()
        if active:
            return min(active, key=lambda task: (-(task.priority / max(0.1, task.tau_k)), task.generation_time))
        pending = [task for task in self.tasks if not task.completed and not task.expired and not task.rejected]
        return min(pending, key=lambda task: task.generation_time) if pending else None

    def _select_task(self, action: dict[str, Any]) -> EnvTask | None:
        active = self._active_tasks()
        if not active:
            return None
        task_id = action.get("task_id")
        if task_id:
            for task in active:
                if task.task_id == task_id:
                    return task
        if "task_index" in action:
            return active[int(action["task_index"]) % len(active)]
        return self._front_task()

    def _task_by_id(self, task_id: str | None) -> EnvTask | None:
        if task_id is None:
            return None
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def _nearest_uav(self, task: EnvTask) -> UAVNode:
        return min(self.uavs, key=lambda uav: task.area4d.distance_to(uav.x_m, uav.y_m))

    def _service_level(self, action: dict[str, Any]) -> int:
        levels = self.service_levels()
        requested = int(action.get("service_level", levels[0]))
        if requested in levels:
            return requested
        return min(levels, key=lambda level: abs(level - requested))

    def _nearest_service_level(self, requested: int) -> int:
        levels = self.service_levels()
        if requested in levels:
            return requested
        return min(levels, key=lambda level: abs(level - requested))

    @staticmethod
    def _semantic_path(action: dict[str, Any]) -> str:
        raw = action.get("semantic_path", action.get("path", None))
        if raw is None:
            try:
                service_level = int(action.get("service_level", 0))
            except (TypeError, ValueError):
                service_level = 0
            return SERVICE_LEVEL_TO_SEMANTIC_PATH.get(service_level, "image")
        path = str(raw)
        return path if path in SEMANTIC_PATHS else SERVICE_LEVEL_TO_SEMANTIC_PATH.get(int(action.get("service_level", 0)), "cache")

    def _bandwidth_hz(self, action: dict[str, Any]) -> float:
        value = float(action.get("bandwidth", action.get("bandwidth_hz", self.env_cfg["bandwidth_hz"])))
        if 0.0 < value <= 1.0:
            value *= float(self.env_cfg["bandwidth_hz"])
        return max(1.0, value)

    @staticmethod
    def _power_w(action: dict[str, Any]) -> float:
        return max(1e-6, float(action.get("power", action.get("power_w", 0.1))))

    @staticmethod
    def _share(value: Any) -> float:
        return max(0.01, min(1.0, float(value)))

    def _mobility_mode(self, action: dict[str, Any], service_level: int) -> str:
        raw = action.get("mobility_mode")
        if raw is None:
            raw = "reposition" if action.get("waypoint") is not None else ("stay" if int(service_level) == 0 else "serve_task")
        mode = str(raw)
        return mode if mode in MOBILITY_MODES else ("stay" if int(service_level) == 0 else "serve_task")

    @staticmethod
    def _waypoint_delta(value: Any) -> tuple[float, float]:
        if isinstance(value, dict):
            return float(value.get("dx", value.get("x", 0.0))), float(value.get("dy", value.get("y", 0.0)))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        return 0.0, 0.0

    @staticmethod
    def _tx_dbm(power_w: float) -> float:
        return 30.0 + 10.0 * math.log10(max(1e-9, power_w))

    def _link_budget(
        self,
        task: EnvTask,
        uav: UAVNode,
        action: dict[str, Any],
        interference_dbm: float | None = None,
    ) -> dict[str, float]:
        a2g = self.env_cfg["a2g"]
        horizontal = task.area4d.distance_to(uav.x_m, uav.y_m)
        target_altitude = max(0.0, task.area4d.altitude_min_m)
        altitude_gap = max(1.0, abs(uav.altitude_m - target_altitude))
        distance_3d = max(float(a2g["reference_distance_m"]), math.hypot(horizontal, altitude_gap))
        elevation_deg = math.degrees(math.atan2(altitude_gap, max(1e-9, horizontal)))
        los_probability = self._los_probability(elevation_deg)
        expected_excess_loss = (
            los_probability * float(a2g["los_excess_loss_db"])
            + (1.0 - los_probability) * float(a2g["nlos_excess_loss_db"])
        )
        fading_db = self._fading_db(task)
        path_loss_db = (
            -float(a2g["reference_gain_db"])
            + 10.0 * float(a2g["path_loss_exponent"]) * math.log10(distance_3d / float(a2g["reference_distance_m"]))
            + expected_excess_loss
            + float(a2g["excess_loss_db"])
            - fading_db
        )
        gain_db = -path_loss_db
        bandwidth_hz = max(1.0, float(action["bandwidth"]))
        tx_dbm = self._tx_dbm(float(action["power"]))
        noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + float(a2g["noise_figure_db"])
        snr_db = tx_dbm + gain_db - noise_dbm
        floor = float(a2g["interference_floor_dbm"])
        interference_dbm = floor if interference_dbm is None else float(interference_dbm)
        noise_mw = 10.0 ** (noise_dbm / 10.0)
        interference_mw = 10.0 ** (interference_dbm / 10.0) if bool(a2g.get("interference_enabled", True)) else 0.0
        signal_mw = 10.0 ** ((tx_dbm + gain_db) / 10.0)
        sinr_linear = signal_mw / max(1e-18, noise_mw + interference_mw)
        sinr_db = 10.0 * math.log10(max(1e-18, sinr_linear))
        rate_mbps = bandwidth_hz * math.log2(1.0 + max(0.0, sinr_linear)) / 1_000_000.0
        return {
            "channel_gain_db": gain_db,
            "snr_db": snr_db,
            "sinr_db": sinr_db,
            "rate_mbps": rate_mbps,
            "distance_3d_m": distance_3d,
            "elevation_deg": elevation_deg,
            "los_probability": los_probability,
            "path_loss_db": path_loss_db,
            "interference_dbm": interference_dbm,
            "fading_db": fading_db,
        }

    def _interference_dbm(self, task: EnvTask, action: dict[str, Any], base_link: dict[str, float]) -> float:
        a2g = self.env_cfg["a2g"]
        total_mw = 10.0 ** (float(a2g["interference_floor_dbm"]) / 10.0)
        if not bool(a2g.get("interference_enabled", True)):
            return float(a2g["interference_floor_dbm"])
        for other_raw in action.get("concurrent_actions", []):
            other_task = self._task_by_id(other_raw.get("task_id")) or self._front_task()
            if other_task is None or other_task.task_id == task.task_id:
                continue
            other = self.parse_action(other_raw, other_task)
            if int(other["service_level"]) == 0:
                continue
            other_uav = self.uavs[int(other["uav_assignment"]) % len(self.uavs)]
            other_link = self._link_budget(other_task, other_uav, other, interference_dbm=float(a2g["interference_floor_dbm"]))
            overlap = min(float(action["bandwidth"]), float(other["bandwidth"])) / max(1.0, float(self.env_cfg["bandwidth_hz"]))
            overlap *= float(a2g.get("interference_overlap_scale", 0.02))
            if overlap <= 0.0:
                continue
            rx_dbm = self._tx_dbm(float(other["power"])) + float(other_link["channel_gain_db"])
            total_mw += overlap * 10.0 ** (rx_dbm / 10.0)
        return 10.0 * math.log10(max(1e-18, total_mw))

    def _los_probability(self, elevation_deg: float) -> float:
        a2g = self.env_cfg["a2g"]
        a = float(a2g["los_a"])
        b = float(a2g["los_b"])
        return 1.0 / (1.0 + a * math.exp(-b * (elevation_deg - a)))

    def _fading_db(self, task: EnvTask) -> float:
        a2g = self.env_cfg["a2g"]
        mode = str(a2g.get("fading_mode", "static"))
        if mode == "static":
            return 0.0
        base = self._deterministic_normal(f"{self.episode}:{task.task_id}:base")
        if mode == "fast_fading":
            slot_component = self._deterministic_normal(f"{self.episode}:{task.task_id}:{self.step_count}:fast")
            return float(a2g["fast_fading_std_db"]) * slot_component
        slot_component = self._deterministic_normal(f"{self.episode}:{task.task_id}:{self.step_count}:slow")
        rho = float(a2g.get("fading_correlation", 0.85))
        return float(a2g["slow_fading_std_db"]) * (rho * base + (1.0 - rho) * slot_component)

    @staticmethod
    def _deterministic_normal(key: str) -> float:
        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(key))
        rng = random.Random(seed)
        u1 = max(1e-12, rng.random())
        u2 = rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    @staticmethod
    def _marginalize_lut_3d(
        lut: dict[tuple[str, int, str, str, str, str], LUTEntry],
    ) -> dict[tuple[str, int, str], LUTEntry]:
        """Collapse the 6-D LUT to (qtype, service, link) by mean pooling."""
        pooled: dict[tuple[str, int, str], list[LUTEntry]] = {}
        for (qtype, level, link, _view, _fresh, _risk), entry in lut.items():
            pooled.setdefault((qtype, int(level), link), []).append(entry)
        return {
            key: LUTEntry(
                accuracy=sum(entry.accuracy for entry in entries) / len(entries),
                payload_bytes=sum(entry.payload_bytes for entry in entries) / len(entries),
            )
            for key, entries in pooled.items()
        }

    def _lookup_entry(self, task: EnvTask, service_level: int, snr_bin: str) -> LUTEntry:
        # W3b: 3-D query key -- view/freshness/risk dropped from the lookup.
        key = (task.task_type, int(service_level), snr_bin)
        if key in self.lut3:
            return self.lut3[key]
        legacy = channel_bin_from_snr(snr_db_from_label(snr_bin))
        legacy_key = (task.task_type, int(service_level), legacy)
        if legacy_key in self.lut3:
            return self.lut3[legacy_key]
        candidates = [
            entry
            for (qtype, level, _link), entry in self.lut3.items()
            if qtype == task.task_type and level == int(service_level)
        ]
        if candidates:
            return LUTEntry(
                accuracy=min(entry.accuracy for entry in candidates),
                payload_bytes=sum(entry.payload_bytes for entry in candidates) / len(candidates),
            )
        return LUTEntry(accuracy=0.0, payload_bytes={0: 0.0, 1: 2048.0, 2: 300000.0, 3: 80000.0}.get(int(service_level), 0.0))

    def _semantic_estimate(
        self,
        task: EnvTask,
        service_level: int,
        snr_bin: str,
        entry: LUTEntry,
    ) -> SemanticUtilityEstimate:
        if self.semantic_utility_model is not None:
            return self.semantic_utility_model.U_sem(
                task.task_type,
                int(service_level),
                snr_bin,
                task.view_quality_bin,
                task.freshness_bin,
                task.risk_level,
            )
        if int(service_level) == 0:
            candidates = [
                candidate
                for (qtype, level, _link, view, fresh, risk), candidate in self.lut.items()
                if qtype == task.task_type
                and level == 0
                and view == task.view_quality_bin
                and fresh == task.freshness_bin
                and risk == task.risk_level
            ]
            if candidates:
                accuracy = sum(candidate.accuracy for candidate in candidates) / len(candidates)
                payload_kb = sum(candidate.payload_bytes for candidate in candidates) / len(candidates) / 1024.0
                return SemanticUtilityEstimate(
                    accuracy_mean=accuracy,
                    accuracy_lcb=accuracy,
                    payload_kb=payload_kb,
                    uncertainty=0.0,
                    sample_count=len(candidates),
                )
        return SemanticUtilityEstimate(
            accuracy_mean=float(entry.accuracy),
            accuracy_lcb=float(entry.accuracy),
            payload_kb=float(entry.payload_bytes) / 1024.0,
            uncertainty=0.0,
            sample_count=1 if entry.accuracy > 0.0 or entry.payload_bytes > 0.0 else 0,
        )

    def _queue_delay_s(self, edge: "EdgeNode", tx_delay_s: float) -> float:
        """Shared-link + compute queue wait (task #23 E7v2 site iii).

        queue_model == "legacy" (default): the affine load model
            edge.load * queue_delay_scale_s + edge.gpu_load * gpu_queue_delay_scale_s
        -- bit-for-bit v1-v5.  This is the first-order (small-rho) approximation of
        the M/G/1 wait: W = rho*E[S^2]/(2E[S]) ~= (E[S^2]/(2E[S])) * rho, i.e.
        linear in the link utilisation rho == edge.load, with queue_delay_scale_s
        playing the role of the per-unit-load residual E[S^2]/(2E[S]).

        queue_model == "mg1": the SAME shared M/G/1 non-preemptive priority
        estimator used by the separation-capacity chain (bubbles_separation.
        mg1_priority_wait), with the link service time S = this delivery's tx
        airtime and rho = edge.load (C2/critical high-priority).  The GPU/compute
        queue keeps the affine gpu term (a distinct resource).  Opt-in so legacy
        runs stay identical.
        """
        gpu_q = float(edge.gpu_load) * float(self.env_cfg["gpu_queue_delay_scale_s"])
        model = str(self.env_cfg.get("queue_model", "legacy") or "legacy").lower()
        if model != "mg1":
            return float(edge.load) * float(self.env_cfg["queue_delay_scale_s"]) + gpu_q
        rho = max(0.0, min(0.999, float(edge.load)))
        es = max(1e-6, float(tx_delay_s))
        payload_rho = max(0.0, min(0.999, rho * float(self.env_cfg.get("queue_payload_load_ratio", 1.0))))
        classes = [
            {"lam": rho / es, "es": es, "es2": es * es},               # C2/critical hi-pri
            {"lam": payload_rho / es, "es": es, "es2": es * es},        # payload lo-pri
        ]
        w = bubbles_separation.mg1_priority_wait(classes, target_index=0)
        if not math.isfinite(w):
            # saturation guard: fall back to the affine model so the sim never
            # returns an infinite delay.
            return float(edge.load) * float(self.env_cfg["queue_delay_scale_s"]) + gpu_q
        return w + gpu_q

    def _delay_parts(
        self,
        task: EnvTask,
        uav: UAVNode,
        edge: EdgeNode,
        action: dict[str, Any],
        payload_bytes: float,
        link: dict[str, float],
    ) -> dict[str, float]:
        level = int(action["service_level"])
        mobility = self._mobility_plan(uav, task, action)
        fly_delay = float(mobility["arrival_delay_s"])
        sense_delay = float(self.env_cfg["sensing_delay_s_by_level"].get(str(level), 0.0))
        rate_bps = max(1.0, float(link["rate_mbps"]) * 1_000_000.0)
        tx_delay = 0.0 if level == 0 else 8.0 * max(0.0, payload_bytes) / rate_bps
        processing_base = float(self.env_cfg["processing_delay_s_by_level"].get(str(level), 0.3))
        cpu_work = float(self.env_cfg["cpu_workload_by_level"].get(str(level), 0.1))
        gpu_work = float(self.env_cfg["gpu_workload_by_level"].get(str(level), 0.1))
        cpu_delay = processing_base * cpu_work / max(1e-6, float(action["cpu_share"]))
        gpu_delay = processing_base * gpu_work / max(1e-6, float(action["gpu_share"]))
        infer_delay = max(cpu_delay, gpu_delay, 0.02 if level == 0 else 0.0)
        queue_delay = self._queue_delay_s(edge, tx_delay)
        model_cached = level == 0 or level in edge.cached_service_levels
        load_delay = float(self.env_cfg["model_cache_hit_delay_s"] if model_cached else self.env_cfg["model_load_delay_s"])
        total = fly_delay + sense_delay + tx_delay + queue_delay + infer_delay + load_delay
        return {
            "fly_delay_s": fly_delay,
            "arrival_delay_s": fly_delay,
            "fly_distance_m": float(mobility["fly_distance_m"]),
            "step_fly_distance_m": float(mobility["step_fly_distance_m"]),
            "waypoint_x": float(mobility["waypoint_x"]),
            "waypoint_y": float(mobility["waypoint_y"]),
            "target_altitude_m": float(mobility["altitude_m"]),
            "coverage_gain": float(mobility["coverage_gain"]),
            "utm_conflict_risk": float(mobility["utm_conflict_risk"]),
            "sense_delay_s": sense_delay,
            "tx_delay_s": tx_delay,
            "queue_delay_s": queue_delay,
            "infer_delay_s": infer_delay,
            "load_delay_s": load_delay,
            "total_delay_s": total,
        }

    def _energy_parts(self, delay: dict[str, float], action: dict[str, Any]) -> dict[str, float]:
        level = int(action["service_level"])
        fly_energy = float(delay.get("fly_distance_m", 0.0)) * float(self.env_cfg["flight_energy_j_per_m"])
        stay_hover_s = float(self.env_cfg["slot_s"]) if str(action.get("mobility_mode", "stay")) == "stay" else 0.0
        hover_energy = (delay["sense_delay_s"] + delay["infer_delay_s"] + stay_hover_s) * float(self.env_cfg["hover_power_w"])
        tx_energy = float(action["power"]) * delay["tx_delay_s"]
        compute_energy = float(self.env_cfg["compute_energy_j_by_level"].get(str(level), 0.0))
        total = fly_energy + hover_energy + tx_energy + compute_energy
        return {
            "fly_energy_j": fly_energy,
            "hover_energy_j": hover_energy,
            "mobility_energy_j": fly_energy + stay_hover_s * float(self.env_cfg["hover_power_w"]),
            "tx_energy_j": tx_energy,
            "compute_energy_j": compute_energy,
            "total_energy_j": total,
        }

    def _requires_operational_intent(self, action: dict[str, Any]) -> bool:
        if int(action["service_level"]) <= 0:
            return False
        return str(action.get("sensing_decision", "observe")) in {"observe", "revisit"} or str(
            action.get("mobility_mode", "serve_task")
        ) in {"serve_task", "reposition", "avoid_conflict", "return_base"}

    def _airspace_conflict(self, task: EnvTask, action: dict[str, Any]) -> bool:
        return bool(self._strategic_conflict_task_ids(task, action))

    def _strategic_conflict_task_ids(self, task: EnvTask, action: dict[str, Any]) -> list[str]:
        if not self._requires_operational_intent(action):
            return []
        if _bubbles_profile_active(self.env_cfg):
            return self._bubbles_tactical_conflict_task_ids(task, action)
        conflict_ids: list[str] = []
        for other_raw in action.get("concurrent_actions", []):
            other_task = self._task_by_id(other_raw.get("task_id"))
            if other_task is None or other_task.task_id == task.task_id:
                continue
            other = self.parse_action(other_raw, other_task)
            if self._requires_operational_intent(other) and self._area4d_overlaps_with_buffer(task.area4d, other_task.area4d):
                conflict_ids.append(other_task.task_id)
        utm = self.env_cfg.get("utm", {})
        if bool(utm.get("background_operational_intents", False)):
            for other_task in self._active_tasks():
                if other_task.task_id == task.task_id or other_task.completed or other_task.task_id in conflict_ids:
                    continue
                if self._background_intent_is_active(other_task) and self._area4d_overlaps_with_buffer(task.area4d, other_task.area4d):
                    conflict_ids.append(other_task.task_id)
        return conflict_ids

    def _background_intent_is_active(self, task: EnvTask) -> bool:
        utm = self.env_cfg.get("utm", {})
        density = max(0.0, min(1.0, float(utm.get("background_operational_intent_density", 1.0))))
        if density >= 1.0:
            return True
        key = f"{self.episode}:{self.step_count}:{task.operational_intent_id}:{task.task_id}:background_intent"
        score = 0.5 + 0.5 * self._deterministic_normal(key) / 4.0
        return max(0.0, min(1.0, score)) < density

    def _bubbles_separation_params(self) -> bubbles_separation.SeparationParams:
        confidence = str(self.env_cfg.get("bubbles_t4_confidence", "2sigma")).strip().lower()
        sigma = bubbles_separation.T4_CONFIDENCE_SIGMA.get(confidence, 2.0)
        return bubbles_separation.SeparationParams(t4_confidence_sigma=sigma)

    def _bubbles_aircraft_kinematics(
        self, target: EnvTask, uav: "UAVNode | None" = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Approximate an aircraft state (position, velocity) for a task volume.

        Position is the operational-volume centre at its mid-altitude; velocity
        is the SAIL III-IV cruise vector pointing from ``uav`` toward the volume
        centre (zero if the UAV is already on top of it). For the acting task the
        caller passes the UAV selected by the action, so the CPA conflict outcome
        is controllable through ``uav_assignment`` (P1-env fix: the nearest UAV
        was used unconditionally before, making conflicts largely exogenous to
        the policy). Background/other tasks carry no action and keep the
        nearest-UAV approximation.
        """
        cruise = float(self.env_cfg.get("uav_speed_mps", bubbles_separation.TABLE_B2["SAIL_III_IV"].cruise_mps))
        cx, cy = float(target.area4d.center_x_m), float(target.area4d.center_y_m)
        cz = 0.5 * (float(target.area4d.altitude_min_m) + float(target.area4d.altitude_max_m))
        uav = uav if uav is not None else self._nearest_uav(target)
        dx, dy = cx - uav.x_m, cy - uav.y_m
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            vel = (0.0, 0.0, 0.0)
        else:
            vel = (cruise * dx / norm, cruise * dy / norm, 0.0)
        return (cx, cy, cz), vel

    def _bubbles_tactical_conflict_task_ids(self, task: EnvTask, action: dict[str, Any]) -> list[str]:
        """CPA-based two-condition tactical-conflict detection (D2.1 p.38/p.61).

        Replaces the legacy Area4D-overlap distance test when the BUBBLES profile
        is active. A conflicting task is one whose predicted closest point of
        approach breaches the SAIL III-IV separation minimum ``d_TC`` (and ``h_TC``
        vertically) while the time-to-CPA is below ``TC_th``.
        """
        params = self._bubbles_separation_params()
        perf = bubbles_separation.TABLE_B2.get(
            str(self.env_cfg.get("bubbles_traffic_class", "SAIL_III_IV")),
            bubbles_separation.TABLE_B2["SAIL_III_IV"],
        )
        d_tc, h_tc = bubbles_separation.tactical_conflict_distance(perf, params)
        tc_threshold = float(self.env_cfg.get("bubbles_tc_threshold_s", 60.0))
        assignment = action.get("uav_assignment", None)
        assigned_uav = self.uavs[int(assignment) % len(self.uavs)] if assignment is not None and self.uavs else None
        p_self, v_self = self._bubbles_aircraft_kinematics(task, uav=assigned_uav)

        candidates: dict[str, EnvTask] = {}
        for other_raw in action.get("concurrent_actions", []):
            other_task = self._task_by_id(other_raw.get("task_id"))
            if other_task is None or other_task.task_id == task.task_id:
                continue
            other = self.parse_action(other_raw, other_task)
            if self._requires_operational_intent(other):
                candidates[other_task.task_id] = other_task
        utm = self.env_cfg.get("utm", {})
        if bool(utm.get("background_operational_intents", False)):
            for other_task in self._active_tasks():
                if other_task.task_id == task.task_id or other_task.completed:
                    continue
                if self._background_intent_is_active(other_task):
                    candidates.setdefault(other_task.task_id, other_task)

        conflict_ids: list[str] = []
        for other_id, other_task in candidates.items():
            p_other, v_other = self._bubbles_aircraft_kinematics(other_task)
            if bubbles_separation.is_tactical_conflict(
                p_self, v_self, p_other, v_other, d_tc, tc_threshold, h_tc_m=h_tc
            ):
                conflict_ids.append(other_id)
        return conflict_ids

    def _area4d_overlaps_with_buffer(self, first: Area4D, second: Area4D) -> bool:
        utm = self.env_cfg.get("utm", {})
        spatial_buffer = float(utm.get("spatial_buffer_m", 0.0)) if bool(utm.get("enabled", True)) else 0.0
        altitude_buffer = float(utm.get("altitude_buffer_m", 0.0)) if bool(utm.get("enabled", True)) else 0.0
        temporal_buffer = int(utm.get("temporal_buffer_steps", 0)) if bool(utm.get("enabled", True)) else 0
        spatial = first.distance_to(second.center_x_m, second.center_y_m) <= first.radius_m + second.radius_m + spatial_buffer
        altitude = (
            first.altitude_min_m - altitude_buffer <= second.altitude_max_m + altitude_buffer
            and second.altitude_min_m - altitude_buffer <= first.altitude_max_m + altitude_buffer
        )
        temporal = first.start_step <= second.end_step + temporal_buffer and second.start_step <= first.end_step + temporal_buffer
        return spatial and altitude and temporal

    def _utm_evaluation(self, task: EnvTask, action: dict[str, Any], conflict_task_ids: list[str]) -> dict[str, Any]:
        utm = self.env_cfg.get("utm", {})
        enabled = bool(utm.get("enabled", True))
        needs_intent = self._requires_operational_intent(action)
        dss_available = bool(utm.get("dss_available", True))
        mode = str(utm.get("mode", "nominal_planning"))
        dss_delay = float(utm.get("dss_delay_s", 0.0)) if enabled and needs_intent else 0.0
        if enabled and needs_intent and not dss_available:
            dss_delay = max(dss_delay, float(utm.get("contingent_delay_s", dss_delay)))
        notification_delay = (
            float(utm.get("subscription_notification_delay_s", 0.0))
            if enabled and needs_intent and bool(conflict_task_ids)
            else 0.0
        )
        if not enabled or not needs_intent:
            state = task.operational_intent_state if task.operational_intent_state in OPERATIONAL_INTENT_STATES else "accepted"
        elif not dss_available:
            state = "contingent"
        elif mode == "off_nominal_planning" and (
            task.operational_intent_state == "nonconforming" or task.risk_level == "critical"
        ):
            state = "nonconforming"
        elif conflict_task_ids:
            state = "nonconforming"
        else:
            state = "activated"
        utm_violation = enabled and needs_intent and (
            not dss_available or bool(conflict_task_ids) or state in {"nonconforming", "contingent"}
        )
        return {
            "operational_intent_state": state,
            "strategic_conflict_count": len(conflict_task_ids),
            "spatial_buffer_m": float(utm.get("spatial_buffer_m", 0.0)),
            "altitude_buffer_m": float(utm.get("altitude_buffer_m", 0.0)),
            "temporal_buffer_steps": int(utm.get("temporal_buffer_steps", 0)),
            "dss_available": dss_available,
            "dss_delay_s": dss_delay,
            "subscription_notification_delay_s": notification_delay,
            "conflict_notification_pending": notification_delay > 0.0,
            "utm_constraint_violation": utm_violation,
        }

    def _semantic_cache_hit_probability(self, task: EnvTask) -> float:
        base = float(self.env_cfg["cache_hit_probability"].get(task.freshness_bin, 0.0))
        reusable = any(
            entry.task_id != task.task_id
            and (entry.question_type or entry.task_type) == task.task_type
            and math.hypot(entry.x_m - task.x_m, entry.y_m - task.y_m) <= float(self.env_cfg["semantic_cache_radius_m"])
            for entry in self.semantic_cache_entries
        )
        boost = float(self.env_cfg["semantic_cache_reuse_boost"]) if reusable else 0.0
        if task.priority >= 2.0:
            boost *= 0.5
        return max(0.0, min(1.0, base + boost))

    def _gpu_memory_ok(self, edge: EdgeNode, service_level: int) -> bool:
        if service_level == 0 or service_level in edge.cached_service_levels:
            return True
        required = float(self.env_cfg["model_memory_mb_by_level"].get(str(service_level), 0.0))
        return edge.gpu_memory_used_mb + required <= edge.gpu_memory_capacity_mb + 1e-9

    def _mobility_plan(self, uav: UAVNode, task: EnvTask, action: dict[str, Any]) -> dict[str, float]:
        mode = str(action.get("mobility_mode", "serve_task"))
        target_x, target_y = uav.x_m, uav.y_m
        target_altitude = uav.altitude_m
        if mode == "serve_task":
            target_x, target_y = task.x_m, task.y_m
            target_altitude = max(task.area4d.altitude_min_m, min(task.area4d.altitude_max_m, uav.altitude_m))
        elif mode == "reposition":
            waypoint = action.get("waypoint")
            if isinstance(waypoint, (list, tuple)) and len(waypoint) >= 2:
                target_x, target_y = float(waypoint[0]), float(waypoint[1])
                if len(waypoint) >= 3:
                    target_altitude = float(waypoint[2])
            else:
                dx, dy = action.get("waypoint_delta", (0.0, 0.0))
                target_x, target_y = uav.x_m + float(dx), uav.y_m + float(dy)
            target_altitude = uav.altitude_m + float(action.get("altitude_delta", 0.0))
        elif mode == "avoid_conflict":
            conflict_center = self._conflict_centroid(task)
            if conflict_center is None:
                conflict_center = (task.x_m, task.y_m)
            away_x = uav.x_m - float(conflict_center[0])
            away_y = uav.y_m - float(conflict_center[1])
            norm = math.hypot(away_x, away_y)
            if norm <= 1e-9:
                away_x, away_y, norm = 1.0, 0.0, 1.0
            step = uav.speed_mps * float(self.env_cfg["slot_s"])
            target_x = uav.x_m + away_x / norm * step
            target_y = uav.y_m + away_y / norm * step
            target_altitude = uav.altitude_m + max(5.0, float(self.env_cfg.get("utm", {}).get("altitude_buffer_m", 0.0)))
        elif mode == "return_base":
            target_x, target_y = uav.base_x_m, uav.base_y_m
            target_altitude = uav.base_altitude_m
        target_altitude = max(5.0, min(200.0, target_altitude))
        distance_2d = math.hypot(target_x - uav.x_m, target_y - uav.y_m)
        altitude_distance = abs(target_altitude - uav.altitude_m)
        distance_3d = math.hypot(distance_2d, altitude_distance)
        step_distance = 0.0 if mode == "stay" else min(distance_3d, uav.speed_mps * float(self.env_cfg["slot_s"]))
        before = self._coverage_score(uav.x_m, uav.y_m)
        after = self._coverage_score(target_x, target_y)
        risk = self._utm_conflict_risk(task, action)
        if mode == "avoid_conflict":
            risk = max(0.0, risk - 0.35)
        return {
            "waypoint_x": target_x,
            "waypoint_y": target_y,
            "altitude_m": target_altitude,
            "fly_distance_m": 0.0 if mode == "stay" else distance_3d,
            "step_fly_distance_m": step_distance,
            "arrival_delay_s": 0.0 if mode == "stay" else distance_3d / max(1e-6, uav.speed_mps),
            "coverage_gain": after - before,
            "utm_conflict_risk": risk,
        }

    def _conflict_centroid(self, task: EnvTask) -> tuple[float, float] | None:
        candidates = [
            other
            for other in self._active_tasks()
            if other.task_id != task.task_id and self._area4d_overlaps_with_buffer(task.area4d, other.area4d)
        ]
        if not candidates:
            return None
        return (
            sum(candidate.x_m for candidate in candidates) / len(candidates),
            sum(candidate.y_m for candidate in candidates) / len(candidates),
        )

    def _coverage_score(self, x_m: float, y_m: float) -> float:
        tasks = [task for task in self.tasks if not task.completed and task.generation_time <= self.step_count + 2]
        if not tasks:
            return 0.0
        radius = max(1.0, float(self.env_cfg["area_spacing_m"]))
        scores = [max(0.0, 1.0 - math.hypot(task.x_m - x_m, task.y_m - y_m) / radius) * task.priority for task in tasks]
        return sum(scores) / max(1.0, sum(task.priority for task in tasks))

    def _utm_conflict_risk(self, task: EnvTask, action: dict[str, Any] | None = None) -> float:
        parsed = self.parse_action(action or {"service_level": 1}, task) if action is None or "bandwidth" not in action else action
        if not self._requires_operational_intent(parsed):
            return 0.0
        active = [other for other in self._active_tasks() if other.task_id != task.task_id and not other.completed]
        if not active:
            return 0.0
        overlaps = [
            other
            for other in active
            if self._background_intent_is_active(other) and self._area4d_overlaps_with_buffer(task.area4d, other.area4d)
        ]
        return max(0.0, min(1.0, len(overlaps) / max(1.0, len(active))))

    def _move_uav(self, uav: UAVNode, task: EnvTask, action: dict[str, Any]) -> None:
        mobility = self._mobility_plan(uav, task, action)
        target_x, target_y = float(mobility["waypoint_x"]), float(mobility["waypoint_y"])
        target_altitude = float(mobility["altitude_m"])
        distance = math.hypot(math.hypot(target_x - uav.x_m, target_y - uav.y_m), target_altitude - uav.altitude_m)
        step_distance = min(distance, float(mobility["step_fly_distance_m"]))
        uav.total_flight_m += step_distance
        if distance > 1e-9:
            ratio = step_distance / distance
            uav.x_m += (target_x - uav.x_m) * ratio
            uav.y_m += (target_y - uav.y_m) * ratio
            uav.altitude_m += (target_altitude - uav.altitude_m) * ratio

    def _update_edge(self, edge: EdgeNode, action: dict[str, Any], info: dict[str, Any]) -> None:
        edge.load = min(1.0, float(self.env_cfg["edge_load_decay"]) * edge.load + (1.0 - float(self.env_cfg["edge_load_decay"])) * float(action["cpu_share"]))
        edge.gpu_load = min(1.0, float(self.env_cfg["edge_load_decay"]) * edge.gpu_load + (1.0 - float(self.env_cfg["edge_load_decay"])) * float(action["gpu_share"]))
        level = int(action["service_level"])
        if bool(info["success"]) and level > 0:
            cached = [item for item in edge.cached_service_levels if item != level]
            cached.append(level)
            selected: list[int] = []
            used = edge.gpu_memory_capacity_mb * float(self.env_cfg["gpu_memory_load"])
            for cached_level in reversed(cached):
                memory = float(self.env_cfg["model_memory_mb_by_level"].get(str(cached_level), 0.0))
                if len(selected) < edge.model_cache_capacity and used + memory <= edge.gpu_memory_capacity_mb + 1e-9:
                    selected.append(cached_level)
                    used += memory
            edge.cached_service_levels = tuple(sorted(selected))
            edge.gpu_memory_used_mb = min(edge.gpu_memory_capacity_mb, used)

    def _update_semantic_cache(self, task: EnvTask, action: dict[str, Any], info: dict[str, Any]) -> None:
        aged = [replace(entry, cache_age=entry.cache_age + 1) for entry in self.semantic_cache_entries]
        new_entries: list[SemanticCacheEntry] = []
        if bool(info["success"]) and self._requires_operational_intent(action):
            new_entries.append(
                SemanticCacheEntry(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    risk_level=task.risk_level,
                    priority=task.priority,
                    x_m=task.x_m,
                    y_m=task.y_m,
                    cache_age=0,
                    updated_step=self.step_count,
                    area_id=task.area_id,
                    question_type=task.task_type,
                    quality_lcb=float(info.get("semantic_accuracy_lcb", 0.0)),
                    uncertainty=float(info.get("semantic_uncertainty", 0.0)),
                )
            )
        dedup: dict[tuple[str, str], SemanticCacheEntry] = {}
        for entry in [*aged, *new_entries]:
            key = (entry.task_id, entry.task_type)
            prev = dedup.get(key)
            if prev is None or entry.updated_step >= prev.updated_step:
                dedup[key] = entry
        entries = list(dedup.values())
        entries.sort(key=lambda item: (-item.priority, item.cache_age, -item.updated_step))
        self.semantic_cache_entries = entries[: int(self.env_cfg["semantic_cache_capacity"])]

    def _advance_cache_ages(self) -> None:
        for task in self.tasks:
            if task.completed:
                continue
            task.cache_age += 1
            task.freshness_bin = self._freshness_from_age(task.cache_age)

    def _expire_overdue_tasks(self) -> None:
        for task in self.tasks:
            if task.completed or task.expired:
                continue
            if self._remaining_deadline_s(task) <= 0.0:
                task.expired = True
                task.task_status = "expired"

    def _freshness_from_age(self, cache_age: int) -> str:
        thresholds = self.env_cfg["freshness_slots"]
        if cache_age <= int(thresholds["fresh"]):
            return "fresh"
        if cache_age <= int(thresholds["stale"]):
            return "stale"
        return "expired"

    def _reward(self, task: EnvTask, info: dict[str, Any]) -> float:
        return (
            float(self.env_cfg["reward_success"]) * task.priority * float(info["answer_accuracy_est"]) * float(bool(info["success"]))
            - float(self.env_cfg["reward_delay"]) * float(info["delay_s"])
            - float(self.env_cfg["reward_energy"]) * float(info["energy_j"])
            - float(self.env_cfg["reward_payload"]) * float(info["payload_kb"])
            - float(self.env_cfg["reward_violation"]) * (float(bool(info["quality_violation"])) + float(bool(info["deadline_violation"])))
            - float(self.env_cfg["reward_conflict"]) * float(bool(info["airspace_conflict"]))
        )

    def _reject_reward(self, task: EnvTask, info: dict[str, Any]) -> float:
        feasible = bool(info.get("reject_feasible", False))
        base = 0.25 if feasible else 2.0 * float(self.env_cfg["reward_violation"])
        return -base * float(task.priority)

    def _observation(self) -> dict[str, Any]:
        task = self._front_task()
        if task is None:
            obs = {
                "episode_step": self.step_count,
                "scenario": self.scenario_name,
                "formal_scenario": self.formal_scenario_name,
                "benchmark_split": self.benchmark_split,
                "scalability_profile": dict(self.scalability_profile),
                "network_layers": SEMANTIC_NETWORK_LAYERS,
                "task_type": "",
                "question_type": "",
                "risk_level": "",
                "view_quality_bin": "",
                "freshness_bin": "",
                "task_status": "",
                "remaining_deadline_s": 0.0,
                "defer_count": 0,
                "expired": False,
                "rejected": False,
                "reject_feasible": False,
                "reject_reason": "",
                "operational_intent_id": "",
                "operational_intent_state": "",
                "sensed_snr_db": 0.0,
                "snr_bin": "",
                "uav_state": [asdict(uav) for uav in self.uavs],
                "edge_load": [edge.load for edge in self.edges],
                "cache_state": self._cache_state(None),
                "candidate_path_metrics": {},
                "candidate_mobility_metrics": {},
                "task_queue": [],
                "pending_tasks": 0,
                "feasible_uavs": [],
                "uav_task_distances_m": {},
                "uav_battery_ratio": {},
                "predicted_fly_delay_s": {},
                "predicted_fly_energy_j": {},
                "task_area4d": {},
                "utm_conflict_risk": 0.0,
                "future_task_proximity": {},
                "coverage_score_by_uav": {},
                "feasible_mobility_mask": self._feasible_mobility_mask([]),
                "mobility_actor_state": {},
                "vector": [],
                "action_mask": self.action_mask(),
            }
            obs["graph"] = self.graph_observation()
            return obs
        nearest = self._nearest_uav(task)
        action = self.default_action(0)
        action["uav_assignment"] = nearest.uav_id
        link = self._link_budget(task, nearest, self.parse_action(action, task), interference_dbm=None)
        snr_bin = snr_db_to_bin_label(float(link["sinr_db"]), self.snr_bins_db)
        payloads = {}
        accuracies = {}
        delays = {}
        for level in self.service_levels():
            candidate = self.candidate_metrics(level, {"task_id": task.task_id, "snr_bin": snr_bin})
            payloads[level] = candidate["payload_kb"]
            accuracies[level] = candidate["accuracy"]
            delays[level] = candidate["delay_s"]
        # Change 5: ground-truth spec-attainability certificate, stored on the task
        # and exposed as a state feature (only when the escalation layer is armed).
        if self._escalation_enabled():
            task.spec_attainable = self._compute_spec_attainable(task, snr_bin)
        obs = {
            "episode_step": self.step_count,
            "scenario": self.scenario_name,
            "formal_scenario": self.formal_scenario_name,
            "benchmark_split": self.benchmark_split,
            "scalability_profile": dict(self.scalability_profile),
            "network_layers": SEMANTIC_NETWORK_LAYERS,
            "task_type": task.task_type,
            "question_type": task.task_type,
            "risk_level": task.risk_level,
            "object_count": int(task.object_count),
            "count_bucket": self._count_bucket_for_task(task),
            "spec_attainable": bool(task.spec_attainable),
            "view_quality_bin": task.view_quality_bin,
            "freshness_bin": task.freshness_bin,
            "task_status": task.task_status,
            "remaining_deadline_s": round(float(self._remaining_deadline_s(task)), 6),
            "defer_count": int(task.defer_count),
            "expired": bool(task.expired),
            "rejected": bool(task.rejected),
            "operational_intent_id": task.operational_intent_id,
            "operational_intent_state": task.operational_intent_state,
            "sensed_snr_db": round(float(link["snr_db"]), 6),
            "sinr_db": round(float(link["sinr_db"]), 6),
            "snr_bin": snr_bin,
            "deadline_s": task.tau_k,
            "epsilon_k": task.epsilon_k,
            "payload_kb_estimates_by_service": payloads,
            "accuracy_estimates_by_service": accuracies,
            "delay_estimates_by_service": delays,
            "task_id": task.task_id,
            "uav_state": [asdict(uav) for uav in self.uavs],
            "edge_load": [edge.load for edge in self.edges],
            "cache_state": self._cache_state(task),
            "candidate_path_metrics": self.candidate_path_metrics(task),
            "candidate_mobility_metrics": self.candidate_mobility_metrics(task),
            "task_queue": [self._task_obs(item) for item in self._active_tasks()],
            "pending_tasks": sum(1 for item in self.tasks if not item.completed and not item.expired and not item.rejected),
            "feasible_uavs": [uav.uav_id for uav in self.uavs if uav.battery_j > float(self.env_cfg["return_energy_reserve_j"])],
            "action_mask": self.action_mask(),
        }
        reject_metric = obs["candidate_path_metrics"].get("reject", {})
        obs["reject_feasible"] = bool(reject_metric.get("reject_feasible", reject_metric.get("joint_feasible", False)))
        obs["reject_reason"] = str(reject_metric.get("reject_reason", ""))
        obs.update(self._mobility_observation(task))
        obs["vector"] = self._observation_vector(obs)
        obs["graph"] = self.graph_observation()
        return obs

    def graph_observation(self) -> dict[str, Any]:
        active_tasks = self._active_tasks()
        if not active_tasks:
            front = self._front_task()
            active_tasks = [front] if front is not None else []
        uav_nodes = [
            {
                "id": f"uav:{uav.uav_id}",
                "uav_id": uav.uav_id,
                "features": [
                    round(uav.x_m, 6),
                    round(uav.y_m, 6),
                    round(uav.altitude_m, 6),
                    round(uav.battery_j / max(1.0, float(self.env_cfg["initial_battery_j"])), 6),
                    round(uav.speed_mps, 6),
                    round(uav.utilization, 6),
                ],
            }
            for uav in self.uavs
        ]
        task_nodes = [
            {
                "id": f"task:{task.task_id}",
                "task_id": task.task_id,
                "features": [
                    round(task.x_m, 6),
                    round(task.y_m, 6),
                    round(task.priority, 6),
                    round(task.tau_k, 6),
                    round(task.epsilon_k, 6),
                    round(float(task.cache_age), 6),
                    self._category_index(task.task_type, ["presence", "counting", "risk"]),
                    self._category_index(task.risk_level, ["normal", "critical"]),
                    self._category_index(task.view_quality_bin, ["poor", "medium", "good"]),
                    self._category_index(task.freshness_bin, ["fresh", "stale", "expired"]),
                ],
            }
            for task in active_tasks
        ]
        edge_nodes = [
            {
                "id": f"edge:{edge.edge_id}",
                "edge_id": edge.edge_id,
                "features": [
                    round(edge.load, 6),
                    round(edge.gpu_load, 6),
                    round(edge.gpu_memory_used_mb / max(1.0, edge.gpu_memory_capacity_mb), 6),
                    round(float(len(edge.cached_service_levels)) / max(1.0, float(edge.model_cache_capacity)), 6),
                ],
            }
            for edge in self.edges
        ]
        uav_task_edges: list[dict[str, Any]] = []
        for uav in self.uavs:
            for task in active_tasks:
                action = self.parse_action({"service_level": 1, "uav_assignment": uav.uav_id}, task)
                link = self._link_budget(task, uav, action, interference_dbm=None)
                uav_task_edges.append(
                    {
                        "source": f"uav:{uav.uav_id}",
                        "target": f"task:{task.task_id}",
                        "type": "uav_task_link",
                        "features": [
                            round(float(link["distance_3d_m"]), 6),
                            round(float(link["snr_db"]), 6),
                            round(float(link["sinr_db"]), 6),
                            round(float(link["rate_mbps"]), 6),
                            round(float(link["los_probability"]), 6),
                            round(float(link["path_loss_db"]), 6),
                        ],
                    }
                )
        task_edge_edges: list[dict[str, Any]] = []
        for task in active_tasks:
            for edge in self.edges:
                task_edge_edges.append(
                    {
                        "source": f"task:{task.task_id}",
                        "target": f"edge:{edge.edge_id}",
                        "type": "task_edge_compute",
                        "features": [
                            round(edge.load, 6),
                            round(edge.gpu_load, 6),
                            round(float(1 in edge.cached_service_levels), 6),
                            round(float(2 in edge.cached_service_levels), 6),
                            round(edge.gpu_memory_used_mb / max(1.0, edge.gpu_memory_capacity_mb), 6),
                        ],
                    }
                )
        return {
            "schema_version": "semantic_network_graph_v1",
            "feature_schema": self.graph_observation_schema(),
            "node_sets": {"uav": uav_nodes, "task": task_nodes, "edge": edge_nodes},
            "edge_sets": {"uav_task_link": uav_task_edges, "task_edge_compute": task_edge_edges},
        }

    @staticmethod
    def graph_observation_schema() -> dict[str, Any]:
        return {
            "node_sets": {
                "uav": ["x_m", "y_m", "altitude_m", "battery_ratio", "speed_mps", "utilization"],
                "task": [
                    "x_m",
                    "y_m",
                    "priority",
                    "deadline_s",
                    "epsilon",
                    "cache_age",
                    "task_type_id",
                    "risk_level_id",
                    "view_quality_id",
                    "freshness_id",
                ],
                "edge": ["cpu_load", "gpu_load", "gpu_memory_ratio", "model_cache_ratio"],
            },
            "edge_sets": {
                "uav_task_link": ["distance_3d_m", "snr_db", "sinr_db", "rate_mbps", "los_probability", "path_loss_db"],
                "task_edge_compute": ["cpu_load", "gpu_load", "semantic_token_model_cached", "image_model_cached", "gpu_memory_ratio"],
            },
        }

    @staticmethod
    def _category_index(value: str, choices: list[str]) -> float:
        try:
            return float(choices.index(value))
        except ValueError:
            return -1.0

    def _task_obs(self, task: EnvTask) -> dict[str, Any]:
        data = asdict(task)
        data["area4d"] = asdict(task.area4d)
        data["task_status"] = task.task_status
        data["remaining_deadline_s"] = round(float(self._remaining_deadline_s(task)), 6)
        data["defer_count"] = int(task.defer_count)
        data["expired"] = bool(task.expired)
        data["rejected"] = bool(task.rejected)
        return data

    def _cache_state(self, task: EnvTask | None) -> dict[str, Any]:
        status = self._cache_status(task) if task else {}
        return {
            "cached_completed_tasks": sum(1 for item in self.tasks if item.completed),
            "semantic_cache_entries": len(self.semantic_cache_entries),
            "front_task_cache_age": task.cache_age if task else 0,
            "front_task_freshness_bin": task.freshness_bin if task else "",
            "front_task_hit_probability": self._semantic_cache_hit_probability(task) if task else 0.0,
            "cache_exact_match": bool(status.get("cache_exact_match", False)),
            "cache_nearby_match": bool(status.get("cache_nearby_match", False)),
            "cache_eligible": bool(status.get("cache_eligible", False)),
            "cache_quality_lcb": float(status.get("cache_quality_lcb", 0.0)),
            "cache_age": int(status.get("cache_age", task.cache_age if task else 0)),
            "cache_freshness_bin": str(status.get("cache_freshness_bin", task.freshness_bin if task else "")),
            "cache_hit_probability": self._semantic_cache_hit_probability(task) if task else 0.0,
        }

    def _mobility_observation(self, task: EnvTask) -> dict[str, Any]:
        active = self._active_tasks()
        distances: dict[int, dict[str, float]] = {}
        predicted_delay: dict[int, dict[str, float]] = {}
        predicted_energy: dict[int, dict[str, float]] = {}
        battery_ratio: dict[int, float] = {}
        proximity: dict[int, float] = {}
        coverage: dict[int, float] = {}
        for uav in self.uavs:
            battery_ratio[uav.uav_id] = round(uav.battery_j / max(1.0, float(self.env_cfg["initial_battery_j"])), 6)
            distances[uav.uav_id] = {
                item.task_id: round(math.hypot(item.x_m - uav.x_m, item.y_m - uav.y_m), 6)
                for item in active
            }
            predicted_delay[uav.uav_id] = {
                item.task_id: round(math.hypot(item.x_m - uav.x_m, item.y_m - uav.y_m) / max(1e-6, uav.speed_mps), 6)
                for item in active
            }
            predicted_energy[uav.uav_id] = {
                item.task_id: round(math.hypot(item.x_m - uav.x_m, item.y_m - uav.y_m) * float(self.env_cfg["flight_energy_j_per_m"]), 6)
                for item in active
            }
            coverage[uav.uav_id] = round(self._coverage_score(uav.x_m, uav.y_m), 6)
            future_tasks = [item for item in self.tasks if not item.completed and item.generation_time > self.step_count]
            if future_tasks:
                nearest = min(math.hypot(item.x_m - uav.x_m, item.y_m - uav.y_m) for item in future_tasks)
                proximity[uav.uav_id] = round(max(0.0, 1.0 - nearest / max(1.0, float(self.env_cfg["area_spacing_m"]) * 2.0)), 6)
            else:
                proximity[uav.uav_id] = 0.0
        default_intent_action = self.parse_action({"service_level": 1, "mobility_mode": "serve_task"}, task)
        state = {
            "uav_task_distances_m": distances,
            "uav_battery_ratio": battery_ratio,
            "predicted_fly_delay_s": predicted_delay,
            "predicted_fly_energy_j": predicted_energy,
            "task_area4d": asdict(task.area4d),
            "utm_conflict_risk": round(self._utm_conflict_risk(task, default_intent_action), 6),
            "future_task_proximity": proximity,
            "coverage_score_by_uav": coverage,
            "feasible_mobility_mask": self._feasible_mobility_mask(active),
        }
        state["mobility_actor_state"] = dict(state)
        return state

    def _feasible_mobility_mask(self, active_tasks: list[EnvTask] | None = None) -> dict[str, bool]:
        active_tasks = self._active_tasks() if active_tasks is None else active_tasks
        has_active = bool(active_tasks)
        any_battery = any(uav.battery_j > float(self.env_cfg["return_energy_reserve_j"]) for uav in self.uavs)
        has_conflict_risk = any(
            any(self._area4d_overlaps_with_buffer(task.area4d, other.area4d) for other in active_tasks if other.task_id != task.task_id)
            for task in active_tasks
        )
        return {
            "stay": True,
            "serve_task": has_active and any_battery,
            "reposition": any_battery,
            "avoid_conflict": has_active and any_battery and has_conflict_risk,
            "return_base": any_battery,
        }

    def _observation_vector(self, obs: dict[str, Any]) -> list[float]:
        task_types = ("presence", "counting", "risk")
        risk_levels = ("normal", "critical")
        view_bins = ("poor", "medium", "good")
        freshness_bins = ("fresh", "stale", "expired")
        lo = min(self.snr_bins_db)
        hi = max(self.snr_bins_db)
        snr_scaled = 0.5 if math.isclose(lo, hi) else max(0.0, min(1.0, (float(obs["sensed_snr_db"]) - lo) / (hi - lo)))
        vec = [
            *[float(obs["task_type"] == item) for item in task_types],
            *[float(obs["risk_level"] == item) for item in risk_levels],
            *[float(obs["view_quality_bin"] == item) for item in view_bins],
            *[float(obs["freshness_bin"] == item) for item in freshness_bins],
            snr_scaled,
            min(1.0, float(obs["episode_step"]) / max(1.0, float(self.env_cfg["episode_steps"]))),
            min(1.0, float(obs["deadline_s"]) / 10.0),
            float(obs["epsilon_k"]),
            float(obs["cache_state"]["front_task_hit_probability"]),
            float(obs.get("utm_conflict_risk", 0.0)),
            min(1.0, float(obs["pending_tasks"]) / max(1.0, float(len(self.tasks)))),
        ]
        for level in self.service_levels():
            vec.append(min(1.0, float(obs["payload_kb_estimates_by_service"].get(level, 0.0)) / 300.0))
            vec.append(float(obs["accuracy_estimates_by_service"].get(level, 0.0)))
            vec.append(min(1.0, float(obs["delay_estimates_by_service"].get(level, 0.0)) / 10.0))
        for uav in self.uavs[:4]:
            vec.extend(
                [
                    uav.x_m / max(1.0, float(self.env_cfg["area_spacing_m"]) * 2.0),
                    uav.y_m / max(1.0, float(self.env_cfg["area_spacing_m"]) * 2.0),
                    uav.altitude_m / 200.0,
                    uav.battery_j / max(1.0, float(self.env_cfg["initial_battery_j"])),
                    min(1.0, uav.utilization),
                    float(obs.get("coverage_score_by_uav", {}).get(uav.uav_id, 0.0)),
                    float(obs.get("future_task_proximity", {}).get(uav.uav_id, 0.0)),
                ]
            )
        vec.extend([0.0] * max(0, (4 - min(len(self.uavs), 4)) * 7))
        for edge in self.edges[:2]:
            vec.extend(
                [
                    edge.load,
                    edge.gpu_load,
                    edge.gpu_memory_used_mb / max(1.0, edge.gpu_memory_capacity_mb),
                    float(len(edge.cached_service_levels)) / max(1.0, float(edge.model_cache_capacity)),
                ]
            )
        vec.extend([0.0] * max(0, (2 - min(len(self.edges), 2)) * 4))
        return [round(float(item), 6) for item in vec]

    @staticmethod
    def _empty_info() -> dict[str, Any]:
        return {
            "answer_accuracy_est": 0.0,
            "scenario": "",
            "formal_scenario": "",
            "benchmark_split": "",
            "task_status": "",
            "remaining_deadline_s": 0.0,
            "defer_count": 0,
            "expired": False,
            "rejected": False,
            "reject_feasible": False,
            "reject_reason": "",
            "expected_saved_energy_j": 0.0,
            "expected_saved_delay_s": 0.0,
            "avoided_utm_violation": False,
            "avoided_deadline_violation": False,
            "reject_penalty": 0.0,
            "deadline_s": 0.0,
            "epsilon_k": 0.0,
            "priority": 0.0,
            "semantic_accuracy_mean": 0.0,
            "semantic_accuracy_lcb": 0.0,
            "semantic_uncertainty": 1.0,
            "semantic_sample_count": 0,
            "semantic_payload_kb": 0.0,
            "semantic_quality_gap": 0.0,
            "semantic_success": False,
            "semantic_path": "",
            "delay_s": 0.0,
            "energy_j": 0.0,
            "payload_kb": 0.0,
            "quality_violation": False,
            "deadline_violation": False,
            "airspace_conflict": False,
            "utm_constraint_violation": False,
            "utm_conflict_violation": False,
            "risk_violation": False,
            "operational_intent_id": "",
            "operational_intent_state": "",
            "airspace_state": "",
            "cache_exact_match": False,
            "cache_nearby_match": False,
            "cache_eligible": False,
            "cache_quality_lcb": 0.0,
            "cache_uncertainty": 1.0,
            "cache_age": 0,
            "cache_freshness_bin": "",
            "cache_hit_probability": 0.0,
            "operational_priority": 0.0,
            "strategic_conflict": False,
            "strategic_conflict_count": 0,
            "strategic_conflict_task_ids": "",
            "dss_available": True,
            "dss_delay_s": 0.0,
            "utm_delay_s": 0.0,
            "subscription_notification_delay_s": 0.0,
            "conflict_notification_pending": False,
            "snr_bin": "",
            "service_level": 0,
            "mobility_mode": "stay",
            "waypoint_x": 0.0,
            "waypoint_y": 0.0,
            "altitude_m": 0.0,
            "fly_distance_m": 0.0,
            "coverage_gain": 0.0,
            "mobility_energy_j": 0.0,
            "arrival_delay_s": 0.0,
            "utm_conflict_risk": 0.0,
            "semantic_service_name": "",
            "semantic_evidence_type": "",
            "semantic_utility": 0.0,
            "semantic_efficiency": 0.0,
            "success": False,
        }


def load_multi_uav_env(config_path: str | Path, seed: int | None = None, scenario: str | None = None) -> MultiUAVVQAEnv:
    cfg = load_config(config_path)
    if scenario is not None:
        env_cfg = dict(cfg.get("multi_uav_env", {}))
        env_cfg["scenario"] = _normalize_scenario_name(scenario)
        cfg["multi_uav_env"] = env_cfg
    tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut = load_lut(resolve_path(cfg["paths"]["vlm_lut_csv"]))
    return MultiUAVVQAEnv(tasks, lut, cfg, seed=seed, semantic_utility_model=_load_semantic_utility_model(cfg))


def _load_semantic_utility_model(cfg: dict[str, Any]) -> SemanticUtilityModel | None:
    utility_path = resolve_path(
        cfg.get("paths", {}).get("semantic_utility_csv", "outputs/lut/v1_9_semantic_utility_with_ci.csv")
    )
    if not utility_path.exists():
        return None
    return SemanticUtilityModel.from_csv(utility_path)


def formal_scenario_specs_markdown() -> str:
    lines = [
        "# Formal Semantic Network Scenario Specs",
        "",
        "This file is generated from `src/vqa_semcom/sim/multi_uav_env.py` and is owned by the environment thread.",
        "",
        "## Network Architecture Layers",
        "",
    ]
    for layer, items in SEMANTIC_NETWORK_LAYERS.items():
        lines.append(f"- `{layer}`: {', '.join(items)}")
    lines.extend(["", "## Semantic Service Levels", ""])
    for level, meta in SERVICE_LEVELS.items():
        enabled = "reserved/disabled by default" if level == 3 else "enabled when present in LUT/config"
        lines.append(f"- `s={level}` `{meta['name']}`: {meta['evidence_type']} ({enabled})")
    lines.extend(["", "## Formal Train/Test Scenarios", ""])
    lines.append("| name | split | base scenario | description |")
    lines.append("|---|---|---|---|")
    for name, spec in FORMAL_SCENARIO_PRESETS.items():
        base = spec.get("base_scenario", "")
        if spec.get("scenario_mixture"):
            base = "+".join(str(item) for item in spec["scenario_mixture"])
        lines.append(f"| `{name}` | {spec.get('split', '')} | {base} | {spec.get('description', '')} |")
    lines.extend(["", "## Scalability Presets", ""])
    for family, presets in SCALABILITY_PRESETS.items():
        lines.append(f"### {family}")
        lines.append("")
        for name, values in presets.items():
            lines.append(f"- `{name}`: `{values}`")
        lines.append("")
    lines.extend(
        [
            "## Graph Observation Schema",
            "",
            "- node sets: `uav`, `task`, `edge`",
            "- edge sets: `uav_task_link`, `task_edge_compute`",
            "- schema is available at runtime via `env.graph_observation_schema()` and observations include `obs['graph']`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_formal_scenario_specs(path: Path) -> None:
    from vqa_semcom.config import ensure_parent

    ensure_parent(path)
    path.write_text(formal_scenario_specs_markdown() + "\n", encoding="utf-8")


def write_env_trace(rows: list[dict[str, Any]], csv_path: Path, summary_path: Path) -> None:
    from vqa_semcom.config import ensure_parent

    ensure_parent(csv_path)
    ensure_parent(summary_path)
    fieldnames = [
        "step",
        "scenario",
        "formal_scenario",
        "benchmark_split",
        "task_id",
        "uav_id",
        "edge_id",
        "service_level",
        "mobility_mode",
        "waypoint_x",
        "waypoint_y",
        "altitude_m",
        "fly_distance_m",
        "coverage_gain",
        "mobility_energy_j",
        "arrival_delay_s",
        "utm_conflict_risk",
        "semantic_service_name",
        "semantic_evidence_type",
        "semantic_utility",
        "semantic_efficiency",
        "answer_accuracy_est",
        "semantic_accuracy_mean",
        "semantic_accuracy_lcb",
        "semantic_uncertainty",
        "semantic_sample_count",
        "semantic_payload_kb",
        "semantic_quality_gap",
        "semantic_success",
        "deadline_s",
        "epsilon_k",
        "priority",
        "delay_s",
        "energy_j",
        "payload_kb",
        "quality_violation",
        "deadline_violation",
        "risk_violation",
        "airspace_conflict",
        "utm_constraint_violation",
        "utm_conflict_violation",
        "operational_intent_id",
        "operational_intent_state",
        "airspace_state",
        "operational_priority",
        "strategic_conflict",
        "strategic_conflict_count",
        "strategic_conflict_task_ids",
        "utm_spatial_buffer_m",
        "utm_altitude_buffer_m",
        "utm_temporal_buffer_steps",
        "dss_available",
        "dss_delay_s",
        "utm_delay_s",
        "subscription_notification_delay_s",
        "conflict_notification_pending",
        "snr_bin",
        "sensed_snr_db",
        "sinr_db",
        "rate_mbps",
        "distance_3d_m",
        "elevation_deg",
        "los_probability",
        "path_loss_db",
        "interference_dbm",
        "fading_db",
        "cache_hit_probability",
        "semantic_cache_hit",
        "gpu_memory_ok",
        "gpu_memory_used_mb",
        "gpu_memory_capacity_mb",
        "battery_remaining_j",
        "fly_delay_s",
        "step_fly_distance_m",
        "sense_delay_s",
        "tx_delay_s",
        "queue_delay_s",
        "infer_delay_s",
        "load_delay_s",
        "utm_dss_delay_s",
        "utm_notification_delay_s",
        "fly_energy_j",
        "hover_energy_j",
        "tx_energy_j",
        "compute_energy_j",
        "success",
        "reward",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    denom = max(1, len(rows))
    success = sum(float(row.get("success", False)) for row in rows) / denom
    avg_delay = sum(float(row.get("delay_s", 0.0)) for row in rows) / denom
    avg_energy = sum(float(row.get("energy_j", 0.0)) for row in rows) / denom
    avg_payload = sum(float(row.get("payload_kb", 0.0)) for row in rows) / denom
    avg_sinr = sum(float(row.get("sinr_db", 0.0)) for row in rows) / denom
    avg_rate = sum(float(row.get("rate_mbps", 0.0)) for row in rows) / denom
    avg_semantic_utility = sum(float(row.get("semantic_utility", 0.0)) for row in rows) / denom
    conflicts = sum(float(row.get("airspace_conflict", False)) for row in rows) / denom
    utm_violations = sum(float(row.get("utm_constraint_violation", False)) for row in rows) / denom
    dss_outage_rate = sum(1.0 - float(bool(row.get("dss_available", True))) for row in rows) / denom
    lines = [
        "# Multi-UAV Environment Smoke Summary",
        "",
        f"- steps: {len(rows)}",
        f"- success_rate: {success:.3f}",
        f"- average_delay_s: {avg_delay:.3f}",
        f"- average_energy_j: {avg_energy:.3f}",
        f"- average_payload_kb: {avg_payload:.3f}",
        f"- average_sinr_db: {avg_sinr:.3f}",
        f"- average_rate_mbps: {avg_rate:.3f}",
        f"- average_semantic_utility: {avg_semantic_utility:.3f}",
        f"- airspace_conflict_rate: {conflicts:.3f}",
        f"- utm_constraint_violation_rate: {utm_violations:.3f}",
        f"- dss_outage_rate: {dss_outage_rate:.3f}",
        "",
        "Outputs are environment-thread artifacts and intentionally live under `outputs/env`.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
