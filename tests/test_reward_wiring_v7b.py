"""Task #36 v7b: mission-aligned reward WIRING into the CONTROLLER path.

The env-layer multi_uav_env._reward() already implemented mission_aligned success
attribution (task #36), but the two-timescale / monolithic PPO learner does not
optimise that reward: it optimises the CONTROLLER reward
(_semantic_controller_reward) minus the dual penalty.  Under semantic_reward_mode
!= "env" the controller RECOMPUTES the reward from info and discards raw_reward,
so the env-layer mission_aligned signal never reached the gradient -- the v7 bug
(proposed/no_lagrangian behaved bit-for-bit like the legacy v6 run while only the
env-side raw_return log moved).

This suite covers the v7b wiring:
  * the compliant-but-blocked token now out-scores the banned cache in the
    CONTROLLER reward under mission_aligned (the ordering flip the learner needs);
  * the shaped reward (controller - dual conflict penalty) also flips;
  * the ENV layer and the CONTROLLER layer agree on the rescue direction
    (two-layer semantic consistency);
  * legacy is bit-for-bit unchanged (the gate is off by default);
  * non-compliant blocked services and unblocked services are unaffected.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.rl.v19_ppo import (  # noqa: E402
    DualState,
    PPOTrainConfig,
    _mission_aligned_reward_info,
    _mission_aligned_success_credit,
    _semantic_controller_reward,
)
from vqa_semcom.sim.multi_uav_env import MultiUAVVQAEnv  # noqa: E402
from vqa_semcom.sim.resource_env import LUTEntry  # noqa: E402


# --------------------------------------------------------------------------- #
# Controller-layer fixtures (the reward the LEARNER optimises).
# --------------------------------------------------------------------------- #
def _obs(**ov):
    o = dict(
        risk_level="critical", epsilon_k=0.464, epsilon=0.464,
        deadline_s=5.0, tau_k=5.0, defer_count=0, freshness_bin="fresh",
        operational_intent_state="accepted",
    )
    o.update(ov)
    return o


def _ctrl_info(**ov):
    """A quality+deadline+battery+resource COMPLIANT service that is UTM/airspace
    BLOCKED (success=False only on the airspace axis) -- the v6/v7 peak
    diagnostic token that was flooding cache.  Served on the token path (s1)."""
    info = dict(
        risk_level="critical", semantic_path="token", service_level=1,
        answer_accuracy_est=0.86, semantic_accuracy_mean=0.86, semantic_accuracy_lcb=0.80,
        semantic_uncertainty=0.05, semantic_success=True, success=False,
        service_compliant=True,
        quality_violation=False, deadline_violation=False,
        battery_violation=False, resource_violation=False,
        airspace_conflict=True, utm_constraint_violation=True, utm_conflict_violation=True,
        delay_s=2.16, energy_j=15.0, payload_kb=80.0, energy_budget_j=120.0,
        reject_feasible=False, spec_attainable=True,
    )
    info.update(ov)
    return info


def _cfg(mode="semantic_utility", constrained=True, rss="legacy", disc=0.8):
    return PPOTrainConfig(
        semantic_reward_mode=mode, constrained=constrained,
        reward_success_semantics=rss, reward_blocked_service_discount=disc,
    )


def _attractive_cache_info(**ov):
    """The s0 cache shortcut the v6/v7 policy FLOODED: a cache-served token that
    clears the quality LCB (accuracy >= epsilon, quality_violation=False) and is
    fast/cheap and unblocked, so under LEGACY it out-scores the penalised
    compliant-but-blocked delivery.  This is the inversion the v7b wiring must
    flip (mission_aligned lifts the blocked token above this cache).  Accuracy
    0.50 sits just above the critical epsilon 0.464."""
    info = dict(
        risk_level="critical", semantic_path="cache", service_level=0,
        answer_accuracy_est=0.50, semantic_accuracy_mean=0.50, semantic_accuracy_lcb=0.45,
        semantic_uncertainty=0.05, semantic_success=True, success=True,
        service_compliant=True,
        quality_violation=False, deadline_violation=False,
        battery_violation=False, resource_violation=False,
        airspace_conflict=False, utm_constraint_violation=False, utm_conflict_violation=False,
        delay_s=0.24, energy_j=2.0, payload_kb=8.0, energy_budget_j=120.0,
        reject_feasible=False, spec_attainable=True, freshness_bin="fresh",
    )
    info.update(ov)
    return info


class MissionAlignedCreditTest(unittest.TestCase):
    def test_default_is_legacy_no_credit(self):
        self.assertEqual(_mission_aligned_success_credit(_ctrl_info(), _cfg(rss="legacy")), 0.0)

    def test_mission_aligned_credits_compliant_blocked(self):
        self.assertAlmostEqual(
            _mission_aligned_success_credit(_ctrl_info(), _cfg(rss="mission_aligned", disc=0.8)), 0.8, places=9)

    def test_noncompliant_blocked_not_credited(self):
        nc = _ctrl_info(quality_violation=True, semantic_success=False, service_compliant=False)
        self.assertEqual(_mission_aligned_success_credit(nc, _cfg(rss="mission_aligned")), 0.0)

    def test_unblocked_success_credit_is_strict(self):
        ok = _ctrl_info(success=True, airspace_conflict=False, utm_constraint_violation=False, utm_conflict_violation=False)
        self.assertEqual(_mission_aligned_success_credit(ok, _cfg(rss="mission_aligned")), 1.0)

    def test_view_clears_block_only_on_rescue(self):
        view, credit = _mission_aligned_reward_info(_ctrl_info(), _cfg(rss="mission_aligned"))
        self.assertFalse(view["airspace_conflict"])
        self.assertFalse(view["utm_constraint_violation"])
        self.assertAlmostEqual(credit, 0.8, places=9)
        # legacy leaves the block intact
        view_l, credit_l = _mission_aligned_reward_info(_ctrl_info(), _cfg(rss="legacy"))
        self.assertTrue(view_l["airspace_conflict"])
        self.assertEqual(credit_l, 0.0)

    def test_derives_compliance_when_no_flag(self):
        """When info has no explicit service_compliant, the credit derives it from
        the RL-authoritative violation fields (quality/deadline/battery/resource)."""
        info = _ctrl_info()
        info.pop("service_compliant")
        self.assertAlmostEqual(_mission_aligned_success_credit(info, _cfg(rss="mission_aligned")), 0.8, places=9)
        info2 = _ctrl_info(deadline_violation=True)
        info2.pop("service_compliant")
        self.assertEqual(_mission_aligned_success_credit(info2, _cfg(rss="mission_aligned")), 0.0)


class ControllerRewardWiringTest(unittest.TestCase):
    def test_compliant_blocked_beats_attractive_cache_mission_aligned(self):
        """The v7b fix, in the LEARNER's reward: under mission_aligned the
        compliant-but-blocked token STRICTLY out-scores the attractive s0 cache
        shortcut that the policy was flooding."""
        cfg = _cfg(rss="mission_aligned")
        r_blocked = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, cfg)
        r_cache = _semantic_controller_reward(_obs(), _attractive_cache_info(), 0.0, cfg)
        self.assertGreater(r_blocked, r_cache)

    def test_legacy_compliant_blocked_loses_to_attractive_cache(self):
        """Regression witness: under legacy the ordering is INVERTED -- the
        compliant-but-blocked token scores BELOW the attractive cache shortcut
        (the v6/v7 defect that floods cache and pins lambda_quality_critical)."""
        cfg = _cfg(rss="legacy")
        r_blocked = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, cfg)
        r_cache = _semantic_controller_reward(_obs(), _attractive_cache_info(), 0.0, cfg)
        self.assertLess(r_blocked, r_cache)

    def test_mission_aligned_improves_blocked_vs_cache_gap(self):
        """Robust monotone witness (fixture-independent direction): mission_aligned
        moves the blocked-token-minus-cache gap strictly upward."""
        r_blocked_leg = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, _cfg(rss="legacy"))
        r_blocked_ma = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, _cfg(rss="mission_aligned"))
        r_cache = _semantic_controller_reward(_obs(), _attractive_cache_info(), 0.0, _cfg(rss="legacy"))
        r_cache_ma = _semantic_controller_reward(_obs(), _attractive_cache_info(), 0.0, _cfg(rss="mission_aligned"))
        # the attractive (unblocked, compliant) cache is unchanged by the semantics
        self.assertAlmostEqual(r_cache, r_cache_ma, places=9)
        self.assertGreater(r_blocked_ma - r_cache_ma, r_blocked_leg - r_cache)

    def test_mission_aligned_lifts_compliant_blocked_reward(self):
        r_leg = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, _cfg(rss="legacy"))
        r_ma = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, _cfg(rss="mission_aligned"))
        self.assertGreater(r_ma, r_leg)

    def test_shaped_reward_drops_conflict_dual_on_rescue(self):
        """The dual conflict penalty (shaped_reward = controller - dual.penalty)
        must also drop for a compliant-blocked rescue -- feed dual the mission
        view."""
        dual = DualState()
        dual.conflict = 8.0  # a live conflict price
        info = _ctrl_info()
        # legacy view keeps the block -> dual charges the conflict lambda.
        view_leg, _ = _mission_aligned_reward_info(info, _cfg(rss="legacy"))
        pen_leg = dual.penalty(view_leg)
        # mission view clears the block -> dual conflict term drops.
        view_ma, _ = _mission_aligned_reward_info(info, _cfg(rss="mission_aligned"))
        pen_ma = dual.penalty(view_ma)
        self.assertGreater(pen_leg, pen_ma)
        self.assertAlmostEqual(pen_ma, 0.0, places=9)

    def test_noncompliant_blocked_not_rescued_in_controller(self):
        nc = _ctrl_info(quality_violation=True, semantic_success=False, service_compliant=False)
        r_leg = _semantic_controller_reward(_obs(), nc, 0.0, _cfg(rss="legacy"))
        r_ma = _semantic_controller_reward(_obs(), nc, 0.0, _cfg(rss="mission_aligned"))
        self.assertAlmostEqual(r_leg, r_ma, places=9)

    def test_unblocked_service_unchanged_by_semantics(self):
        ok = _ctrl_info(success=True, airspace_conflict=False, utm_constraint_violation=False, utm_conflict_violation=False)
        r_leg = _semantic_controller_reward(_obs(), dict(ok), 0.0, _cfg(rss="legacy"))
        r_ma = _semantic_controller_reward(_obs(), dict(ok), 0.0, _cfg(rss="mission_aligned"))
        self.assertAlmostEqual(r_leg, r_ma, places=9)

    def test_env_mode_passthrough_unaffected(self):
        """semantic_reward_mode='env' returns raw_reward verbatim under both
        semantics (the env reward already carries #36; no double application)."""
        for rss in ("legacy", "mission_aligned"):
            cfg = _cfg(mode="env", rss=rss)
            self.assertAlmostEqual(_semantic_controller_reward(_obs(), _ctrl_info(), -3.14, cfg), -3.14, places=9)

    def test_legacy_bit_for_bit_over_random_infos(self):
        """Legacy controller reward is byte-identical whether or not the
        reward_success_semantics knob exists (gate fully off).  Compares the
        default-legacy config against an explicit legacy config over a random
        battery across every semantic mode and constrained flag."""
        rng = random.Random(20260709)
        paths = ["cache", "token", "image", "defer", "cache_update", "reject"]
        risks = ["critical", "high", "normal"]
        modes = ["semantic_utility", "no_semantic_utility", "accuracy_only", "uncertainty_aware"]
        for _ in range(1500):
            info = dict(
                risk_level=rng.choice(risks), semantic_path=rng.choice(paths),
                service_level=rng.randint(0, 3),
                answer_accuracy_est=rng.uniform(0.3, 0.99),
                semantic_accuracy_mean=rng.uniform(0.3, 0.99),
                semantic_accuracy_lcb=rng.uniform(0.2, 0.95),
                semantic_uncertainty=rng.uniform(0, 0.5),
                semantic_success=rng.random() < 0.6, success=rng.random() < 0.4,
                service_compliant=rng.random() < 0.5,
                quality_violation=rng.random() < 0.4, deadline_violation=rng.random() < 0.3,
                battery_violation=rng.random() < 0.1, resource_violation=rng.random() < 0.1,
                airspace_conflict=rng.random() < 0.4, utm_constraint_violation=rng.random() < 0.4,
                utm_conflict_violation=rng.random() < 0.4,
                delay_s=rng.uniform(0, 6), energy_j=rng.uniform(0, 40), payload_kb=rng.uniform(0, 200),
                energy_budget_j=120.0, defer_count=rng.randint(0, 3),
                freshness_bin=rng.choice(["fresh", "stale", "expired"]),
                operational_intent_state=rng.choice(["accepted", "nonconforming", "contingent"]),
                reject_feasible=rng.random() < 0.5, spec_attainable=rng.random() < 0.5,
            )
            obs = _obs(risk_level=info["risk_level"], epsilon_k=rng.uniform(0.3, 0.7),
                       deadline_s=rng.uniform(1, 6), tau_k=rng.uniform(1, 6))
            mode = rng.choice(modes)
            raw = rng.uniform(-5, 5)
            for con in (True, False):
                r_default = _semantic_controller_reward(dict(obs), dict(info), raw,
                                                        PPOTrainConfig(semantic_reward_mode=mode, constrained=con))
                r_legacy = _semantic_controller_reward(dict(obs), dict(info), raw,
                                                       _cfg(mode=mode, constrained=con, rss="legacy"))
                self.assertAlmostEqual(r_default, r_legacy, places=12)


# --------------------------------------------------------------------------- #
# Two-layer semantic consistency: the ENV reward (_reward) and the CONTROLLER
# reward must agree on the mission_aligned rescue DIRECTION for the same event.
# --------------------------------------------------------------------------- #
def _env_obj(extra_env: dict | None = None) -> MultiUAVVQAEnv:
    tasks = [{
        "question_type": "presence", "question": "Are there cars?", "risk_level": "critical",
        "epsilon_k": "0.464", "tau_k": "30.0", "view_quality_bin": "good", "freshness_bin": "fresh",
        "object_count": "3",
    }]
    lut = {
        ("presence", lvl, snr, "good", "fresh", "normal"): LUTEntry(0.5, 0.0)
        for lvl in (0, 1, 2) for snr in ("0dB", "10dB", "20dB")
    }
    env_block = {
        "num_uavs": 2, "episode_steps": 4, "tasks_per_episode": 1, "snr_noise_std_db": 0.0,
        "base_snr_db": 12.0, "area_spacing_m": 100.0, "uav_speed_mps": 20.0,
        "deadline_semantics": "comm_window",
    }
    if extra_env:
        env_block.update(extra_env)
    cfg = {
        "bins": {"snr_db": [0, 10, 20], "freshness": ["fresh"], "service_levels": [0, 1, 2]},
        "simulation": {"seed": 5, "bandwidth_hz": 1_000_000},
        "multi_uav_env": env_block,
    }
    return MultiUAVVQAEnv(tasks, lut, cfg, seed=5)


class _Task:
    def __init__(self, priority: float):
        self.priority = float(priority)


def _env_blocked_info(**ov):
    info = {
        "success": False, "answer_accuracy_est": 0.86, "delay_s": 11.27,
        "deadline_charged_delay_s": 2.16, "energy_j": 15.0, "payload_kb": 80.0,
        "quality_violation": False, "deadline_violation": False,
        "service_compliant": True, "airspace_conflict": True, "utm_constraint_violation": True,
    }
    info.update(ov)
    return info


class TwoLayerConsistencyTest(unittest.TestCase):
    def test_both_layers_lift_compliant_blocked_under_mission_aligned(self):
        """Two-layer semantic consistency: switching legacy -> mission_aligned
        RAISES the reward of the SAME compliant-blocked event at BOTH the env
        layer (_reward) and the controller layer (_semantic_controller_reward)."""
        # env layer
        env_leg = _env_obj()
        env_ma = _env_obj({"reward_success_semantics": "mission_aligned"})
        env_r_leg = env_leg._reward(_Task(1.0), _env_blocked_info())
        env_r_ma = env_ma._reward(_Task(1.0), _env_blocked_info())
        self.assertGreater(env_r_ma, env_r_leg)
        # controller layer
        ctrl_r_leg = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, _cfg(rss="legacy"))
        ctrl_r_ma = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, _cfg(rss="mission_aligned"))
        self.assertGreater(ctrl_r_ma, ctrl_r_leg)

    def test_both_layers_flip_blocked_vs_shortcut_ordering(self):
        """Both layers agree that mission_aligned FLIPS the compliant-blocked vs
        cheap-shortcut ordering (legacy: blocked < shortcut; mission: blocked >
        shortcut).  The env-layer shortcut is the banned (reject) cache; the
        controller-layer shortcut is the attractive quality-passing s0 cache."""
        # env layer: blocked token vs banned (reject) cache
        env_leg = _env_obj()
        env_ma = _env_obj({"reward_success_semantics": "mission_aligned"})
        e_blk_leg = env_leg._reward(_Task(1.0), _env_blocked_info())
        e_ban_leg = env_leg._reject_reward(_Task(0.54), {"reject_feasible": False})
        e_blk_ma = env_ma._reward(_Task(1.0), _env_blocked_info())
        e_ban_ma = env_ma._reject_reward(_Task(0.54), {"reject_feasible": False})
        self.assertLess(e_blk_leg, e_ban_leg)     # legacy inversion
        self.assertGreater(e_blk_ma, e_ban_ma)    # mission flip
        # controller layer: blocked token vs attractive s0 cache
        c_blk_leg = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, _cfg(rss="legacy"))
        c_shc_leg = _semantic_controller_reward(_obs(), _attractive_cache_info(), 0.0, _cfg(rss="legacy"))
        c_blk_ma = _semantic_controller_reward(_obs(), _ctrl_info(), 0.0, _cfg(rss="mission_aligned"))
        c_shc_ma = _semantic_controller_reward(_obs(), _attractive_cache_info(), 0.0, _cfg(rss="mission_aligned"))
        self.assertLess(c_blk_leg, c_shc_leg)     # legacy inversion
        self.assertGreater(c_blk_ma, c_shc_ma)    # mission flip


if __name__ == "__main__":
    unittest.main()
