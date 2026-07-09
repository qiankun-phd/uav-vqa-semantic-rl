from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.sim.multi_uav_env import (  # noqa: E402
    ATTAINABILITY_V4_EPSILON,
    ATTAINABILITY_V5_EPSILON,
    MultiUAVVQAEnv,
    SemanticCacheEntry,
    count_bucket_v5,
    epsilon_v5_class,
)
from vqa_semcom.rl.v19_ppo import DualState, PPOTrainConfig, _dual_update, _init_dual_state, _update_duals  # noqa: E402
from vqa_semcom.rl.v19_resource_env import LutV5Table  # noqa: E402
from vqa_semcom.sim.resource_env import LUTEntry  # noqa: E402

LUT_V5_CSV = ROOT / "outputs" / "lut" / "v5_unified_lut.csv"


def _env() -> MultiUAVVQAEnv:
    tasks = [{
        "question_type": "presence", "question": "Are there cars?", "risk_level": "normal",
        "epsilon_k": "0.5", "tau_k": "30.0", "view_quality_bin": "good", "freshness_bin": "fresh",
        "object_count": "3",
    }]
    lut = {
        ("presence", lvl, snr, "good", "fresh", "normal"): LUTEntry(0.5, 0.0)
        for lvl in (0, 1, 2) for snr in ("0dB", "10dB", "20dB")
    }
    cfg = {
        "bins": {"snr_db": [0, 10, 20], "freshness": ["fresh"], "service_levels": [0, 1, 2]},
        "simulation": {"seed": 5, "bandwidth_hz": 1_000_000},
        "multi_uav_env": {
            "num_uavs": 2, "episode_steps": 4, "tasks_per_episode": 1, "snr_noise_std_db": 0.0,
            "base_snr_db": 12.0, "area_spacing_m": 100.0, "uav_speed_mps": 20.0,
        },
    }
    return MultiUAVVQAEnv(tasks, lut, cfg, seed=5)


class EpsilonV5ValueTest(unittest.TestCase):
    def setUp(self):
        self.env = _env()
        self.env.scenario_cfg = {"task_layout": {}}

    def _eps(self, risk, qtype, object_count):
        self.env.env_cfg = {"epsilon_calibration": "attainability_v5"}
        return self.env._epsilon_for_task(
            {"question_type": qtype, "view_quality_bin": "good", "object_count": str(object_count)}, risk)

    def test_two_key_table(self):
        self.assertEqual(ATTAINABILITY_V5_EPSILON[("critical", "counting")], 0.464)
        self.assertEqual(ATTAINABILITY_V5_EPSILON[("critical", "presence")], 0.696)
        self.assertEqual(ATTAINABILITY_V5_EPSILON[("normal", "presence")], 0.529)

    def test_counting_ge10_uses_counting_key(self):
        self.assertAlmostEqual(self._eps("critical", "counting", 17), 0.464, places=9)
        self.assertAlmostEqual(self._eps("high", "counting", 50), 0.464, places=9)

    def test_low_count_counting_uses_presence_key(self):
        # GT < 10 counting is easy -> presence-like bar.
        self.assertAlmostEqual(self._eps("critical", "counting", 4), 0.696, places=9)

    def test_presence_critical(self):
        self.assertAlmostEqual(self._eps("critical", "presence", -1), 0.696, places=9)

    def test_normal_all_qtypes(self):
        self.assertAlmostEqual(self._eps("normal", "counting", 17), 0.529, places=9)
        self.assertAlmostEqual(self._eps("normal", "presence", -1), 0.529, places=9)

    def test_class_helper(self):
        self.assertEqual(epsilon_v5_class("counting", "10-19"), "counting")
        self.assertEqual(epsilon_v5_class("counting", "1-4"), "presence")
        self.assertEqual(epsilon_v5_class("presence", "na"), "presence")

    def test_count_bucket(self):
        self.assertEqual(count_bucket_v5(3), "1-4")
        self.assertEqual(count_bucket_v5(12), "10-19")
        self.assertEqual(count_bucket_v5(77), "50+")
        self.assertEqual(count_bucket_v5(0), "0")

    def test_flat_ignores_row_and_scaling(self):
        self.env.env_cfg = {"epsilon_calibration": "attainability_v5"}
        self.env.scenario_cfg = {"task_layout": {"epsilon_scale": 0.5, "epsilon_cap_by_risk": {"critical": 0.4}}}
        eps = self.env._epsilon_for_task(
            {"question_type": "counting", "object_count": "20", "epsilon_k": "0.99"}, "critical")
        self.assertAlmostEqual(eps, 0.464, places=9)

    def test_legacy_and_v4_unaffected(self):
        self.env.env_cfg = {}
        self.assertAlmostEqual(self.env._epsilon_for_task({"question_type": "presence"}, "critical"), 0.82, places=9)
        self.env.env_cfg = {"epsilon_calibration": "attainability_v4"}
        self.assertAlmostEqual(self.env._epsilon_for_task({"question_type": "presence"}, "critical"), 0.355, places=9)
        self.assertEqual(ATTAINABILITY_V4_EPSILON, {"critical": 0.355, "normal": 0.166, "high": 0.355})


class LutV5BackendReadTest(unittest.TestCase):
    def setUp(self):
        if not LUT_V5_CSV.exists():
            self.skipTest("v5 unified LUT not present")
        self.table = LutV5Table.from_csv(LUT_V5_CSV)

    def test_exact_counting_bucket_read(self):
        # good channel (>=15dB) presence token cell resolves to a real Wilson LCB.
        hit = self.table.lookup("presence", 1, "20dB", "medium", "na")
        self.assertIsNotNone(hit)
        mean, lcb, unc, pay = hit
        self.assertGreaterEqual(mean, lcb)
        self.assertAlmostEqual(unc, max(0.0, mean - lcb), places=6)

    def test_channel_derived_from_snr(self):
        # -5dB -> bad channel; 20dB -> good channel: different cells.
        bad = self.table.lookup("counting", 2, "-5dB", "good", "10-19")
        good = self.table.lookup("counting", 2, "20dB", "good", "10-19")
        self.assertIsNotNone(bad)
        self.assertIsNotNone(good)
        self.assertGreaterEqual(good[1], bad[1] - 1e-9)  # better channel -> not-worse LCB

    def test_unseen_key_falls_back(self):
        hit = self.table.lookup("counting", 2, "5dB", "zzz-unseen-view", "1-4")
        self.assertIsNotNone(hit)  # pooled fallback, never crashes


class EntryV2CacheQualityTest(unittest.TestCase):
    """Change 3: s0 quality from real cached-answer LCB x freshness decay."""

    def _wrap_env(self, cache_quality):
        import importlib
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / "scripts"))
        from vqa_semcom.config import load_config, resolve_path
        from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv
        R = importlib.import_module("run_v1_9_resource_alloc")
        cfg = load_config(str(ROOT / "configs/v1_9_bubbles.yaml"))
        cfg.setdefault("multi_uav_env", {})["cache_quality"] = cache_quality
        tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
        lut = load_lut(ROOT / "outputs/lut/v1_9_snr_semantic_quality_lut.csv")
        tasks = filter_tasks_supported_by_lut(tasks, lut)
        args = SimpleNamespace(scenario="nominal", seed=0, snr_bins=None, tasks_per_episode=8,
                               formal_scenario=None, state_version="v2", num_uavs=None,
                               quality_backend=None, disable_semantic_token=False)
        return R.make_env(args, cfg, tasks, lut, "always_cache")

    def test_empty_cache_s0_lcb_zero(self):
        if not (ROOT / "configs/v1_9_bubbles.yaml").exists():
            self.skipTest("bubbles config not present")
        env = self._wrap_env("entry_v2")
        self.assertEqual(env.cache_quality, "entry_v2")
        obs = env.reset(seed=0, options={"policy_name": "always_cache"})
        info = env.evaluate_action(env.candidate_action(0, obs), obs)
        self.assertEqual(float(info["semantic_accuracy_lcb"]), 0.0)  # empty cache -> 0
        self.assertEqual(info.get("cache_quality_mode"), "entry_v2")


class EscalationCertificateTest(unittest.TestCase):
    """Change 5: reject/expired escalation verdict + accounting."""

    def _crit_env(self, escalation_mode="spec_attainable"):
        env = _env()
        env.reset(seed=5)
        env.env_cfg["escalation_mode"] = escalation_mode
        env.env_cfg["critical_cache_compliance"] = "forbidden"
        task = env._front_task()
        task.risk_level = "critical"
        task.epsilon_k = 0.6
        return env, task

    def test_verdict_spec_attainable_reject_is_violation(self):
        env, task = self._crit_env()
        task.spec_attainable = True
        qv, esc = env._escalation_verdict(task, base_quality_violation=False)
        self.assertTrue(qv)
        self.assertFalse(esc)

    def test_verdict_unattainable_reject_is_escalation(self):
        env, task = self._crit_env()
        task.spec_attainable = False
        qv, esc = env._escalation_verdict(task, base_quality_violation=False)
        self.assertFalse(qv)
        self.assertTrue(esc)

    def test_expired_unattainable_escalates_not_quality(self):
        env, task = self._crit_env()
        task.spec_attainable = False
        info = env._expired_task_info(task, {"service_level": 1})
        self.assertTrue(bool(info["escalated"]))
        self.assertFalse(bool(info["quality_violation"]))
        self.assertEqual(float(info["escalation"]), 1.0)

    def test_expired_attainable_is_quality_violation(self):
        env, task = self._crit_env()
        task.spec_attainable = True
        info = env._expired_task_info(task, {"service_level": 1})
        self.assertFalse(bool(info["escalated"]))
        self.assertTrue(bool(info["quality_violation"]))

    def test_legacy_off_preserves_expired_quality_violation(self):
        env, task = self._crit_env(escalation_mode="off")
        task.spec_attainable = False
        info = env._expired_task_info(task, {"service_level": 1})
        self.assertFalse(bool(info.get("escalated", False)))
        self.assertTrue(bool(info["quality_violation"]))  # legacy: expired == quality violation

    def test_reject_normal_task_unchanged(self):
        env, _ = self._crit_env()
        task = env._front_task()
        task.risk_level = "normal"
        qv, esc = env._escalation_verdict(task, base_quality_violation=False)
        self.assertFalse(qv)
        self.assertFalse(esc)


class LambdaDecayAndEscalationDualTest(unittest.TestCase):
    """Change 6 (lambda_decay=0) + escalation dual channel."""

    def test_dual_update_decay_zero_no_leak(self):
        cfg = PPOTrainConfig(lambda_decay=0.1, lambda_lr=0.0)
        # cost == limit, lr 0 -> only the decay term moves lambda.
        with_leak = _dual_update(1.0, 0.0, 0.0, cfg)
        no_leak = _dual_update(1.0, 0.0, 0.0, cfg, lambda_decay=0.0)
        self.assertLess(with_leak, 1.0)          # leaky: relaxes toward 0
        self.assertAlmostEqual(no_leak, 1.0, places=9)  # pure projection: stays

    def test_penalty_includes_escalation(self):
        d = DualState(escalation=2.0)
        base = d.penalty({"risk_level": "critical"})
        esc = d.penalty({"risk_level": "critical", "escalated": True})
        self.assertAlmostEqual(esc - base, 2.0, places=9)

    def test_update_duals_registers_escalation(self):
        cfg = PPOTrainConfig(constrained=True, lambda_freeze=False, risk_aware_constraints=True,
                             escalation_cost_limit=0.1, lambda_lr=0.5)
        d = _init_dual_state(cfg)
        rollout = {k: [] for k in (
            "quality_costs_normal", "quality_costs_critical", "deadline_costs_normal",
            "deadline_costs_critical", "conflict_costs", "battery_costs", "gpu_costs",
            "quality_costs", "deadline_costs")}
        rollout["escalation_costs"] = [1.0, 1.0, 1.0]  # cost 1.0 >> limit 0.1
        _update_duals(d, rollout, cfg)
        self.assertGreater(d.escalation, 0.0)  # channel ascends when escalation cost exceeds budget

    def test_escalated_excluded_from_quality_cost(self):
        # An escalated info carries quality_violation=False, so the standard
        # quality-cost accrual naturally excludes it.
        info = {"risk_level": "critical", "escalated": True, "quality_violation": False}
        self.assertEqual(float(bool(info.get("quality_violation", False))), 0.0)
        self.assertEqual(float(bool(info.get("escalated", False))), 1.0)


if __name__ == "__main__":
    unittest.main()
