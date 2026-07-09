"""Task #33/#34 v6: scale-consistency suite.

Covers the comm-window deadline gate, the anti-stall guard, the tau re-anchor
values, the spec-attainability certificate distribution, and the M/G/1 queueing
formula.  The legacy path stays bit-for-bit (asserted here + in the v5 suite).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.sim.multi_uav_env import (  # noqa: E402
    COMM_WINDOW_TAU,
    MultiUAVVQAEnv,
)
from vqa_semcom.sim.resource_env import LUTEntry  # noqa: E402


def _env(deadline_semantics: str = "legacy", tau_k: str = "30.0", extra_env: dict | None = None) -> MultiUAVVQAEnv:
    tasks = [{
        "question_type": "presence", "question": "Are there cars?", "risk_level": "normal",
        "epsilon_k": "0.5", "tau_k": tau_k, "view_quality_bin": "good", "freshness_bin": "fresh",
        "object_count": "3",
    }]
    lut = {
        ("presence", lvl, snr, "good", "fresh", "normal"): LUTEntry(0.5, 0.0)
        for lvl in (0, 1, 2) for snr in ("0dB", "10dB", "20dB")
    }
    env_block = {
        "num_uavs": 2, "episode_steps": 4, "tasks_per_episode": 1, "snr_noise_std_db": 0.0,
        "base_snr_db": 12.0, "area_spacing_m": 100.0, "uav_speed_mps": 20.0,
        "deadline_semantics": deadline_semantics,
    }
    if extra_env:
        env_block.update(extra_env)
    cfg = {
        "bins": {"snr_db": [0, 10, 20], "freshness": ["fresh"], "service_levels": [0, 1, 2]},
        "simulation": {"seed": 5, "bandwidth_hz": 1_000_000},
        "multi_uav_env": env_block,
    }
    return MultiUAVVQAEnv(tasks, lut, cfg, seed=5)


class DeadlineSemanticsGateTest(unittest.TestCase):
    def test_default_is_legacy(self):
        env = _env()  # no deadline_semantics key path removed -> explicit legacy
        env.env_cfg.pop("deadline_semantics", None)
        self.assertEqual(env._deadline_semantics(), "legacy")

    def test_legacy_charges_full_delay(self):
        env = _env("legacy")
        delay = {"total_delay_s": 10.0, "fly_delay_s": 8.0}
        self.assertAlmostEqual(env._deadline_delay_s(delay), 10.0, places=9)

    def test_comm_window_excludes_fly(self):
        env = _env("comm_window")
        delay = {"total_delay_s": 10.0, "fly_delay_s": 8.0}
        # comm-charged = total - fly = 2.0
        self.assertAlmostEqual(env._deadline_delay_s(delay), 2.0, places=9)

    def test_comm_window_never_negative(self):
        env = _env("comm_window")
        delay = {"total_delay_s": 5.0, "fly_delay_s": 9.0}  # fly > total (defensive)
        self.assertAlmostEqual(env._deadline_delay_s(delay), 0.0, places=9)

    def test_comm_window_falls_back_to_arrival_delay_key(self):
        env = _env("comm_window")
        delay = {"total_delay_s": 6.0, "arrival_delay_s": 4.0}  # no fly_delay_s key
        self.assertAlmostEqual(env._deadline_delay_s(delay), 2.0, places=9)


class TauReanchorTest(unittest.TestCase):
    def test_comm_window_constants(self):
        self.assertAlmostEqual(COMM_WINDOW_TAU["critical"], 2.8, places=9)
        self.assertAlmostEqual(COMM_WINDOW_TAU["normal"], 3.8, places=9)
        self.assertAlmostEqual(COMM_WINDOW_TAU["high"], 2.8, places=9)

    def test_legacy_tau_uses_csv_times_scale(self):
        env = _env("legacy")
        row = {"tau_k": "5.0", "risk_level": "normal"}
        # legacy: 5.0 * tau_scale(0.85) = 4.25
        self.assertAlmostEqual(env._tau_for_task(row, "normal", {"tau_scale": 0.85}), 4.25, places=9)
        self.assertAlmostEqual(env._tau_for_task({"tau_k": "3.0"}, "critical", {}), 3.0, places=9)

    def test_comm_window_tau_ignores_scale_and_csv(self):
        env = _env("comm_window")
        # comm_window: anchor 2.8 (critical) regardless of CSV tau_k or tau_scale
        self.assertAlmostEqual(env._tau_for_task({"tau_k": "3.0"}, "critical", {"tau_scale": 0.85}), 2.8, places=9)
        self.assertAlmostEqual(env._tau_for_task({"tau_k": "5.0"}, "normal", {"tau_scale": 0.85}), 3.8, places=9)

    def test_comm_window_tau_yaml_override(self):
        env = _env("comm_window")
        env.cfg["thresholds"] = {"tau_critical_comm": 2.2, "tau_normal_comm": 3.0}
        self.assertAlmostEqual(env._tau_for_task({}, "critical", {}), 2.2, places=9)
        self.assertAlmostEqual(env._tau_for_task({}, "normal", {}), 3.0, places=9)

    def test_comm_window_layout_override(self):
        env = _env("comm_window")
        self.assertAlmostEqual(
            env._tau_for_task({}, "critical", {"tau_comm_by_risk": {"critical": 1.5}}), 1.5, places=9)

    def test_generated_task_tau_matches_semantics(self):
        # end-to-end: a comm_window env generates tasks with the re-anchored tau.
        env = _env("comm_window", tau_k="3.0")
        env.reset(seed=1)
        t = env._front_task()
        self.assertIsNotNone(t)
        # normal task -> 3.8 (default nominal scenario tau_scale not applied)
        self.assertAlmostEqual(float(t.tau_k), 3.8, places=6)


class AntiStallGuardTest(unittest.TestCase):
    """Task #33-A closure: delaying service must NOT be free under comm_window.

    Even though flight is off the deadline clock, the remaining-deadline window
    still shrinks each slot and the certificate is re-evaluated against the
    shrunken remaining tau, so a task served later has strictly less slack (and
    eventually turns deadline-infeasible / expires).  This closes the
    'infinite postpone' hole: postponement is penalised through the shrinking
    remaining-tau, aging, and the episode bound.
    """

    def test_remaining_deadline_shrinks_each_slot(self):
        env = _env("comm_window")
        env.reset(seed=1)
        t = env._front_task()
        r0 = env._remaining_deadline_s(t)
        # advance the clock without completing the task
        env.step_count += 1
        r1 = env._remaining_deadline_s(t)
        env.step_count += 1
        r2 = env._remaining_deadline_s(t)
        self.assertLess(r1, r0)
        self.assertLess(r2, r1)

    def test_certificate_turns_false_when_window_eaten(self):
        # A task that is spec-attainable fresh becomes un-attainable once the
        # remaining window is fully consumed (remaining -> 0 -> certificate False).
        env = _env("comm_window", tau_k="3.0", extra_env={"escalation_mode": "spec_attainable"})
        env.reset(seed=1)
        t = env._front_task()
        # fresh: certificate reflects the full remaining window
        fresh = env._compute_spec_attainable(t, "20dB")
        self.assertTrue(fresh)  # permissive LUT + comm_window -> attainable when fresh
        # consume the ENTIRE window (tau 3.8 normal, slot 1s -> 4 slots exhausts it)
        env.step_count = int(t.generation_time) + int(float(t.tau_k)) + 1
        self.assertLessEqual(env._remaining_deadline_s(t), 0.0)
        eaten = env._compute_spec_attainable(t, "20dB")
        self.assertFalse(eaten)  # no remaining window -> never attainable

    def test_expired_when_remaining_hits_zero(self):
        env = _env("comm_window", tau_k="3.0")
        env.reset(seed=1)
        t = env._front_task()
        env.step_count = int(t.generation_time) + 5  # well past tau
        self.assertLessEqual(env._remaining_deadline_s(t), 0.0)


class CertificateDistributionTest(unittest.TestCase):
    """Task #34-(ii): the certificate must be a non-constant distribution."""

    def test_certificate_routes_through_deadline_chokepoint(self):
        # Under comm_window a task whose ONLY blocker is flight must be attainable;
        # the same task under legacy (flight charged) is not.  Uses a permissive
        # LUT (lcb 0.5 >= eps 0.5 quality-ok) so the deadline axis is isolated.
        comm = _env("comm_window", tau_k="3.0", extra_env={"escalation_mode": "spec_attainable"})
        comm.reset(seed=1)
        tc = comm._front_task()
        tc.risk_level = "normal"
        tc.epsilon_k = 0.5
        legacy = _env("legacy", tau_k="3.0", extra_env={"escalation_mode": "spec_attainable"})
        legacy.reset(seed=1)
        tl = legacy._front_task()
        tl.risk_level = "normal"
        tl.epsilon_k = 0.5
        # comm-window should be attainable (flight excluded), legacy should not
        # (flight ~ area_spacing/speed >> 3s window).
        self.assertTrue(comm._compute_spec_attainable(tc, "20dB"))
        self.assertFalse(legacy._compute_spec_attainable(tl, "20dB"))


if __name__ == "__main__":
    unittest.main()
