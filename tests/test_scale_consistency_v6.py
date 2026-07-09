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


class EscalationAwareOracleTest(unittest.TestCase):
    """Task #33/#34: the escalation-aware oracle rejects spec-unattainable
    critical/high tasks (routing them to escalation) instead of serving-and-failing."""

    def _wrap(self):
        import importlib
        import sys as _sys
        from types import SimpleNamespace
        _sys.path.insert(0, str(ROOT / "scripts"))
        from vqa_semcom.config import load_config, resolve_path
        from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv
        R = importlib.import_module("run_v1_9_resource_alloc")
        cfg = load_config(str(ROOT / "configs/v1_9_bubbles.yaml"))
        eo = dict(cfg.get("multi_uav_env", {}))
        eo.update({"escalation_mode": "spec_attainable", "critical_cache_compliance": "forbidden",
                   "deadline_semantics": "comm_window", "epsilon_calibration": "attainability_v5"})
        cfg["multi_uav_env"] = eo
        tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
        lut = load_lut(ROOT / "outputs/lut/v1_9_snr_semantic_quality_lut.csv")
        tasks = filter_tasks_supported_by_lut(tasks, lut)
        args = SimpleNamespace(scenario="utm_conflict", seed=0, snr_bins=None, tasks_per_episode=10,
                               formal_scenario=None, state_version="v2", num_uavs=None,
                               quality_backend="lut_v5", disable_semantic_token=False)
        return R, R.make_env(args, cfg, tasks, lut, "oracle_escalation_aware")

    def test_registered_policy(self):
        import importlib
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        R = importlib.import_module("run_v1_9_resource_alloc")
        self.assertIn("oracle_escalation_aware", R.BASELINE_POLICIES)

    def test_rejects_unattainable_critical(self):
        if not (ROOT / "configs/v1_9_bubbles.yaml").exists():
            self.skipTest("bubbles config not present")
        R, w = self._wrap()
        obs = w.reset(seed=0, options={"policy_name": "oracle_escalation_aware"})
        # force a spec-UNattainable critical front task in the obs
        obs = dict(obs)
        obs["risk_level"] = "critical"
        obs["spec_attainable"] = False
        act = R.choose_baseline_action("oracle_escalation_aware", w, obs)
        self.assertEqual(str(act.get("semantic_path")), "reject")


class MissionSuccessMetricTest(unittest.TestCase):
    """Task #34-(iv): mission_success = quality AND deadline compliant."""

    def _rec(self, **kw):
        from types import SimpleNamespace
        base = {"quality_violation": False, "deadline_violation": False,
                "escalated": False, "risk_level": "critical"}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_mission_needs_both_axes(self):
        import importlib
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        R = importlib.import_module("run_v1_9_resource_alloc")
        self.assertTrue(R._is_mission_success(self._rec()))
        self.assertFalse(R._is_mission_success(self._rec(quality_violation=True)))
        self.assertFalse(R._is_mission_success(self._rec(deadline_violation=True)))
        self.assertFalse(R._is_mission_success(self._rec(quality_violation=True, deadline_violation=True)))

    def test_admitted_excludes_escalated(self):
        import importlib
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        R = importlib.import_module("run_v1_9_resource_alloc")
        recs = [
            self._rec(),                                   # admitted mission ok
            self._rec(deadline_violation=True),            # admitted mission fail
            self._rec(escalated=True, quality_violation=True),  # escalated -> excluded
        ]
        # admitted set = 2 records, 1 mission-ok -> 0.5
        self.assertAlmostEqual(R._admitted_mission_success_rate(recs), 0.5, places=9)
        # overall mission rate over all 3 = 1/3
        self.assertAlmostEqual(R._mission_success_rate(recs), 1.0 / 3.0, places=9)


class CacheBanEngagementTest(unittest.TestCase):
    """Task #34-(iii): the critical cache-compliance ban must actually bite a
    spec-attainable critical task served cache-only under the escalation layer."""

    def _cache_env(self):
        env = _env("comm_window", tau_k="30.0", extra_env={
            "escalation_mode": "spec_attainable",
            "critical_cache_compliance": "forbidden",
            "cache_quality": "legacy",
        })
        env.reset(seed=3)
        # Seed an ELIGIBLE, high-quality cache entry co-located with the front
        # task so cache_eligible=True and the cache LCB clears epsilon -- then the
        # ONLY thing that can make the cache-only path non-compliant is the ban.
        from vqa_semcom.sim.multi_uav_env import SemanticCacheEntry
        task = env._front_task()
        task.epsilon_k = 0.5
        env.semantic_cache_entries = [SemanticCacheEntry(
            task_id=task.task_id, task_type=task.task_type, risk_level=task.risk_level,
            priority=1.0, x_m=task.x_m, y_m=task.y_m, cache_age=0, updated_step=0,
            area_id=task.area_id, question_type=task.task_type, quality_lcb=0.99, uncertainty=0.01,
        )]
        return env, task

    def test_cache_only_critical_spec_attainable_is_noncompliant(self):
        env, task = self._cache_env()
        task.risk_level = "critical"
        # sanity: with the ban OFF the eligible cache WOULD be compliant
        cs = env._cache_status(task)
        self.assertTrue(cs["cache_eligible"])
        self.assertGreaterEqual(cs["cache_quality_lcb"], task.epsilon_k)
        task.spec_attainable = True  # transmission WAS possible -> ban bites
        info = env.evaluate_action(env.candidate_action(0, {"task_id": task.task_id}), task_id=task.task_id, mutate=False)
        self.assertEqual(str(info.get("semantic_path")), "cache")
        # eligible cache clears eps, yet the ban forces non-compliance:
        self.assertTrue(bool(info.get("quality_violation")))

    def test_cache_ban_relaxed_when_not_spec_attainable(self):
        env, task = self._cache_env()
        task.risk_level = "critical"
        task.spec_attainable = False  # no feasible tx -> not gaming -> ban relaxed
        info = env.evaluate_action(env.candidate_action(0, {"task_id": task.task_id}), task_id=task.task_id, mutate=False)
        self.assertEqual(str(info.get("semantic_path")), "cache")
        # eligible cache clears eps AND the ban is relaxed -> compliant.
        self.assertFalse(bool(info.get("quality_violation")))


class MG1QueueingTest(unittest.TestCase):
    """Task #23 E7v2: shared-link M/G/1 non-preemptive priority wait (hand-checked)."""

    def test_pk_formula(self):
        from vqa_semcom.sim.bubbles_separation import mg1_pk_wait
        # M/M/1: rho=0.5, E[S]=1, E[S^2]=2 -> W = 0.5*2/(2*1) = 0.5
        self.assertAlmostEqual(mg1_pk_wait(0.5, 1.0, 2.0), 0.5, places=9)
        # M/D/1: rho=0.5, E[S]=1, E[S^2]=1 -> W = 0.25
        self.assertAlmostEqual(mg1_pk_wait(0.5, 1.0, 1.0), 0.25, places=9)
        self.assertEqual(mg1_pk_wait(0.0, 1.0, 1.0), 0.0)     # idle
        self.assertEqual(mg1_pk_wait(0.5, 0.0, 1.0), 0.0)     # degenerate service
        self.assertEqual(mg1_pk_wait(1.2, 1.0, 1.0), float("inf"))  # saturated

    def test_priority_two_class_hand(self):
        from vqa_semcom.sim.bubbles_separation import mg1_priority_wait
        cls = [{"lam": 0.3, "es": 1.0, "es2": 1.0}, {"lam": 0.4, "es": 1.0, "es2": 1.0}]
        # R = (0.3+0.4)/2 = 0.35; hi: 0.35/((1)(0.7)) = 0.5; lo: 0.35/((0.7)(0.3)) = 1.6667
        self.assertAlmostEqual(mg1_priority_wait(cls, 0), 0.5, places=9)
        self.assertAlmostEqual(mg1_priority_wait(cls, 1), 5.0 / 3.0, places=9)

    def test_priority_saturation(self):
        from vqa_semcom.sim.bubbles_separation import mg1_priority_wait
        cls = [{"lam": 0.7, "es": 1.0, "es2": 1.0}, {"lam": 0.5, "es": 1.0, "es2": 1.0}]
        # sigma through lo class = 1.2 > 1 -> lo wait saturates
        self.assertEqual(mg1_priority_wait(cls, 1), float("inf"))

    def test_t4_comm_is_baseline_plus_W(self):
        import importlib
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        B = importlib.import_module("build_separation_capacity")
        w = B.evidence_queue_wait(0.5, B.LOAD_PRESETS["peak"], c2_dedicated=False)
        self.assertGreater(w, 0.0)
        # heavier airtime -> larger W (monotone)
        w_big = B.evidence_queue_wait(2.0, B.LOAD_PRESETS["peak"], c2_dedicated=False)
        self.assertGreater(w_big, w)
        # dedicated C2 removes the C2 top-class load -> strictly smaller evidence wait
        w_ded = B.evidence_queue_wait(0.5, B.LOAD_PRESETS["peak"], c2_dedicated=True)
        self.assertLess(w_ded, w)

    def test_env_queue_model_legacy_default(self):
        # legacy queue model reproduces the affine formula exactly.
        env = _env("legacy")
        env.env_cfg["queue_delay_scale_s"] = 0.5
        env.env_cfg["gpu_queue_delay_scale_s"] = 0.2
        from types import SimpleNamespace
        edge = SimpleNamespace(load=0.4, gpu_load=0.3)
        # legacy: 0.4*0.5 + 0.3*0.2 = 0.26
        self.assertAlmostEqual(env._queue_delay_s(edge, 0.1), 0.26, places=9)

    def test_env_queue_model_mg1_optin(self):
        env = _env("legacy", extra_env={"queue_model": "mg1", "queue_delay_scale_s": 0.5,
                                        "gpu_queue_delay_scale_s": 0.0})
        from types import SimpleNamespace
        edge = SimpleNamespace(load=0.4, gpu_load=0.0)
        # mg1 wait is finite and positive for rho 0.4, service 0.1
        w = env._queue_delay_s(edge, 0.1)
        self.assertGreater(w, 0.0)
        self.assertTrue(w == w and w != float("inf"))  # finite


class EscalationBudgetDeterminismTest(unittest.TestCase):
    """Task #34-(i): the delta_esc estimator is deterministic (same seed/config
    -> same decomposition), so the single calibration JSON is reproducible."""

    def test_spec_decompose_deterministic(self):
        import importlib
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        V5 = importlib.import_module("calibrate_epsilon_v5")
        per_task = [
            {"qtype": "presence", "counting_ge10": False,
             "svc": {1: (0.60, True, True), 2: (0.80, False, False)}},
            {"qtype": "counting", "counting_ge10": True,
             "svc": {1: (0.30, True, True), 2: (0.40, True, True)}},
        ]
        d1 = V5.spec_decompose(per_task, 0.464, 0.696)
        d2 = V5.spec_decompose(per_task, 0.464, 0.696)
        self.assertEqual(d1, d2)
        # task 1: presence eps 0.696; lvl2 lcb 0.80>=eps but deadline False,
        #   lvl1 lcb 0.60<eps -> quality-attainable via lvl2 but NOT deadline ->
        #   deadline_blocked.  task 2: counting eps 0.464; both lcb<eps ->
        #   quality_unreachable.
        self.assertAlmostEqual(d1["spec_unattainable"], 1.0, places=9)
        self.assertAlmostEqual(d1["quality_unreachable"], 0.5, places=9)
        self.assertAlmostEqual(d1["deadline_blocked"], 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
