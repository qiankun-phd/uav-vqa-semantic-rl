from __future__ import annotations

import math
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.sim import bubbles_separation as bs
from vqa_semcom.sim.multi_uav_env import MultiUAVVQAEnv
from vqa_semcom.sim.resource_env import LUTEntry


# --------------------------------------------------------------------------- #
# (a) Reproduce D2.1 Appendix G, Table G-4 (p.124) horizontal separation minima #
# --------------------------------------------------------------------------- #
class SeparationChainTest(unittest.TestCase):
    def test_t4_distribution_matches_table_g2(self) -> None:
        mean, std = bs.t4_distribution()
        self.assertAlmostEqual(mean, 7.82, places=2)   # Table G-2 total average delay
        self.assertAlmostEqual(std, 4.24, places=2)    # Table G-2 total std dev

    def test_sail_i_ii_horizontal_tc_matches_table_g4(self) -> None:
        # Table G-4 (p.124): SAIL I-II horizontal TC = 370.53 m.
        d_tc, _ = bs.tactical_conflict_distance(bs.TABLE_B2["SAIL_I_II"])
        rel_err = abs(d_tc - 370.53) / 370.53
        self.assertLess(rel_err, 0.01, f"d_TC={d_tc:.2f} m, rel_err={rel_err:.4%}")

    def test_sail_i_ii_vertical_tc_matches_table_g5(self) -> None:
        # Table G-4/G-5 vertical column: SAIL I-II vertical TC = 31.46 m.
        _, h_tc = bs.tactical_conflict_distance(bs.TABLE_B2["SAIL_I_II"])
        self.assertAlmostEqual(h_tc, 31.46, delta=0.2)

    def test_minima_are_monotonic_chain(self) -> None:
        m = bs.separation_minima(bs.TABLE_B2["SAIL_III_IV"])
        self.assertLess(m.d_nmac_m, m.d_ic_m)
        self.assertLess(m.d_ic_m, m.d_sl_m)
        self.assertLess(m.d_sl_m, m.d_tc_m)

    def test_lower_confidence_shrinks_tc(self) -> None:
        perf = bs.TABLE_B2["SAIL_I_II"]
        d_2sigma, _ = bs.tactical_conflict_distance(perf, confidence_sigma=2.0)
        d_0sigma, _ = bs.tactical_conflict_distance(perf, confidence_sigma=0.0)
        self.assertLess(d_0sigma, d_2sigma)


# --------------------------------------------------------------------------- #
# (b) CPA two-condition tactical-conflict geometry (head-on/parallel/crossing) #
# --------------------------------------------------------------------------- #
class CPAGeometryTest(unittest.TestCase):
    def test_head_on_is_conflict(self) -> None:
        # Two aircraft 400 m apart on the x-axis closing head-on at 14 m/s.
        p1, v1 = (0.0, 0.0, 60.0), (14.0, 0.0, 0.0)
        p2, v2 = (400.0, 0.0, 60.0), (-14.0, 0.0, 0.0)
        cpa = bs.closest_point_of_approach(p1, v1, p2, v2)
        self.assertGreater(cpa.time_to_cpa_s, 0.0)
        self.assertLess(cpa.horizontal_sep_m, 1.0)     # they nearly collide
        d_tc, h_tc = bs.tactical_conflict_distance(bs.TABLE_B2["SAIL_III_IV"])
        self.assertTrue(bs.is_tactical_conflict(p1, v1, p2, v2, d_tc, 60.0, h_tc_m=h_tc))

    def test_parallel_same_speed_is_not_conflict(self) -> None:
        # Parallel tracks 30 m apart, identical velocity -> never converge.
        p1, v1 = (0.0, 0.0, 60.0), (14.0, 0.0, 0.0)
        p2, v2 = (0.0, 30.0, 60.0), (14.0, 0.0, 0.0)
        cpa = bs.closest_point_of_approach(p1, v1, p2, v2)
        self.assertTrue(math.isinf(cpa.time_to_cpa_s))
        d_tc, h_tc = bs.tactical_conflict_distance(bs.TABLE_B2["SAIL_III_IV"])
        # Horizontal sep (30 m) < d_TC but time-to-CPA is infinite -> no conflict.
        self.assertFalse(bs.is_tactical_conflict(p1, v1, p2, v2, d_tc, 60.0, h_tc_m=h_tc))

    def test_crossing_paths_conflict_and_far_miss(self) -> None:
        d_tc, h_tc = bs.tactical_conflict_distance(bs.TABLE_B2["SAIL_III_IV"])
        # Crossing at the origin, both arriving simultaneously -> conflict.
        p1, v1 = (-140.0, 0.0, 60.0), (14.0, 0.0, 0.0)
        p2, v2 = (0.0, -140.0, 60.0), (0.0, 14.0, 0.0)
        cpa = bs.closest_point_of_approach(p1, v1, p2, v2)
        self.assertAlmostEqual(cpa.time_to_cpa_s, 10.0, delta=0.5)
        self.assertLess(cpa.horizontal_sep_m, 1.0)
        self.assertTrue(bs.is_tactical_conflict(p1, v1, p2, v2, d_tc, 60.0, h_tc_m=h_tc))
        # Same crossing but the second aircraft is offset in time -> large miss.
        p2b = (0.0, -1400.0, 60.0)
        far = bs.closest_point_of_approach(p1, v1, p2b, v2)
        self.assertGreater(far.horizontal_sep_m, d_tc)
        self.assertFalse(bs.is_tactical_conflict(p1, v1, p2b, v2, d_tc, 60.0, h_tc_m=h_tc))

    def test_time_threshold_gates_slow_closure(self) -> None:
        # Closing but very slowly -> time-to-CPA exceeds threshold -> no conflict.
        d_tc, h_tc = bs.tactical_conflict_distance(bs.TABLE_B2["SAIL_III_IV"])
        p1, v1 = (0.0, 0.0, 60.0), (10.0, 0.0, 0.0)
        p2, v2 = (5000.0, 5.0, 60.0), (9.9, 0.0, 0.0)  # nearly co-speed, far away
        self.assertFalse(bs.is_tactical_conflict(p1, v1, p2, v2, d_tc, 60.0, h_tc_m=h_tc))

    def test_daily_curve_matches_table_g13(self) -> None:
        self.assertEqual(len(bs.BUBBLES_DAILY_DEMAND), 24)
        self.assertEqual(max(bs.BUBBLES_DAILY_DEMAND), 20)
        self.assertAlmostEqual(bs.BUBBLES_DAILY_MEAN_CONCURRENCY, 8.17, places=2)
        # Peak hours (09:00-13:00) should receive generation steps in the busy band.
        steps = [bs.bubbles_daily_generation_step(i, 24, 12) for i in range(24)]
        self.assertTrue(all(0 <= s < 12 for s in steps))
        self.assertEqual(steps, sorted(steps))  # monotone with task index


# --------------------------------------------------------------------------- #
# (c) Regression: profile OFF is bit-identical to the untouched default path.   #
# --------------------------------------------------------------------------- #
def _tasks() -> list[dict[str, str]]:
    return [
        {
            "question_type": "presence",
            "question": "Are there cars?",
            "risk_level": "normal",
            "epsilon_k": "0.5",
            "tau_k": "30.0",
            "view_quality_bin": "good",
            "freshness_bin": "fresh",
        },
        {
            "question_type": "presence",
            "question": "Any people?",
            "risk_level": "critical",
            "epsilon_k": "0.6",
            "tau_k": "20.0",
            "view_quality_bin": "good",
            "freshness_bin": "fresh",
        },
    ]


def _lut() -> dict:
    lut = {}
    for snr in ("0dB", "10dB", "20dB"):
        lut[("presence", 0, snr, "good", "fresh", "normal")] = LUTEntry(0.45, 0.0)
        lut[("presence", 1, snr, "good", "fresh", "normal")] = LUTEntry(0.70, 2048.0)
        lut[("presence", 2, snr, "good", "fresh", "normal")] = LUTEntry(0.90, 300000.0)
        lut[("presence", 0, snr, "good", "fresh", "critical")] = LUTEntry(0.45, 0.0)
        lut[("presence", 1, snr, "good", "fresh", "critical")] = LUTEntry(0.70, 2048.0)
        lut[("presence", 2, snr, "good", "fresh", "critical")] = LUTEntry(0.90, 300000.0)
    return lut


def _cfg(profile: str | None) -> dict:
    # Wide, non-overlapping operational volumes with active background intents:
    # the legacy Area4D-overlap test then reports no strategic conflict, while
    # the BUBBLES CPA test still flags converging trajectories -- so the two
    # branches diverge and the gate is demonstrably not a no-op.
    env = {
        "num_uavs": 3,
        "num_areas": 3,
        "episode_steps": 12,
        "tasks_per_episode": 9,
        "area_spacing_m": 900.0,
        "area_radius_m": 120.0,
        "uav_speed_mps": 18.0,
        "utm": {
            "enabled": True,
            "background_operational_intents": True,
            "background_operational_intent_density": 1.0,
            "spatial_buffer_m": 10.0,
        },
    }
    if profile is not None:
        env["scenario_profile"] = profile
    return {
        "bins": {"snr_db": [0, 10, 20], "freshness": ["fresh"], "service_levels": [0, 1, 2]},
        "simulation": {"seed": 5, "bandwidth_hz": 1_000_000},
        "multi_uav_env": env,
    }


def _rollout(cfg: dict, steps: int = 24) -> list[tuple]:
    env = MultiUAVVQAEnv(_tasks(), _lut(), cfg, seed=5)
    env.reset(seed=5, options={"tasks_per_episode": 9})
    trace = []
    for i in range(steps):
        action = {
            "service_level": 1 + (i % 2),
            "uav_assignment": i % 3,
            "mobility_mode": "serve_task",
        }
        _, reward, done, info = env.step(action)
        trace.append(
            (
                round(float(reward), 6),
                bool(info.get("airspace_conflict", False)),
                round(float(info.get("delay_s", 0.0)), 6),
                round(float(info.get("energy_j", 0.0)), 6),
                bool(info.get("success", False)),
            )
        )
        if done:
            env.reset(seed=5, options={"tasks_per_episode": 9})
    return trace


class RegressionTest(unittest.TestCase):
    def test_profile_off_identical_to_default(self) -> None:
        # Absent key vs explicit non-bubbles value must yield identical rollouts.
        base = _rollout(_cfg(None))
        off_null = _rollout(_cfg(""))
        off_nominal = _rollout(_cfg("nominal"))
        self.assertEqual(base, off_null)
        self.assertEqual(base, off_nominal)

    def test_default_uav_speed_unchanged(self) -> None:
        env = MultiUAVVQAEnv(_tasks(), _lut(), _cfg(None), seed=5)
        env.reset(seed=5)
        self.assertEqual(float(env.env_cfg["uav_speed_mps"]), 18.0)
        self.assertNotIn("uav_roc_mps", env.env_cfg)

    def test_bubbles_profile_overrides_envelope(self) -> None:
        env = MultiUAVVQAEnv(_tasks(), _lut(), _cfg("bubbles"), seed=5)
        env.reset(seed=5)
        self.assertEqual(float(env.env_cfg["uav_speed_mps"]), 14.0)   # SAIL III-IV cruise
        self.assertEqual(float(env.env_cfg["uav_roc_mps"]), 5.0)
        self.assertEqual(float(env.env_cfg["uav_rod_mps"]), 4.0)
        self.assertEqual(float(env.env_cfg["tls_mac_fat_per_fh"]), 2.5e-7)

    def test_bubbles_profile_changes_conflict_rate(self) -> None:
        base = _rollout(_cfg(None))
        bubbles = _rollout(_cfg("bubbles"))
        base_conflicts = sum(1 for row in base if row[1])
        bubbles_conflicts = sum(1 for row in bubbles if row[1])
        # The CPA criterion must produce a *different* conflict signal than the
        # legacy Area4D overlap test (otherwise the gate is a no-op).
        self.assertNotEqual(base_conflicts, bubbles_conflicts)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
