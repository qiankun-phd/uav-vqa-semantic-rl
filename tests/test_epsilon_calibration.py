from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.sim.multi_uav_env import (  # noqa: E402
    ATTAINABILITY_V1_EPSILON,
    ATTAINABILITY_V2_EPSILON,
    MultiUAVVQAEnv,
)
from vqa_semcom.sim.resource_env import LUTEntry  # noqa: E402


def _env() -> MultiUAVVQAEnv:
    tasks = [
        {
            "question_type": "presence",
            "question": "Are there cars?",
            "risk_level": "normal",
            "epsilon_k": "0.5",
            "tau_k": "30.0",
            "view_quality_bin": "good",
            "freshness_bin": "fresh",
        }
    ]
    lut = {
        ("presence", lvl, snr, "good", "fresh", "normal"): LUTEntry(0.5, 0.0)
        for lvl in (0, 1, 2)
        for snr in ("0dB", "10dB", "20dB")
    }
    cfg = {
        "bins": {"snr_db": [0, 10, 20], "freshness": ["fresh"], "service_levels": [0, 1, 2]},
        "simulation": {"seed": 5, "bandwidth_hz": 1_000_000},
        "multi_uav_env": {
            "num_uavs": 2,
            "episode_steps": 4,
            "tasks_per_episode": 1,
            "snr_noise_std_db": 0.0,
            "base_snr_db": 12.0,
            "area_spacing_m": 100.0,
            "uav_speed_mps": 20.0,
        },
    }
    return MultiUAVVQAEnv(tasks, lut, cfg, seed=5)


class EpsilonCalibrationTest(unittest.TestCase):
    """Task #28: attainability-anchored quality constraint vs legacy constants."""

    def setUp(self) -> None:
        self.env = _env()
        # Neutralise scenario scaling/caps so we test the calibration branch in
        # isolation (no floors, no epsilon_scale, no epsilon_cap_by_risk).
        self.env.scenario_cfg = {"task_layout": {}}
        self.row = {"question_type": "presence", "view_quality_bin": "good"}

    def _eps(self, mode, risk):
        env_cfg = {} if mode is None else {"epsilon_calibration": mode}
        self.env.env_cfg = env_cfg
        return self.env._epsilon_for_task(self.row, risk)

    def test_legacy_default_bit_identical(self):
        # Absent key defaults to legacy; row has no epsilon_k -> default constants.
        self.assertAlmostEqual(self._eps(None, "critical"), 0.82, places=9)
        self.assertAlmostEqual(self._eps(None, "normal"), 0.65, places=9)
        # Explicit "legacy" identical to default.
        self.assertAlmostEqual(self._eps("legacy", "critical"), 0.82, places=9)
        self.assertAlmostEqual(self._eps("legacy", "normal"), 0.65, places=9)

    def test_legacy_respects_row_epsilon(self):
        # Legacy still honours a per-row epsilon_k override (base = row value).
        self.env.env_cfg = {}
        eps = self.env._epsilon_for_task({"epsilon_k": "0.77"}, "critical")
        self.assertAlmostEqual(eps, 0.77, places=9)

    def test_attainability_v1_values(self):
        self.assertAlmostEqual(self._eps("attainability_v1", "critical"), 0.615, places=9)
        self.assertAlmostEqual(self._eps("attainability_v1", "normal"), 0.166, places=9)
        self.assertAlmostEqual(self._eps("attainability_v1", "high"), 0.615, places=9)

    def test_attainability_ignores_row_and_scaling(self):
        # Recalibrated mode is a flat per-risk constant: row epsilon_k and
        # scenario epsilon_scale must NOT perturb it.
        self.env.env_cfg = {"epsilon_calibration": "attainability_v1"}
        self.env.scenario_cfg = {"task_layout": {"epsilon_scale": 0.5,
                                                 "epsilon_cap_by_risk": {"critical": 0.4}}}
        eps = self.env._epsilon_for_task({"epsilon_k": "0.99"}, "critical")
        self.assertAlmostEqual(eps, 0.615, places=9)

    def test_calibration_table_matches_anchor(self):
        # Provenance guard: constants == documented ratio x oracle LCB ceiling.
        self.assertAlmostEqual(ATTAINABILITY_V1_EPSILON["critical"], round(0.90 * 0.6835, 3), places=9)
        self.assertAlmostEqual(ATTAINABILITY_V1_EPSILON["normal"], round(0.75 * 0.2209, 3), places=9)

    def test_attainability_v2_values(self):
        # v2 quantile-anchored + cache-guarded constants (docs/EPSILON_RECAL_V2.md).
        self.assertAlmostEqual(self._eps("attainability_v2", "critical"), 0.633, places=9)
        self.assertAlmostEqual(self._eps("attainability_v2", "normal"), 0.297, places=9)
        self.assertAlmostEqual(self._eps("attainability_v2", "high"), 0.633, places=9)

    def test_attainability_v2_critical_above_cache_p90(self):
        # Guardrail invariant: eps_critical must clear the measured cache
        # accuracy ceiling (P90 = 0.583) by at least the 0.05 margin, so cache
        # can never be a critical-task compliance shortcut.
        cache_p90 = 0.583
        self.assertGreaterEqual(ATTAINABILITY_V2_EPSILON["critical"], cache_p90 + 0.05 - 1e-9)

    def test_attainability_v2_ignores_row_and_scaling(self):
        # v2, like v1, is a flat per-risk constant: row epsilon_k and scenario
        # epsilon_scale / caps must not perturb it.
        self.env.env_cfg = {"epsilon_calibration": "attainability_v2"}
        self.env.scenario_cfg = {"task_layout": {"epsilon_scale": 0.5,
                                                 "epsilon_cap_by_risk": {"critical": 0.4}}}
        eps = self.env._epsilon_for_task({"epsilon_k": "0.99"}, "critical")
        self.assertAlmostEqual(eps, 0.633, places=9)

    def test_v1_and_legacy_unaffected_by_v2(self):
        # Adding v2 must not perturb legacy or v1 selection.
        self.assertAlmostEqual(self._eps("legacy", "critical"), 0.82, places=9)
        self.assertAlmostEqual(self._eps("attainability_v1", "critical"), 0.615, places=9)
        self.assertAlmostEqual(self._eps("attainability_v1", "normal"), 0.166, places=9)
        self.assertEqual(ATTAINABILITY_V1_EPSILON, {"critical": 0.615, "normal": 0.166, "high": 0.615})



if __name__ == "__main__":
    unittest.main()
