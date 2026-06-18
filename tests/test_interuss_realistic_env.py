from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.sim.multi_uav_env import (
    Area4D,
    MultiUAVVQAEnv,
    OPERATIONAL_INTENT_STATES,
    available_utm_realistic_scenarios,
)
from vqa_semcom.sim.resource_env import LUTEntry


class InterUSSRealisticEnvTest(unittest.TestCase):
    def _env(self) -> MultiUAVVQAEnv:
        tasks = [
            {
                "question_type": "presence",
                "question": "Are emergency vehicles present?",
                "risk_level": "normal",
                "epsilon_k": "0.50",
                "tau_k": "30.0",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
            },
            {
                "question_type": "presence",
                "question": "Is the road blocked?",
                "risk_level": "critical",
                "epsilon_k": "0.60",
                "tau_k": "30.0",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
            },
        ]
        lut = {}
        for risk in ("normal", "critical"):
            for view in ("poor", "medium", "good"):
                for fresh in ("fresh", "stale", "expired"):
                    for snr in ("0dB", "10dB", "20dB"):
                        lut[("presence", 0, snr, view, fresh, risk)] = LUTEntry(0.45, 0.0)
                        lut[("presence", 1, snr, view, fresh, risk)] = LUTEntry(0.72, 2048.0)
                        lut[("presence", 2, snr, view, fresh, risk)] = LUTEntry(0.90, 300000.0)
        cfg = {
            "bins": {
                "snr_db": [0, 10, 20],
                "freshness": ["fresh", "stale", "expired"],
                "service_levels": [0, 1, 2],
            },
            "simulation": {"seed": 17, "bandwidth_hz": 1_000_000},
            "multi_uav_env": {
                "scenario": "nominal",
                "enabled_service_levels": [0, 1, 2],
                "enable_service_level_3": False,
                "num_uavs": 3,
                "num_edges": 1,
                "tasks_per_episode": 6,
                "episode_steps": 6,
                "area_spacing_m": 120.0,
                "area_radius_m": 70.0,
                "a2g": {"fading_mode": "static"},
            },
        }
        return MultiUAVVQAEnv(tasks, lut, cfg, seed=17)

    def test_utm_realistic_scenarios_are_registered(self) -> None:
        expected = {
            "test_utm_nominal_planning",
            "test_utm_off_nominal_planning",
            "test_utm_intent_conflict",
            "test_utm_dss_outage",
            "test_utm_notification_delay",
        }
        self.assertEqual(set(available_utm_realistic_scenarios()), expected)

    def test_nominal_planning_exports_operational_intent_fields(self) -> None:
        env = self._env()
        env.reset(seed=17, options={"formal_scenario": "test_utm_nominal_planning"})
        task = env._front_task()
        self.assertIsNotNone(task)
        info = env.evaluate_action(
            {"task_id": task.task_id, "service_level": 1, "sensing_decision": "observe"},
            task_id=task.task_id,
        )
        self.assertTrue(info["operational_intent_id"].startswith("oi_"))
        self.assertIn(info["operational_intent_state"], OPERATIONAL_INTENT_STATES)
        self.assertEqual(info["operational_intent_state"], "activated")
        self.assertFalse(info["utm_constraint_violation"])
        self.assertTrue(info["dss_available"])
        self.assertGreaterEqual(info["dss_delay_s"], 0.0)

    def test_intent_conflict_uses_buffered_strategic_detection(self) -> None:
        env = self._env()
        env.reset(seed=17, options={"formal_scenario": "test_utm_intent_conflict"})
        active = env._active_tasks()
        self.assertGreaterEqual(len(active), 2)
        info = env.evaluate_action(
            {
                "task_id": active[0].task_id,
                "service_level": 1,
                "sensing_decision": "observe",
                "concurrent_actions": [
                    {"task_id": active[1].task_id, "service_level": 1, "sensing_decision": "observe"}
                ],
            },
            task_id=active[0].task_id,
        )
        self.assertTrue(info["strategic_conflict"])
        self.assertTrue(info["airspace_conflict"])
        self.assertGreaterEqual(info["strategic_conflict_count"], 1)
        self.assertEqual(info["operational_intent_state"], "nonconforming")

    def test_cache_only_overlap_does_not_create_operational_conflict(self) -> None:
        env = self._env()
        env.reset(seed=17, options={"formal_scenario": "test_utm_intent_conflict"})
        active = env._active_tasks()
        self.assertGreaterEqual(len(active), 2)
        info = env.evaluate_action(
            {
                "task_id": active[0].task_id,
                "service_level": 0,
                "sensing_decision": "reuse_cache",
                "concurrent_actions": [
                    {"task_id": active[1].task_id, "service_level": 1, "sensing_decision": "observe"}
                ],
            },
            task_id=active[0].task_id,
        )
        self.assertFalse(info["strategic_conflict"])
        self.assertFalse(info["airspace_conflict"])
        self.assertFalse(info["utm_constraint_violation"])

    def test_dss_outage_sets_contingent_state_and_violation(self) -> None:
        env = self._env()
        env.reset(seed=17, options={"formal_scenario": "test_utm_dss_outage"})
        task = env._front_task()
        self.assertIsNotNone(task)
        info = env.evaluate_action(
            {"task_id": task.task_id, "service_level": 1, "sensing_decision": "observe"},
            task_id=task.task_id,
        )
        self.assertFalse(info["dss_available"])
        self.assertEqual(info["operational_intent_state"], "contingent")
        self.assertTrue(info["utm_constraint_violation"])
        self.assertGreater(info["utm_dss_delay_s"], 0.0)

    def test_notification_delay_is_reported_for_conflict_updates(self) -> None:
        env = self._env()
        env.reset(seed=17, options={"formal_scenario": "test_utm_notification_delay"})
        active = env._active_tasks()
        self.assertGreaterEqual(len(active), 2)
        info = env.evaluate_action(
            {
                "task_id": active[0].task_id,
                "service_level": 2,
                "sensing_decision": "observe",
                "concurrent_actions": [
                    {"task_id": active[1].task_id, "service_level": 2, "sensing_decision": "observe"}
                ],
            },
            task_id=active[0].task_id,
        )
        self.assertTrue(info["conflict_notification_pending"])
        self.assertGreater(info["subscription_notification_delay_s"], 0.0)
        self.assertGreater(info["utm_notification_delay_s"], 0.0)

    def test_spatial_temporal_buffer_expands_conflict_detection(self) -> None:
        env = self._env()
        env.reset(seed=17, options={"formal_scenario": "test_utm_intent_conflict"})
        first = Area4D(0.0, 0.0, 50.0, 50.0, 100.0, 0, 2)
        second = Area4D(120.0, 0.0, 50.0, 50.0, 100.0, 4, 6)
        self.assertFalse(first.overlaps(second))
        self.assertTrue(env._area4d_overlaps_with_buffer(first, second))


if __name__ == "__main__":
    unittest.main()
