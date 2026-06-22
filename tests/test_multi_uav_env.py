from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.sim.multi_uav_env import (
    Area4D,
    MultiUAVVQAEnv,
    SemanticCacheEntry,
    available_scenarios,
    semantic_scenario_preset_names,
)
from vqa_semcom.sim.resource_env import LUTEntry


class MultiUAVEnvTest(unittest.TestCase):
    def _env(self) -> MultiUAVVQAEnv:
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
            ("presence", 0, "0dB", "good", "fresh", "normal"): LUTEntry(0.40, 0.0),
            ("presence", 1, "0dB", "good", "fresh", "normal"): LUTEntry(0.60, 2048.0),
            ("presence", 2, "0dB", "good", "fresh", "normal"): LUTEntry(0.80, 300000.0),
            ("presence", 0, "10dB", "good", "fresh", "normal"): LUTEntry(0.45, 0.0),
            ("presence", 1, "10dB", "good", "fresh", "normal"): LUTEntry(0.70, 2048.0),
            ("presence", 2, "10dB", "good", "fresh", "normal"): LUTEntry(0.90, 300000.0),
            ("presence", 0, "20dB", "good", "fresh", "normal"): LUTEntry(0.50, 0.0),
            ("presence", 1, "20dB", "good", "fresh", "normal"): LUTEntry(0.80, 2048.0),
            ("presence", 2, "20dB", "good", "fresh", "normal"): LUTEntry(0.95, 300000.0),
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

    def test_reset_observation_contains_contract_fields(self) -> None:
        env = self._env()
        obs = env.reset(seed=5)
        for field in [
            "task_type",
            "risk_level",
            "view_quality_bin",
            "freshness_bin",
            "sensed_snr_db",
            "snr_bin",
            "uav_state",
            "edge_load",
            "cache_state",
        ]:
            self.assertIn(field, obs)
        self.assertEqual(obs["task_type"], "presence")
        self.assertTrue(obs["uav_state"])
        self.assertIn(obs["snr_bin"], {"0dB", "10dB", "20dB"})

    def test_step_info_contains_reward_components(self) -> None:
        env = self._env()
        env.reset(seed=5)
        obs, reward, done, info = env.step(
            {
                "service_level": 1,
                "bandwidth": 1_000_000.0,
                "power": 0.1,
                "cpu_share": 0.5,
                "gpu_share": 0.5,
                "uav_assignment": 0,
                "waypoint": None,
            }
        )
        self.assertIn(done, {True, False})
        self.assertIsInstance(float(reward), float)
        self.assertIn("task_queue", obs)
        for field in [
            "answer_accuracy_est",
            "semantic_accuracy_mean",
            "semantic_accuracy_lcb",
            "semantic_uncertainty",
            "semantic_sample_count",
            "semantic_payload_kb",
            "semantic_quality_gap",
            "semantic_success",
            "delay_s",
            "energy_j",
            "payload_kb",
            "quality_violation",
            "deadline_violation",
            "deadline_s",
            "epsilon_k",
            "risk_level",
            "view_quality_bin",
            "freshness_bin",
            "snr_bin",
            "service_level",
        ]:
            self.assertIn(field, info)
        self.assertEqual(info["service_level"], 1)
        self.assertGreaterEqual(info["answer_accuracy_est"], 0.0)
        self.assertEqual(info["quality_violation"], not info["semantic_success"])
        self.assertAlmostEqual(info["semantic_payload_kb"], info["payload_kb"])

    def test_uav_moves_and_energy_is_spent(self) -> None:
        env = self._env()
        env.reset(seed=5)
        before = (env.uavs[0].x_m, env.uavs[0].y_m, env.uavs[0].battery_j)
        _obs, _reward, _done, info = env.step({"service_level": 2, "uav_assignment": 0, "power": 0.2})
        after = (env.uavs[0].x_m, env.uavs[0].y_m, env.uavs[0].battery_j)
        self.assertNotEqual(before[:2], after[:2])
        self.assertLess(after[2], before[2])
        self.assertGreater(info["delay_s"], 0.0)
        self.assertGreater(info["energy_j"], 0.0)

    def test_area4d_overlap_and_cache_only_conflict_policy(self) -> None:
        env = self._env()
        env.reset(seed=5)
        first = env.tasks[0]
        clone_area = Area4D(
            center_x_m=first.area4d.center_x_m,
            center_y_m=first.area4d.center_y_m,
            radius_m=first.area4d.radius_m,
            altitude_min_m=first.area4d.altitude_min_m,
            altitude_max_m=first.area4d.altitude_max_m,
            start_step=first.area4d.start_step,
            end_step=first.area4d.end_step,
        )
        clone = replace(first, task_id="overlap", completed=False, generation_time=0, area4d=clone_area)
        env.tasks.append(clone)
        self.assertTrue(first.area4d.overlaps(clone.area4d))

        observe_info = env.evaluate_action(
            {
                "task_id": first.task_id,
                "service_level": 1,
                "sensing_decision": "observe",
                "concurrent_actions": [
                    {"task_id": clone.task_id, "service_level": 1, "sensing_decision": "observe"}
                ],
            },
            task_id=first.task_id,
        )
        cache_info = env.evaluate_action(
            {
                "task_id": first.task_id,
                "service_level": 0,
                "sensing_decision": "reuse_cache",
                "concurrent_actions": [
                    {"task_id": clone.task_id, "service_level": 0, "sensing_decision": "reuse_cache"}
                ],
            },
            task_id=first.task_id,
        )
        self.assertTrue(observe_info["airspace_conflict"])
        self.assertFalse(cache_info["airspace_conflict"])

    def test_link_budget_monotonicity_and_interference(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env.tasks[0]
        low_power = env.evaluate_action({"service_level": 2, "power": 0.05}, task_id=task.task_id)
        high_power = env.evaluate_action({"service_level": 2, "power": 1.0}, task_id=task.task_id)
        self.assertGreaterEqual(high_power["rate_mbps"], low_power["rate_mbps"])

        env.uavs[0].x_m = task.x_m
        env.uavs[0].y_m = task.y_m
        near = env.evaluate_action({"service_level": 2, "power": 0.5, "uav_assignment": 0}, task_id=task.task_id)
        env.uavs[0].x_m = task.x_m + 2000.0
        env.uavs[0].y_m = task.y_m + 2000.0
        far = env.evaluate_action({"service_level": 2, "power": 0.5, "uav_assignment": 0}, task_id=task.task_id)
        self.assertLessEqual(far["rate_mbps"], near["rate_mbps"])

        interferer = replace(task, task_id="interferer", completed=False, generation_time=0)
        env.tasks.append(interferer)
        no_interference = env.evaluate_action({"service_level": 2, "power": 0.5}, task_id=task.task_id)
        interfered = env.evaluate_action(
            {
                "service_level": 2,
                "power": 0.5,
                "concurrent_actions": [{"task_id": interferer.task_id, "service_level": 2, "power": 1.0}],
            },
            task_id=task.task_id,
        )
        self.assertLessEqual(interfered["sinr_db"], no_interference["sinr_db"])

    def test_cache_service_semantic_qos_is_snr_invariant(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env.tasks[0]
        low = env.evaluate_action({"service_level": 0}, task_id=task.task_id, obs={"task_id": task.task_id, "snr_bin": "0dB"})
        high = env.evaluate_action({"service_level": 0}, task_id=task.task_id, obs={"task_id": task.task_id, "snr_bin": "20dB"})
        self.assertAlmostEqual(low["semantic_accuracy_mean"], high["semantic_accuracy_mean"])
        self.assertAlmostEqual(low["semantic_accuracy_lcb"], high["semantic_accuracy_lcb"])
        self.assertAlmostEqual(low["semantic_payload_kb"], high["semantic_payload_kb"])

    def test_high_risk_task_has_stricter_semantic_threshold(self) -> None:
        tasks = [
            {
                "question_type": "presence",
                "question": "Normal task?",
                "risk_level": "normal",
                "epsilon_k": "0.50",
                "tau_k": "30.0",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
            },
            {
                "question_type": "presence",
                "question": "Critical task?",
                "risk_level": "critical",
                "epsilon_k": "0.50",
                "tau_k": "30.0",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
            },
        ]
        lut = {
            ("presence", 0, "10dB", "good", "fresh", "normal"): LUTEntry(0.40, 0.0),
            ("presence", 1, "10dB", "good", "fresh", "normal"): LUTEntry(0.70, 2048.0),
            ("presence", 2, "10dB", "good", "fresh", "normal"): LUTEntry(0.90, 300000.0),
            ("presence", 0, "10dB", "good", "fresh", "critical"): LUTEntry(0.40, 0.0),
            ("presence", 1, "10dB", "good", "fresh", "critical"): LUTEntry(0.70, 2048.0),
            ("presence", 2, "10dB", "good", "fresh", "critical"): LUTEntry(0.90, 300000.0),
        }
        cfg = {
            "bins": {"snr_db": [10], "freshness": ["fresh"], "service_levels": [0, 1, 2]},
            "simulation": {"seed": 9, "bandwidth_hz": 1_000_000},
            "multi_uav_env": {
                "num_uavs": 1,
                "episode_steps": 2,
                "tasks_per_episode": 2,
                "num_areas": 1,
                "task_layout": {},
            },
        }
        env = MultiUAVVQAEnv(tasks, lut, cfg, seed=9)
        normal = env._epsilon_for_task(tasks[0], "normal")
        critical = env._epsilon_for_task(tasks[1], "critical")
        self.assertGreater(critical, normal)
        self.assertGreaterEqual(critical, 0.75)

    def test_semantic_cache_and_model_cache_effects(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env.tasks[0]
        before = env._semantic_cache_hit_probability(task)
        env.semantic_cache_entries.append(
            SemanticCacheEntry(
                task_id="previous",
                task_type=task.task_type,
                risk_level=task.risk_level,
                priority=task.priority,
                x_m=task.x_m,
                y_m=task.y_m,
                cache_age=0,
                updated_step=0,
            )
        )
        after = env._semantic_cache_hit_probability(task)
        self.assertGreater(after, before)

        env.edges[0].cached_service_levels = ()
        miss = env.evaluate_action({"service_level": 2, "gpu_share": 0.8}, task_id=task.task_id)
        env.edges[0].cached_service_levels = (2,)
        hit = env.evaluate_action({"service_level": 2, "gpu_share": 0.8}, task_id=task.task_id)
        self.assertLess(hit["load_delay_s"], miss["load_delay_s"])

    def test_action_contract_keeps_service_three_disabled_by_default(self) -> None:
        env = self._env()
        env.reset(seed=5)
        self.assertEqual(env.action_spec()["service_levels"], [0, 1, 2])
        self.assertNotIn(3, env.action_mask()["service_level_allowed"])
        parsed = env.parse_action({"service_level": 3})
        self.assertIn(parsed["service_level"], [0, 1, 2])

    def test_fixed_scenario_presets_are_available_and_keep_roi_disabled(self) -> None:
        expected = {"conflict-heavy", "interference-heavy", "cache-heavy", "mobility-stress"}
        self.assertTrue(expected.issubset(set(available_scenarios())))
        env = self._env()
        for scenario in expected:
            obs = env.reset(seed=5, options={"scenario": scenario})
            self.assertEqual(obs["action_mask"]["service_level_allowed"], {0: True, 1: True, 2: True})
            self.assertNotIn(3, env.service_levels())

    def test_paper_scenario_presets_reset_step_and_export_semantic_risk_fields(self) -> None:
        expected = {
            "nominal_patrol",
            "disaster_hotspot",
            "low_snr_blockage",
            "edge_overload",
            "utm_conflict",
        }
        self.assertEqual(set(semantic_scenario_preset_names()), expected)
        self.assertTrue(expected.issubset(set(available_scenarios())))

        required_info = [
            "semantic_accuracy_lcb",
            "semantic_quality_gap",
            "epsilon_k",
            "deadline_s",
            "energy_j",
            "utm_delay_s",
            "utm_conflict_violation",
            "risk_violation",
            "airspace_state",
        ]
        env = self._env()
        for scenario in semantic_scenario_preset_names():
            env.reset(seed=5, options={"scenario": scenario})
            task = env._front_task()
            self.assertIsNotNone(task)
            info = env.evaluate_action(env.default_action(1), task_id=task.task_id)
            for field in required_info:
                self.assertIn(field, info)
            self.assertGreaterEqual(info["semantic_quality_gap"], 0.0)
            self.assertAlmostEqual(
                info["semantic_quality_gap"],
                max(0.0, info["epsilon_k"] - info["semantic_accuracy_lcb"]),
                places=6,
            )
            self.assertNotIn(3, env.service_levels())

    def test_paper_scenario_presets_have_distinct_stress_knobs(self) -> None:
        disaster = self._env()
        disaster.reset(seed=5, options={"scenario": "disaster_hotspot"})
        critical_ratio = sum(task.risk_level == "critical" for task in disaster.tasks) / len(disaster.tasks)
        self.assertGreaterEqual(critical_ratio, 0.6)
        self.assertGreaterEqual(disaster.env_cfg["semantic_threshold_by_risk"]["critical"], 0.84)

        low_snr = self._env()
        low_snr.reset(seed=5, options={"scenario": "low_snr_blockage"})
        self.assertGreater(float(low_snr.env_cfg["a2g"]["excess_loss_db"]), 10.0)
        self.assertGreater(float(low_snr.env_cfg["a2g"]["nlos_excess_loss_db"]), 20.0)

        edge = self._env()
        edge.reset(seed=5, options={"scenario": "edge_overload"})
        self.assertGreaterEqual(edge.edges[0].load, 0.70)
        self.assertGreaterEqual(edge.edges[0].gpu_load, 0.70)
        self.assertEqual(edge.env_cfg["model_cache_capacity"], 1)

        utm = self._env()
        utm.reset(seed=5, options={"scenario": "utm_conflict"})
        active = utm._active_tasks()
        self.assertGreaterEqual(len(active), 2)
        info = utm.evaluate_action(
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
        self.assertTrue(info["utm_conflict_violation"])
        self.assertGreater(info["utm_delay_s"], 0.0)
        self.assertIn(info["airspace_state"], {"accepted", "activated", "nonconforming", "contingent"})

    def test_conflict_heavy_has_overlapping_burst_tasks(self) -> None:
        env = self._env()
        env.reset(seed=5, options={"scenario": "conflict-heavy"})
        active = env._active_tasks()
        self.assertGreaterEqual(len(active), 2)
        self.assertTrue(active[0].area4d.overlaps(active[1].area4d))
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
        self.assertTrue(info["airspace_conflict"])

    def test_cache_heavy_seeds_semantic_cache(self) -> None:
        env = self._env()
        env.reset(seed=5, options={"scenario": "cache-heavy"})
        self.assertGreater(len(env.semantic_cache_entries), 0)
        task = env.tasks[0]
        self.assertGreaterEqual(env._semantic_cache_hit_probability(task), 0.98)

    def test_mobility_stress_increases_fly_delay(self) -> None:
        nominal = self._env()
        nominal.reset(seed=5, options={"scenario": "nominal"})
        mobility = self._env()
        mobility.reset(seed=5, options={"scenario": "mobility-stress"})
        nominal_task = nominal.tasks[0]
        mobility_task = mobility.tasks[0]
        nominal_info = nominal.evaluate_action({"service_level": 2, "uav_assignment": 0}, task_id=nominal_task.task_id)
        mobility_info = mobility.evaluate_action({"service_level": 2, "uav_assignment": 0}, task_id=mobility_task.task_id)
        self.assertGreater(mobility_info["fly_delay_s"], nominal_info["fly_delay_s"])

    def test_interference_heavy_has_stronger_concurrent_penalty(self) -> None:
        nominal = self._env()
        nominal.reset(seed=5, options={"scenario": "nominal"})
        interference = self._env()
        interference.reset(seed=5, options={"scenario": "interference-heavy"})
        self.assertGreater(
            float(interference.env_cfg["a2g"]["interference_overlap_scale"]),
            float(nominal.env_cfg["a2g"]["interference_overlap_scale"]),
        )

    def test_interfaces_doc_matches_service_and_semantic_qos_fields(self) -> None:
        doc = (ROOT / "docs" / "interfaces.md").read_text(encoding="utf-8")
        self.assertIn("0 cache answer", doc)
        self.assertIn("1 detector semantic tokens", doc)
        self.assertIn("2 raw image evidence", doc)
        self.assertIn("semantic_accuracy_lcb", doc)
        self.assertIn("semantic_uncertainty", doc)
        self.assertIn("semantic_sample_count", doc)


if __name__ == "__main__":
    unittest.main()
