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
            "uav_task_distances_m",
            "uav_battery_ratio",
            "predicted_fly_delay_s",
            "predicted_fly_energy_j",
            "task_area4d",
            "utm_conflict_risk",
            "future_task_proximity",
            "coverage_score_by_uav",
            "feasible_mobility_mask",
            "mobility_actor_state",
        ]:
            self.assertIn(field, obs)
        self.assertEqual(obs["task_type"], "presence")
        self.assertTrue(obs["uav_state"])
        self.assertIn(obs["snr_bin"], {"0dB", "10dB", "20dB"})
        self.assertIn("serve_task", obs["feasible_mobility_mask"])

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
            "mobility_mode",
            "waypoint_x",
            "waypoint_y",
            "altitude_m",
            "fly_distance_m",
            "coverage_gain",
            "mobility_energy_j",
            "arrival_delay_s",
            "utm_conflict_risk",
        ]:
            self.assertIn(field, info)
        self.assertEqual(info["service_level"], 1)
        self.assertEqual(info["mobility_mode"], "serve_task")
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

    def test_mobility_stay_hovers_without_moving(self) -> None:
        env = self._env()
        env.reset(seed=5)
        before = (env.uavs[0].x_m, env.uavs[0].y_m, env.uavs[0].altitude_m)
        _obs, _reward, _done, info = env.step(
            {"service_level": 0, "uav_assignment": 0, "mobility_mode": "stay"}
        )
        after = (env.uavs[0].x_m, env.uavs[0].y_m, env.uavs[0].altitude_m)
        self.assertEqual(before, after)
        self.assertEqual(info["mobility_mode"], "stay")
        self.assertEqual(info["fly_distance_m"], 0.0)
        self.assertGreater(info["mobility_energy_j"], 0.0)

    def test_mobility_serve_task_moves_toward_task_area(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env._front_task()
        self.assertIsNotNone(task)
        before = task.area4d.distance_to(env.uavs[0].x_m, env.uavs[0].y_m)
        _obs, _reward, _done, info = env.step(
            {"service_level": 1, "uav_assignment": 0, "mobility_mode": "serve_task"}
        )
        after = task.area4d.distance_to(env.uavs[0].x_m, env.uavs[0].y_m)
        self.assertLess(after, before)
        self.assertEqual(info["mobility_mode"], "serve_task")
        self.assertGreater(info["arrival_delay_s"], 0.0)

    def test_mobility_reposition_uses_waypoint_delta(self) -> None:
        env = self._env()
        env.reset(seed=5)
        before = (env.uavs[0].x_m, env.uavs[0].y_m)
        _obs, _reward, _done, info = env.step(
            {
                "service_level": 0,
                "uav_assignment": 0,
                "mobility_mode": "reposition",
                "waypoint_delta": [40.0, 0.0],
            }
        )
        after = (env.uavs[0].x_m, env.uavs[0].y_m)
        self.assertGreater(after[0], before[0])
        self.assertAlmostEqual(after[1], before[1], places=6)
        self.assertEqual(info["mobility_mode"], "reposition")
        self.assertGreater(info["fly_distance_m"], 0.0)

    def test_mobility_avoid_conflict_reduces_predicted_risk(self) -> None:
        env = self._env()
        env.reset(seed=5, options={"scenario": "utm_conflict"})
        active = env._active_tasks()
        task = next(
            (
                item
                for item in active
                if env.evaluate_action(
                    {"task_id": item.task_id, "service_level": 1, "mobility_mode": "serve_task"},
                    task_id=item.task_id,
                )["utm_conflict_risk"]
                > 0.0
            ),
            None,
        )
        self.assertIsNotNone(task)
        serve = env.evaluate_action(
            {"task_id": task.task_id, "service_level": 1, "mobility_mode": "serve_task"},
            task_id=task.task_id,
        )
        avoid = env.evaluate_action(
            {"task_id": task.task_id, "service_level": 1, "mobility_mode": "avoid_conflict"},
            task_id=task.task_id,
        )
        self.assertGreater(serve["utm_conflict_risk"], avoid["utm_conflict_risk"])
        self.assertEqual(avoid["mobility_mode"], "avoid_conflict")

    def test_cache_service_does_not_force_operational_intent_with_mobility_fields(self) -> None:
        env = self._env()
        env.reset(seed=5, options={"scenario": "utm_conflict"})
        task = env._front_task()
        self.assertIsNotNone(task)
        info = env.evaluate_action(
            {"task_id": task.task_id, "service_level": 0, "mobility_mode": "serve_task"},
            task_id=task.task_id,
        )
        self.assertFalse(info["utm_conflict_violation"])
        self.assertFalse(info["airspace_conflict"])

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

    def test_expired_cache_is_not_eligible(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env.tasks[0]
        env.semantic_cache_entries.append(
            SemanticCacheEntry(
                task_id="expired_cache",
                task_type=task.task_type,
                risk_level=task.risk_level,
                priority=task.priority,
                x_m=task.x_m,
                y_m=task.y_m,
                cache_age=10,
                updated_step=0,
                area_id=task.area_id,
                question_type=task.task_type,
                quality_lcb=0.95,
                uncertainty=0.01,
            )
        )
        info = env.evaluate_action({"semantic_path": "cache"}, task_id=task.task_id)
        self.assertTrue(info["cache_exact_match"])
        self.assertEqual(info["cache_freshness_bin"], "expired")
        self.assertFalse(info["cache_eligible"])

    def test_low_quality_cache_is_not_eligible(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env.tasks[0]
        env.semantic_cache_entries.append(
            SemanticCacheEntry(
                task_id="weak_cache",
                task_type=task.task_type,
                risk_level=task.risk_level,
                priority=task.priority,
                x_m=task.x_m,
                y_m=task.y_m,
                cache_age=0,
                updated_step=0,
                area_id=task.area_id,
                question_type=task.task_type,
                quality_lcb=task.epsilon_k - 0.1,
                uncertainty=0.2,
            )
        )
        info = env.evaluate_action({"semantic_path": "cache"}, task_id=task.task_id)
        self.assertTrue(info["cache_exact_match"])
        self.assertFalse(info["cache_eligible"])
        self.assertTrue(info["quality_violation"])

    def test_nearby_same_type_cache_can_be_eligible(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env.tasks[0]
        env.semantic_cache_entries.append(
            SemanticCacheEntry(
                task_id="nearby_cache",
                task_type=task.task_type,
                risk_level=task.risk_level,
                priority=task.priority,
                x_m=task.x_m + 5.0,
                y_m=task.y_m + 5.0,
                cache_age=0,
                updated_step=0,
                area_id=task.area_id + 99,
                question_type=task.task_type,
                quality_lcb=task.epsilon_k + 0.1,
                uncertainty=0.05,
            )
        )
        info = env.evaluate_action({"semantic_path": "cache"}, task_id=task.task_id)
        self.assertFalse(info["cache_exact_match"])
        self.assertTrue(info["cache_nearby_match"])
        self.assertTrue(info["cache_eligible"])
        self.assertGreaterEqual(info["semantic_accuracy_lcb"], task.epsilon_k)

    def test_defer_keeps_task_in_queue_without_completion(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env.tasks[0]
        _obs, _reward, _done, info = env.step({"task_id": task.task_id, "semantic_path": "defer"})
        self.assertFalse(task.completed)
        self.assertFalse(info["success"])
        self.assertEqual(task.task_status, "deferred")
        self.assertEqual(task.defer_count, 1)
        self.assertLess(info["remaining_deadline_s"], info["deadline_s"])

    def test_token_image_and_cache_update_refresh_semantic_cache(self) -> None:
        for action in [
            {"semantic_path": "token", "bandwidth": 1_000_000.0, "power": 1.0, "cpu_share": 1.0, "gpu_share": 1.0},
            {"semantic_path": "image", "bandwidth": 1_000_000.0, "power": 1.0, "cpu_share": 1.0, "gpu_share": 1.0},
            {"semantic_path": "cache_update", "bandwidth": 1_000_000.0, "power": 1.0, "cpu_share": 1.0, "gpu_share": 1.0},
        ]:
            env = self._env()
            env.reset(seed=5)
            task = env.tasks[0]
            before = len(env.semantic_cache_entries)
            _obs, _reward, _done, info = env.step({"task_id": task.task_id, **action})
            self.assertTrue(info["success"])
            self.assertGreater(len(env.semantic_cache_entries), before)
            entry = env.semantic_cache_entries[0]
            self.assertEqual(entry.area_id, task.area_id)
            self.assertEqual(entry.question_type, task.task_type)
            self.assertGreaterEqual(entry.quality_lcb, task.epsilon_k)

    def test_candidate_path_metrics_contains_all_semantic_paths(self) -> None:
        env = self._env()
        obs = env.reset(seed=5)
        metrics = obs["candidate_path_metrics"]
        self.assertEqual(set(metrics), {"cache", "token", "image", "defer", "cache_update"})
        for path, data in metrics.items():
            self.assertIn("feasible", data)
            self.assertIn("accuracy_lcb", data)
            self.assertIn("accuracy_mean", data)
            self.assertIn("quality_gap", data)
            self.assertIn("payload_kb", data)
            self.assertIn("delay_s", data)
            self.assertIn("energy_j", data)
            self.assertIn("deadline_slack_s", data)
            self.assertIn("cache_eligible", data)
            self.assertIn("utm_constraint_violation", data)
            self.assertIn("deadline_feasible", data)
            self.assertIn("semantic_feasible", data)
            self.assertIn("energy_feasible", data)
            self.assertIn("utm_feasible", data)
            self.assertIn("joint_feasible", data)
            self.assertIn("tx_delay_s", data)
            self.assertIn("queue_delay_s", data)
            self.assertIn("infer_delay_s", data)
            self.assertIn("load_delay_s", data)
            self.assertIn("arrival_delay_s", data)
            self.assertIn("bottleneck_type", data)
            self.assertIn("required_deadline_reduction_s", data)
            self.assertIn("required_rate_mbps", data)
            self.assertIn("required_bandwidth_hz", data)
            self.assertIn("edge_queue_pressure", data)
            self.assertIn("model_cache_hit", data)
            self.assertEqual(data["feasible"], data["joint_feasible"])
            self.assertEqual(data["semantic_path"], path)

    def test_candidate_mobility_metrics_contains_path_mode_diagnostics(self) -> None:
        env = self._env()
        obs = env.reset(seed=5, options={"scenario": "utm_conflict"})
        metrics = obs["candidate_mobility_metrics"]
        self.assertIn("token", metrics)
        for mode in ("stay", "serve_task", "avoid_conflict", "reposition"):
            self.assertIn(mode, metrics["token"])
            data = metrics["token"][mode]
            for field in [
                "utm_feasible",
                "arrival_delay_s",
                "tx_delay_s",
                "total_delay_s",
                "deadline_slack_s",
                "semantic_feasible",
                "joint_feasible",
                "utm_conflict_risk",
            ]:
                self.assertIn(field, data)
        self.assertLessEqual(
            metrics["token"]["avoid_conflict"]["utm_conflict_risk"],
            metrics["token"]["serve_task"]["utm_conflict_risk"] + 1e-9,
        )

    def test_expired_task_paths_are_not_feasible_or_served(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env._front_task()
        self.assertIsNotNone(task)
        task.expired = True
        task.task_status = "expired"
        metrics = env.candidate_path_metrics(task)
        for path in ("cache", "token", "image", "cache_update"):
            self.assertFalse(metrics[path]["joint_feasible"])
        info = env.evaluate_action({"task_id": task.task_id, "semantic_path": "token"}, task_id=task.task_id)
        self.assertFalse(info["success"])
        self.assertTrue(info["expired"])
        self.assertEqual(info["task_status"], "expired")

    def test_defer_is_infeasible_when_deadline_would_expire(self) -> None:
        env = self._env()
        env.reset(seed=5)
        task = env._front_task()
        self.assertIsNotNone(task)
        task.tau_k = 0.5 * float(env.env_cfg["slot_s"])
        metrics = env.candidate_path_metrics(task)
        self.assertFalse(metrics["defer"]["deadline_feasible"])
        self.assertFalse(metrics["defer"]["joint_feasible"])

    def test_utm_conflict_paths_report_not_joint_feasible(self) -> None:
        env = self._env()
        env.reset(seed=5, options={"scenario": "utm_conflict"})
        task = next(
            item
            for item in env._active_tasks()
            if env.evaluate_action({"task_id": item.task_id, "semantic_path": "token"}, task_id=item.task_id)[
                "utm_conflict_violation"
            ]
        )
        metrics = env.candidate_path_metrics(task)
        self.assertFalse(metrics["token"]["utm_feasible"])
        self.assertFalse(metrics["token"]["joint_feasible"])
        self.assertFalse(metrics["cache_update"]["utm_feasible"])
        self.assertFalse(metrics["cache_update"]["joint_feasible"])

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
            "low_snr_soft",
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

    def test_low_snr_soft_is_distinct_from_hard_blockage(self) -> None:
        env = self._env()
        soft = env.reset(seed=5, options={"scenario": "low_snr_soft"})
        soft_cfg = dict(env.env_cfg)
        hard = env.reset(seed=5, options={"scenario": "low_snr_blockage"})
        hard_cfg = dict(env.env_cfg)
        self.assertEqual(soft["scenario"], "low_snr_soft")
        self.assertNotEqual(soft_cfg["area_spacing_m"], hard_cfg["area_spacing_m"])
        self.assertGreater(soft_cfg["bandwidth_hz"], hard_cfg["bandwidth_hz"])
        self.assertLess(soft_cfg["a2g"]["excess_loss_db"], hard_cfg["a2g"]["excess_loss_db"])
        self.assertLess(soft_cfg["a2g"]["nlos_excess_loss_db"], hard_cfg["a2g"]["nlos_excess_loss_db"])

    def test_normal_patrol_alias_maps_to_nominal_patrol(self) -> None:
        env = self._env()
        obs = env.reset(seed=5, options={"scenario": "normal_patrol"})
        self.assertEqual(obs["scenario"], "nominal_patrol")

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
        self.assertGreaterEqual(edge.edges[0].load, 0.50)
        self.assertGreaterEqual(edge.edges[0].gpu_load, 0.50)
        self.assertEqual(edge.env_cfg["model_cache_capacity"], 1)
        edge_infos = [
            edge.evaluate_action(edge.default_action(1), task_id=task.task_id)
            for task in edge.tasks
        ]
        self.assertTrue(edge_infos)
        self.assertGreaterEqual(sum(info["queue_delay_s"] >= 0.55 for info in edge_infos), 1)
        self.assertGreater(sum(not info["deadline_violation"] for info in edge_infos), 0)
        self.assertGreater(sum(info["semantic_success"] for info in edge_infos), 0)

        utm = self._env()
        utm.reset(seed=5, options={"scenario": "utm_conflict"})
        active = utm._active_tasks()
        self.assertGreaterEqual(len(active), 2)
        pair = next(
            (
                (first, second)
                for first in active
                for second in active
                if first.task_id != second.task_id and utm._area4d_overlaps_with_buffer(first.area4d, second.area4d)
            ),
            None,
        )
        self.assertIsNotNone(pair)
        first, second = pair
        info = utm.evaluate_action(
            {
                "task_id": first.task_id,
                "service_level": 1,
                "sensing_decision": "observe",
                "concurrent_actions": [
                    {"task_id": second.task_id, "service_level": 1, "sensing_decision": "observe"}
                ],
            },
            task_id=first.task_id,
        )
        self.assertTrue(info["utm_conflict_violation"])
        self.assertGreater(info["utm_delay_s"], 0.0)
        self.assertIn(info["airspace_state"], {"accepted", "activated", "nonconforming", "contingent"})

    def test_utm_conflict_background_intents_create_partial_algorithm_style_pressure(self) -> None:
        env = self._env()
        env.reset(seed=5, options={"scenario": "utm_conflict"})
        active = env._active_tasks()
        self.assertGreaterEqual(len(active), 2)
        observe_infos = [
            env.evaluate_action(
                {"task_id": task.task_id, "service_level": 1, "sensing_decision": "observe"},
                task_id=task.task_id,
            )
            for task in active
        ]
        cache_infos = [
            env.evaluate_action(
                {"task_id": task.task_id, "service_level": 0, "sensing_decision": "reuse_cache"},
                task_id=task.task_id,
            )
            for task in active
        ]
        conflict_count = sum(info["utm_conflict_violation"] for info in observe_infos)
        self.assertGreater(conflict_count, 0)
        self.assertLess(conflict_count, len(observe_infos))
        for info in observe_infos:
            self.assertIn("utm_constraint_violation", info)
            self.assertIn("semantic_quality_gap", info)
            self.assertIn("epsilon_k", info)
            self.assertIn("deadline_s", info)
        self.assertFalse(any(info["utm_conflict_violation"] for info in cache_infos))
        self.assertFalse(any(info["airspace_conflict"] for info in cache_infos))

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
