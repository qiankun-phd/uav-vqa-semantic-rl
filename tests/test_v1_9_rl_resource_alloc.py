from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.rl.v19_ppo import (
    PPOServicePolicy,
    PPOTrainConfig,
    TwoTimescalePPOPolicy,
    _obs_tensor,
    _project_mobility_action,
    _project_semantic_feasible_action,
    _resource_floor_for_obs,
    _semantic_controller_reward,
    normalize_hidden_layers,
    resolve_torch_device,
    train_ppo,
    train_two_timescale_ppo,
)
from vqa_semcom.rl.v19_resource_env import V19LUTResourceEnv
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv
from run_v1_9_resource_alloc import SCENARIO_BENCHMARK_POLICIES, SCENARIO_BENCHMARK_SCENARIOS, choose_baseline_action


class V19RLResourceAllocTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs" / "v1_9_snr_lut.yaml")
        cls.lut = load_lut(ROOT / "outputs" / "lut" / "v1_9_snr_semantic_quality_lut.csv")
        tasks = read_csv(resolve_path(cls.cfg["paths"]["tasks_csv"]))
        cls.tasks = filter_tasks_supported_by_lut(tasks, cls.lut)

    def test_env_exposes_interface_contract_fields(self) -> None:
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=0, tasks_per_episode=3)
        obs = env.reset(seed=0)
        for key in [
            "task_type",
            "risk_level",
            "view_quality_bin",
            "freshness_bin",
            "sensed_snr_db",
            "snr_bin",
            "uav_state",
            "edge_load",
            "cache_state",
            "vector",
        ]:
            self.assertIn(key, obs)
        self.assertIn("lyapunov_queues", obs)
        self.assertEqual(set(obs["lyapunov_queues"]), {"quality", "deadline", "energy", "risk", "utm", "defer", "cache_stale"})
        action = env.candidate_action(1, obs)
        next_obs, reward, done, info = env.step(action)
        self.assertIn("answer_accuracy_est", info)
        self.assertIn("semantic_accuracy_mean", info)
        self.assertIn("semantic_accuracy_lcb", info)
        self.assertIn("semantic_uncertainty", info)
        self.assertIn("semantic_sample_count", info)
        self.assertIn("semantic_payload_kb", info)
        self.assertIn("semantic_quality_gap", info)
        self.assertIn("semantic_success", info)
        self.assertIn("epsilon_k", info)
        self.assertIn("epsilon_k", info["record"])
        self.assertGreater(float(info["epsilon_k"]), 0.0)
        self.assertGreaterEqual(float(info["semantic_quality_gap"]), 0.0)
        self.assertIn("q_quality", info)
        self.assertIn("q_deadline", info)
        self.assertIn("q_energy", info)
        self.assertIn("q_risk", info)
        self.assertIn("q_utm", info)
        self.assertIn("q_defer", info)
        self.assertIn("q_cache_stale", info)
        self.assertIn("semantic_path", info)
        self.assertIn("defer_count", info)
        self.assertIn("cache_eligible", info)
        self.assertIn("delay_s", info)
        self.assertIn("fly_delay_s", info)
        self.assertIn("sense_delay_s", info)
        self.assertIn("tx_delay_s", info)
        self.assertIn("queue_delay_s", info)
        self.assertIn("infer_delay_s", info)
        self.assertIn("load_delay_s", info)
        self.assertIn("deadline_token_cache_fallback", info)
        self.assertIn("energy_j", info)
        self.assertIn("payload_kb", info)
        self.assertIn("quality_violation", info)
        self.assertIn("deadline_violation", info)
        self.assertIn("battery_violation", info)
        self.assertIn("resource_violation", info)
        self.assertIn("airspace_conflict", info)
        self.assertIn("utm_constraint_violation", info)
        self.assertIn("utm_conflict_violation", info)
        self.assertIn("dss_delay_s", info)
        self.assertIn("subscription_notification_delay_s", info)
        self.assertIn("gpu_memory_ok", info)
        self.assertIn("service_level", info)
        self.assertIn("mobility_mode", info)
        self.assertIn("fly_distance_m", info)
        self.assertIn("coverage_gain", info)
        self.assertIn("mobility_energy_j", info)
        self.assertIn("arrival_delay_s", info)
        self.assertIn("utm_conflict_risk", info)
        self.assertIn("mobility_mode", info["record"])
        self.assertIn("mobility_energy_j", info["record"])
        self.assertEqual(info["bandwidth_unit"], "Hz")
        self.assertIsInstance(float(reward), float)
        self.assertFalse(done)
        self.assertEqual(next_obs["episode_step"], 1)
        self.assertIn("lyapunov_queues", next_obs)

    def test_action_contract_accepts_minimal_action(self) -> None:
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=1, tasks_per_episode=1)
        env.reset(seed=1)
        _obs, _reward, done, info = env.step(
            {
                "service_level": 0,
                "bandwidth": 1_000_000.0,
                "power": 0.1,
                "cpu_share": 0.0,
                "gpu_share": 0.0,
                "uav_assignment": 0,
                "waypoint": None,
            }
        )
        self.assertTrue(done)
        self.assertEqual(info["service_level"], 0)
        self.assertEqual(info["mobility_mode"], "stay")

    def test_wrapper_passes_formal_scenario_to_canonical_env(self) -> None:
        env = V19LUTResourceEnv(
            self.tasks,
            self.lut,
            self.cfg,
            seed=4,
            tasks_per_episode=2,
            formal_scenario="test_utm_dss_outage",
        )
        obs = env.reset(seed=4)
        self.assertEqual(obs["formal_scenario"], "test_utm_dss_outage")
        _obs, _reward, _done, info = env.step(env.candidate_action(1, obs))
        self.assertEqual(info["formal_scenario"], "test_utm_dss_outage")
        self.assertIn("utm_constraint_violation", info)

    def test_paper_scenarios_reset_through_rl_wrapper(self) -> None:
        for scenario in SCENARIO_BENCHMARK_SCENARIOS:
            cfg = dict(self.cfg)
            env_cfg = dict(cfg.get("multi_uav_env", {}))
            env_cfg["scenario"] = scenario
            cfg["multi_uav_env"] = env_cfg
            env = V19LUTResourceEnv(self.tasks, self.lut, cfg, seed=5, tasks_per_episode=1)
            obs = env.reset(seed=5)
            self.assertEqual(obs["scenario"], scenario)
            _obs, _reward, _done, info = env.step(env.candidate_action(1, obs))
            self.assertEqual(info["scenario"], scenario)
            self.assertIn("semantic_success", info)
            self.assertIn("utm_conflict_violation", info)

    def test_scenario_benchmark_baseline_aliases_emit_actions(self) -> None:
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=6, tasks_per_episode=1)
        obs = env.reset(seed=6)
        for policy in SCENARIO_BENCHMARK_POLICIES:
            if "ppo" in policy or policy in {"monolithic_ppo", "no_mobility_actor"} or policy.startswith("proposed_v2_"):
                continue
            action = choose_baseline_action(policy, env, obs)
            for key in ["service_level", "bandwidth", "power", "cpu_share", "gpu_share", "uav_assignment"]:
                self.assertIn(key, action)
            if policy == "always_cache":
                self.assertLessEqual(float(action["bandwidth"]), 1.0)
                self.assertLessEqual(float(action["power"]), 1e-5)
                self.assertLessEqual(float(action["cpu_share"]), 0.01)
                self.assertLessEqual(float(action["gpu_share"]), 0.01)

    def test_state_v2_extends_observation_vector(self) -> None:
        env_v1 = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=8, tasks_per_episode=2, state_version="v1")
        env_v2 = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=8, tasks_per_episode=2, state_version="v2")
        obs_v1 = env_v1.reset(seed=8)
        obs_v2 = env_v2.reset(seed=8)
        self.assertEqual(obs_v1["state_version"], "v1")
        self.assertEqual(obs_v2["state_version"], "v2")
        self.assertGreater(len(obs_v2["vector"]), len(obs_v1["vector"]))

    def test_state_v2_uses_fixed_canonical_dimension_across_scenarios(self) -> None:
        dims = set()
        for scenario in SCENARIO_BENCHMARK_SCENARIOS:
            cfg = dict(self.cfg)
            env_cfg = dict(cfg.get("multi_uav_env", {}))
            env_cfg["scenario"] = scenario
            cfg["multi_uav_env"] = env_cfg
            env = V19LUTResourceEnv(self.tasks, self.lut, cfg, seed=8, tasks_per_episode=2, state_version="v2")
            obs = env.reset(seed=8)
            dims.add(len(obs["vector"]))
        self.assertEqual(len(dims), 1)

    def test_obs_tensor_rejects_checkpoint_dimension_mismatch(self) -> None:
        try:
            device = resolve_torch_device("cpu")
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        tensor = _obs_tensor({"vector": [1.0, 2.0, 3.0]}, device, expected_dim=3)
        self.assertEqual(tuple(tensor.shape), (1, 3))
        with self.assertRaises(RuntimeError):
            _obs_tensor({"vector": [1.0, 2.0, 3.0]}, device, expected_dim=5)
        with self.assertRaises(RuntimeError):
            _obs_tensor({"vector": [1.0, 2.0, 3.0, 4.0]}, device, expected_dim=3)

    def test_hidden_layers_config_builds_custom_encoder(self) -> None:
        layers = normalize_hidden_layers("256,256,128", hidden_size=32)
        self.assertEqual(layers, (256, 256, 128))
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=9, tasks_per_episode=2)
        try:
            model, trace = train_ppo(
                env,
                PPOTrainConfig(train_episodes=1, update_epochs=1, hidden_size=32, hidden_layers=layers, device="cpu"),
                seed=9,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        self.assertEqual(len(trace), 1)
        self.assertEqual(tuple(model.hidden_layers), layers)
        self.assertEqual(model.encoder[0].out_features, 256)
        self.assertEqual(model.encoder[4].out_features, 128)

    def test_tiny_ppo_training_runs(self) -> None:
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=2, tasks_per_episode=4)
        try:
            model, trace = train_ppo(
                env,
                PPOTrainConfig(train_episodes=2, update_epochs=1, hidden_size=32, device="cpu"),
                seed=2,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        self.assertEqual(len(trace), 2)
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        self.assertIn("success_rate", trace[0])
        self.assertIn("lambda_quality", trace[0])
        self.assertIn("lambda_deadline", trace[0])
        self.assertIn("lambda_quality_normal", trace[0])
        self.assertIn("lambda_deadline_critical", trace[0])
        self.assertIn("mean_q_quality", trace[0])
        self.assertIn("mean_q_deadline", trace[0])
        self.assertIn("mean_semantic_quality_gap", trace[0])

        obs = env.reset(seed=22)
        action = PPOServicePolicy(env, model, PPOTrainConfig(hidden_size=32)).act(obs)
        for key in ["service_level", "bandwidth", "power", "cpu_share", "gpu_share", "uav_assignment"]:
            self.assertIn(key, action)
        self.assertGreaterEqual(float(action["bandwidth"]), 0.0)
        self.assertLessEqual(float(action["bandwidth"]), env.base_bandwidth_hz)
        self.assertGreaterEqual(float(action["power"]), 0.05)
        self.assertLessEqual(float(action["power"]), 1.0)
        self.assertGreaterEqual(float(action["cpu_share"]), 0.01)
        self.assertLessEqual(float(action["cpu_share"]), 1.0)
        self.assertGreaterEqual(float(action["gpu_share"]), 0.01)
        self.assertLessEqual(float(action["gpu_share"]), 1.0)

    def test_tiny_ppo_training_can_use_cuda_when_available(self) -> None:
        try:
            device = resolve_torch_device("cuda")
        except (ModuleNotFoundError, RuntimeError):
            self.skipTest("CUDA torch is not available")
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=17, tasks_per_episode=2)
        model, trace = train_ppo(
            env,
            PPOTrainConfig(train_episodes=1, update_epochs=1, hidden_size=32, device=str(device)),
            seed=17,
        )
        self.assertEqual(len(trace), 1)
        self.assertEqual(next(model.parameters()).device.type, "cuda")

    def test_tiny_two_timescale_ppo_training_runs(self) -> None:
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=7, tasks_per_episode=4)
        try:
            model, trace = train_two_timescale_ppo(
                env,
                PPOTrainConfig(
                    train_episodes=2,
                    update_epochs=1,
                    hidden_size=32,
                    semantic_reward_mode="semantic_utility",
                    lyapunov_reward=True,
                    mobility_update_interval=3,
                ),
                seed=7,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]["mobility_update_interval"], 3.0)
        self.assertIn("mean_flight_energy_j", trace[0])
        self.assertIn("mean_arrival_delay_s", trace[0])
        self.assertIn("mean_coverage_gain", trace[0])
        self.assertIn("mobility_reuse_ratio", trace[0])

        obs = env.reset(seed=27)
        action = TwoTimescalePPOPolicy(env, model, PPOTrainConfig(hidden_size=32, mobility_update_interval=3)).act(obs)
        for key in [
            "service_level",
            "bandwidth",
            "power",
            "cpu_share",
            "gpu_share",
            "uav_assignment",
            "mobility_mode",
            "waypoint_delta",
            "altitude_delta",
        ]:
            self.assertIn(key, action)
        self.assertIn(action["mobility_mode"], {"stay", "serve_task", "reposition", "avoid_conflict", "return_base"})
        self.assertEqual(len(action["waypoint_delta"]), 2)


    def test_semantic_path_two_timescale_ppo_outputs_path_action(self) -> None:
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=18, tasks_per_episode=3, state_version="v2")
        try:
            model, trace = train_two_timescale_ppo(
                env,
                PPOTrainConfig(
                    train_episodes=1,
                    update_epochs=1,
                    hidden_size=32,
                    semantic_reward_mode="semantic_utility",
                    lyapunov_reward=True,
                    semantic_path_actions=True,
                    mobility_update_interval=3,
                    device="cpu",
                ),
                seed=18,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        self.assertEqual(len(trace), 1)
        self.assertIn("path_defer_ratio", trace[0])
        obs = env.reset(seed=28)
        action = TwoTimescalePPOPolicy(
            env,
            model,
            PPOTrainConfig(hidden_size=32, semantic_path_actions=True, mobility_update_interval=3),
        ).act(obs)
        self.assertIn(action["semantic_path"], {"cache", "token", "image", "defer", "cache_update"})
        self.assertIn("service_level", action)

    def test_tiny_proposed_semantic_controller_runs(self) -> None:
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=3, tasks_per_episode=3)
        try:
            _model, trace = train_ppo(
                env,
                PPOTrainConfig(
                    train_episodes=1,
                    update_epochs=1,
                    hidden_size=32,
                    risk_aware_constraints=True,
                    semantic_reward_mode="semantic_utility",
                    lyapunov_reward=True,
                    imitation_warm_start=True,
                    demo_episodes=1,
                    bc_epochs=1,
                    bc_aux_weight=0.1,
                ),
                seed=3,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        self.assertEqual(len(trace), 1)
        self.assertGreater(trace[0]["demo_samples"], 0.0)
        self.assertIn("lambda_conflict", trace[0])
        self.assertIn("lambda_battery", trace[0])
        self.assertIn("lambda_gpu", trace[0])
        self.assertIn("entropy_weight", trace[0])
        self.assertIn("service_prior_weight", trace[0])
        self.assertIn("non_cache_ratio", trace[0])
        self.assertIn("mean_semantic_accuracy_lcb", trace[0])
        self.assertIn("mean_q_utm", trace[0])

    def test_semantic_reward_config_has_cache_collapse_controls(self) -> None:
        cfg = PPOTrainConfig(semantic_reward_mode="semantic_utility", lyapunov_reward=True)
        self.assertGreater(cfg.semantic_success_weight, 6.0)
        self.assertGreater(cfg.semantic_gap_weight, 2.0)
        self.assertGreater(cfg.cache_shortfall_penalty_weight, 0.0)
        self.assertGreater(cfg.high_risk_cache_penalty_weight, 0.0)
        self.assertGreater(cfg.cache_override_gap_threshold, 0.0)
        self.assertGreater(cfg.cache_override_min_improvement, 0.0)
        self.assertGreater(cfg.cache_stale_penalty_weight, 0.0)
        self.assertGreater(cfg.cache_utm_penalty_weight, 0.0)
        self.assertGreater(cfg.token_projection_bonus, 0.0)
        self.assertEqual(cfg.mobility_update_interval, 3)
        self.assertGreater(cfg.coverage_gain_weight, 0.0)
        self.assertGreater(cfg.flight_energy_cost_weight, 0.0)
        self.assertGreater(cfg.semantic_token_exploration_bonus, 0.0)
        self.assertGreater(cfg.deadline_guard_slack, 0.0)
        self.assertGreater(cfg.high_payload_guard_kb, 0.0)
        self.assertTrue(cfg.critical_burst_nearest_uav)
        self.assertGreater(cfg.deadline_slack_reward_weight, 0.0)
        self.assertGreater(cfg.deadline_overrun_penalty_weight, 0.0)
        self.assertGreater(cfg.token_fast_bandwidth_floor, cfg.semantic_token_bandwidth_floor)
        self.assertGreater(cfg.token_cache_fallback_overrun_ratio, 1.0)
        self.assertFalse(cfg.semantic_path_actions)
        self.assertGreater(cfg.defer_penalty_weight, 0.0)
        self.assertGreater(cfg.cache_update_success_bonus, 0.0)

    def test_deadline_slack_reward_penalizes_low_snr_overrun(self) -> None:
        obs = {"epsilon_k": 0.6, "deadline_s": 1.0, "sensed_snr_db": -9.0, "snr_bin": "low"}
        cfg = PPOTrainConfig(semantic_reward_mode="semantic_utility", deadline_slack_reward=True)
        base_info = {
            "risk_level": "normal",
            "semantic_accuracy_lcb": 0.8,
            "semantic_accuracy_mean": 0.8,
            "semantic_success": True,
            "success": True,
            "service_level": 1,
            "energy_j": 10.0,
            "payload_kb": 1.0,
        }
        fast = _semantic_controller_reward(obs, dict(base_info, delay_s=0.5), 0.0, cfg)
        slow = _semantic_controller_reward(obs, dict(base_info, delay_s=2.0, deadline_violation=True), 0.0, cfg)
        self.assertGreater(fast, slow)

    def test_token_fast_projection_raises_low_snr_resource_floors(self) -> None:
        obs = {"deadline_s": 2.0, "sensed_snr_db": -9.0, "snr_bin": "low"}
        cfg = PPOTrainConfig(token_fast_resource_projection=True)
        self.assertGreaterEqual(_resource_floor_for_obs(cfg, 1, "bandwidth", obs), cfg.token_fast_bandwidth_floor)
        self.assertGreaterEqual(_resource_floor_for_obs(cfg, 1, "power", obs), cfg.token_fast_power_floor)
        self.assertGreaterEqual(_resource_floor_for_obs(cfg, 1, "cpu_share", obs), cfg.token_fast_cpu_floor)
        self.assertGreaterEqual(_resource_floor_for_obs(cfg, 1, "gpu_share", obs), cfg.token_fast_gpu_floor)

    def test_deadline_token_cache_fallback_marks_cache_action(self) -> None:
        env = _ProjectionEnv()
        obs = {
            "epsilon_k": 0.4,
            "deadline_s": 0.8,
            "sensed_snr_db": -9.0,
            "snr_bin": "low",
            "action_mask": {"service_level_allowed": {0: True, 1: True, 2: True}},
            "uav_state": [{"uav_id": 0, "x_m": 0.0, "y_m": 0.0}],
        }
        projected = _project_semantic_feasible_action(
            env,
            obs,
            env.candidate_action(1, obs),
            PPOTrainConfig(deadline_token_cache_fallback=True, token_cache_fallback_gap_threshold=0.08),
        )
        self.assertEqual(projected["service_level"], 0)
        self.assertEqual(projected.get("sensing_decision"), "deadline_token_cache_fallback")

    def test_deadline_guard_prefers_token_over_slow_image(self) -> None:
        env = _ProjectionEnv()
        obs = {
            "epsilon_k": 0.8,
            "deadline_s": 2.0,
            "sensed_snr_db": -8.0,
            "snr_bin": "low",
            "action_mask": {"service_level_allowed": {0: True, 1: True, 2: True}},
            "uav_state": [{"uav_id": 0, "x_m": 0.0, "y_m": 0.0}],
        }
        current = env.candidate_action(2, obs)
        projected = _project_semantic_feasible_action(
            env,
            obs,
            current,
            PPOTrainConfig(deadline_aware_evidence_guard=True, payload_delay_aware_projection=True),
        )
        self.assertEqual(projected["service_level"], 1)

    def test_no_image_under_low_snr_blocks_image_projection(self) -> None:
        env = _ProjectionEnv()
        obs = {
            "epsilon_k": 0.8,
            "deadline_s": 5.0,
            "sensed_snr_db": -9.0,
            "snr_bin": "low",
            "action_mask": {"service_level_allowed": {0: True, 1: True, 2: True}},
            "uav_state": [{"uav_id": 0, "x_m": 0.0, "y_m": 0.0}],
        }
        projected = _project_semantic_feasible_action(
            env,
            obs,
            env.candidate_action(2, obs),
            PPOTrainConfig(no_image_under_low_snr=True),
        )
        self.assertEqual(projected["service_level"], 1)

    def test_nearest_uav_mobility_suppresses_burst_reposition(self) -> None:
        obs = {
            "scenario": "disaster_hotspot",
            "risk_level": "critical",
            "task_id": "t0",
            "task_queue": [{"task_id": "t0", "area4d": {"center_x_m": 10.0, "center_y_m": 0.0}}],
            "uav_state": [
                {"uav_id": 0, "x_m": 200.0, "y_m": 0.0},
                {"uav_id": 1, "x_m": 12.0, "y_m": 0.0},
            ],
            "action_mask": {"uav_battery_ok": {0: True, 1: True}, "mobility_mode_allowed": {"serve_task": True}},
        }
        action = _project_mobility_action(
            self,
            obs,
            {"uav_assignment": 0, "mobility_mode": "reposition", "waypoint_delta": [80.0, 40.0], "altitude_delta": 10.0},
            PPOTrainConfig(nearest_uav_mobility=True),
        )
        self.assertEqual(action["uav_assignment"], 1)
        self.assertEqual(action["mobility_mode"], "serve_task")
        self.assertEqual(action["waypoint_delta"], [0.0, 0.0])
        self.assertEqual(action["altitude_delta"], 0.0)


class _ProjectionEnv:
    service_levels = [0, 1, 2]

    def candidate_action(self, service_level: int, obs: dict) -> dict:
        return {
            "service_level": int(service_level),
            "bandwidth": 1.0,
            "power": 0.5,
            "cpu_share": 0.5,
            "gpu_share": 0.5,
            "uav_assignment": 0,
        }

    def parse_action(self, action: dict) -> dict:
        return dict(action)

    def evaluate_action(self, action: dict, obs: dict) -> dict:
        level = int(action["service_level"])
        base = {
            "service_level": level,
            "risk_level": obs.get("risk_level", "normal"),
            "semantic_uncertainty": 0.1,
            "battery_violation": False,
            "resource_violation": False,
            "utm_constraint_violation": False,
            "utm_conflict_violation": False,
            "off_nominal_planning_penalty": 0.0,
            "energy_j": 100.0,
        }
        if level == 0:
            base.update(
                semantic_accuracy_lcb=0.35,
                semantic_success=False,
                quality_violation=True,
                deadline_violation=False,
                delay_s=0.1,
                payload_kb=0.0,
                tx_delay_s=0.0,
                queue_delay_s=0.0,
                infer_delay_s=0.0,
            )
        elif level == 1:
            base.update(
                semantic_accuracy_lcb=0.74,
                semantic_success=False,
                quality_violation=True,
                deadline_violation=False,
                delay_s=1.3,
                payload_kb=1.0,
                tx_delay_s=0.4,
                queue_delay_s=0.4,
                infer_delay_s=0.3,
            )
        else:
            base.update(
                semantic_accuracy_lcb=0.95,
                semantic_success=True,
                quality_violation=False,
                deadline_violation=True,
                delay_s=3.2,
                payload_kb=80.0,
                semantic_payload_kb=80.0,
                tx_delay_s=1.4,
                queue_delay_s=0.9,
                infer_delay_s=0.8,
            )
        return base


if __name__ == "__main__":
    unittest.main()
