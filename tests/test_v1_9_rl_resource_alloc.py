from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.rl.v19_ppo import PPOServicePolicy, PPOTrainConfig, train_ppo
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
        self.assertEqual(set(obs["lyapunov_queues"]), {"quality", "deadline", "energy", "risk", "utm"})
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
        self.assertIn("delay_s", info)
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
            if policy.startswith("ppo_") or policy == "proposed_ppo":
                continue
            action = choose_baseline_action(policy, env, obs)
            for key in ["service_level", "bandwidth", "power", "cpu_share", "gpu_share", "uav_assignment"]:
                self.assertIn(key, action)
            if policy == "always_cache":
                self.assertLessEqual(float(action["bandwidth"]), 1.0)
                self.assertLessEqual(float(action["power"]), 1e-5)
                self.assertLessEqual(float(action["cpu_share"]), 0.01)
                self.assertLessEqual(float(action["gpu_share"]), 0.01)

    def test_tiny_ppo_training_runs(self) -> None:
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=2, tasks_per_episode=4)
        try:
            model, trace = train_ppo(
                env,
                PPOTrainConfig(train_episodes=2, update_epochs=1, hidden_size=32),
                seed=2,
            )
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        self.assertEqual(len(trace), 2)
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
        self.assertGreater(cfg.semantic_token_exploration_bonus, 0.0)


if __name__ == "__main__":
    unittest.main()
