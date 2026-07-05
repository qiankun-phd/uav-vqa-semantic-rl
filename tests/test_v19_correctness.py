"""Correctness gate for the v19 P0/P1 RL fixes (2026-07 design review).

Gate (a): epoch-0 PPO ratio must be identically 1 -- the joint log-prob
recomputed by the update-end code path must match the rollout's stored
old_log_probs element-wise, for the two-timescale policy (slow-head credit,
mask replay, decision masking).

Gate (b): the single-timescale and two-timescale collectors must assign the
same per-step semantic reward to the same (obs, info) under semantic reward
modes (mobility-cost double-count fix); only the dual penalty may differ.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.rl.v19_ppo import (
    DualState,
    PPOTrainConfig,
    TwoTimescaleMobilitySemanticActorCritic,
    _collect_two_timescale_episode,
    _dual_update,
    _init_dual_state,
    _mobility_reward_adjustment,
    _normalize_advantages,
    _scheduled_bc_aux_weight,
    _semantic_controller_reward,
    _two_timescale_log_probs_entropy_values,
    _two_timescale_rollout_tensors,
    _two_timescale_step_semantic_reward,
    _update_duals,
)
from vqa_semcom.rl.v19_resource_env import V19LUTResourceEnv
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv


def _proposed_two_timescale_cfg(**overrides: object) -> PPOTrainConfig:
    base = dict(
        two_timescale=True,
        semantic_reward_mode="semantic_utility",
        risk_aware_constraints=True,
        lyapunov_reward=True,
        constrained=True,
        mobility_update_interval=3,
        device="cpu",
    )
    base.update(overrides)
    return PPOTrainConfig(**base)


class V19CorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg_yaml = load_config(ROOT / "configs" / "v1_9_bubbles.yaml")
        cls.lut = load_lut(ROOT / "outputs" / "lut" / "v1_9_snr_semantic_quality_lut.csv")
        tasks = read_csv(resolve_path(cls.cfg_yaml["paths"]["tasks_csv"]))
        cls.tasks = filter_tasks_supported_by_lut(tasks, cls.lut)

    def _make_env(self, seed: int = 0, tasks_per_episode: int = 6) -> V19LUTResourceEnv:
        return V19LUTResourceEnv(self.tasks, self.lut, self.cfg_yaml, seed=seed, tasks_per_episode=tasks_per_episode)

    def _make_model(self, env: V19LUTResourceEnv, seed: int = 0) -> TwoTimescaleMobilitySemanticActorCritic:
        torch.manual_seed(seed)
        obs = env.reset(seed=seed)
        num_uavs = int(env.action_spec().get("num_uavs", max(1, len(obs.get("uav_state", []) or []))))
        return TwoTimescaleMobilitySemanticActorCritic(len(obs["vector"]), len(env.service_levels), num_uavs)

    def test_epoch0_ratio_is_one_for_two_timescale_rollout(self) -> None:
        """Gate (a): update-end joint log-prob == rollout old_log_probs pre-update."""
        cfg = _proposed_two_timescale_cfg()
        env = self._make_env(seed=3)
        model = self._make_model(env, seed=3)
        dual = DualState(quality_normal=0.4, quality_critical=0.9, deadline_normal=0.3, conflict=1.5)
        rollout = _collect_two_timescale_episode(env, model, seed=3, cfg=cfg, dual=dual)
        self.assertGreater(len(rollout["old_log_probs"]), 3)
        tensors = _two_timescale_rollout_tensors(rollout, torch.device("cpu"))
        with torch.no_grad():
            log_probs, _entropy, _values = _two_timescale_log_probs_entropy_values(model, tensors, cfg)
        old = tensors["old_log_probs"]
        self.assertTrue(
            torch.allclose(log_probs, old, atol=1e-4, rtol=1e-4),
            f"epoch-0 log-prob mismatch: max abs diff {float((log_probs - old).abs().max())}",
        )
        ratio = torch.exp(log_probs - old)
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3))

    def test_slow_head_decision_bookkeeping(self) -> None:
        """Cached slow steps store zero mobility log-prob; decisions follow K."""
        cfg = _proposed_two_timescale_cfg()
        env = self._make_env(seed=5)
        model = self._make_model(env, seed=5)
        rollout = _collect_two_timescale_episode(env, model, seed=5, cfg=cfg, dual=DualState())
        decisions = rollout["mobility_decisions"]
        self.assertGreaterEqual(sum(decisions), 1.0)
        self.assertLess(sum(decisions), len(decisions))  # some steps must reuse the cache
        for decision, log_prob, reused in zip(decisions, rollout["mobility_old_log_probs"], rollout["mobility_reused"]):
            if decision == 0.0:
                self.assertEqual(log_prob, 0.0)
                self.assertEqual(reused, 1.0)
            else:
                self.assertEqual(reused, 0.0)

    def test_two_timescale_reward_matches_single_path(self) -> None:
        """Gate (b): same (obs, info) -> same semantic reward on both paths."""
        env = self._make_env(seed=7)
        obs = env.reset(seed=7)
        action = env.candidate_action(1, obs)
        _next_obs, raw_reward, _done, info = env.step(action)
        for mode in ("semantic_utility", "uncertainty_aware", "accuracy_only", "no_semantic_utility"):
            cfg = _proposed_two_timescale_cfg(semantic_reward_mode=mode)
            single = _semantic_controller_reward(obs, info, raw_reward, cfg)
            double = _two_timescale_step_semantic_reward(obs, info, raw_reward, cfg)
            self.assertAlmostEqual(single, double, places=9, msg=f"mode={mode} double-counts mobility")
        env_cfg = _proposed_two_timescale_cfg(semantic_reward_mode="env")
        self.assertAlmostEqual(
            _two_timescale_step_semantic_reward(obs, info, raw_reward, env_cfg),
            _semantic_controller_reward(obs, info, raw_reward, env_cfg) + _mobility_reward_adjustment(info, env_cfg),
            places=9,
        )

    def test_lyapunov_uses_risk_utm_increments(self) -> None:
        """The queue penalty must charge conflict events once, not per water level.

        v3 calibration zeroes queue_risk_weight/queue_utm_weight by default, so
        this test pins them back to 1.0 to keep the increment-vs-level check
        meaningful.
        """
        cfg = _proposed_two_timescale_cfg(queue_risk_weight=1.0, queue_utm_weight=1.0)
        obs = {"epsilon_k": 0.9, "deadline_s": 5.0}
        base_info = {
            "risk_level": "normal",
            "semantic_accuracy_lcb": 0.5,
            "success": False,
            "service_level": 1,
            "delay_s": 1.0,
            "energy_j": 10.0,
            "q_risk": 50.0,
            "q_utm": 50.0,
            "q_risk_increment": 0.0,
            "q_utm_increment": 0.0,
        }
        bumped = dict(base_info, q_risk_increment=1.0, q_utm_increment=1.0)
        no_event = _semantic_controller_reward(obs, base_info, 0.0, cfg)
        with_event = _semantic_controller_reward(obs, bumped, 0.0, cfg)
        expected_delta = cfg.queue_risk_weight * 1.0 + cfg.queue_utm_weight * 1.0
        self.assertAlmostEqual(no_event - with_event, expected_delta, places=6)
        # Raising only the water level must not change the reward any more.
        high_level = dict(base_info, q_risk=500.0, q_utm=500.0)
        self.assertAlmostEqual(no_event, _semantic_controller_reward(obs, high_level, 0.0, cfg), places=6)

    def test_dual_update_leaks_and_respects_channel_cap(self) -> None:
        cfg = PPOTrainConfig()
        # Cost below limit: lambda must decrease (no one-way ratchet).
        lowered = _dual_update(5.0, observed_cost=0.0, limit=cfg.conflict_cost_limit, cfg=cfg)
        self.assertLess(lowered, 5.0)
        # Cost above limit: bounded by the per-channel ceiling.
        lam = 0.0
        for _ in range(500):
            lam = _dual_update(lam, observed_cost=1.0, limit=cfg.conflict_cost_limit, cfg=cfg, lambda_max=cfg.lambda_max_conflict)
        self.assertLessEqual(lam, cfg.lambda_max_conflict)
        self.assertGreater(lam, 0.9 * cfg.lambda_max_conflict)

    def test_conflict_dual_is_sole_load_bearing_channel(self) -> None:
        """v3 calibration: only -lambda_conflict*violation carries conflict cost.

        Defaults must zero every shaped/queue conflict path, and the conflict
        dual channel must ascend with its dedicated lambda_lr_conflict rate.
        """
        cfg = PPOTrainConfig()
        self.assertEqual(cfg.conflict_cost_weight, 0.0)
        self.assertEqual(cfg.utm_conflict_cost_weight, 0.0)
        self.assertEqual(cfg.queue_risk_weight, 0.0)
        self.assertEqual(cfg.queue_utm_weight, 0.0)
        self.assertEqual(cfg.lambda_lr_conflict, 0.2)
        # Wiring: _update_duals must use lambda_lr_conflict for the conflict channel.
        cfg2 = _proposed_two_timescale_cfg()
        dual = DualState()
        rollout = {
            "quality_costs": [0.0],
            "deadline_costs": [0.0],
            "quality_costs_normal": [0.0],
            "quality_costs_critical": [0.0],
            "deadline_costs_normal": [0.0],
            "deadline_costs_critical": [0.0],
            "conflict_costs": [1.0],
            "battery_costs": [0.0],
            "gpu_costs": [0.0],
        }
        _update_duals(dual, rollout, cfg2)
        self.assertAlmostEqual(
            dual.conflict,
            cfg2.lambda_lr_conflict * (1.0 - cfg2.conflict_cost_limit),
            places=9,
        )

    def test_lambda_init_conflict_warm_start(self) -> None:
        """lambda_init_conflict warm-starts the conflict dual at episode 0.

        Dual-latency probe (2026-07): with init=0 the conflict lambda needs
        hundreds of episodes to reach its ~4 equilibrium, arriving after the
        BC/high-entropy policy-formation phase.  The warm start must (a) seed
        DualState.conflict with the requested value, (b) stay clamped by the
        per-channel ceiling so lambda_max_conflict=0 still disables the
        channel, and (c) remain governed by the leaky dual update -- it can
        rise under violation pressure and fall once cost < limit.
        """
        # Default keeps the existing cold-start behavior.
        self.assertEqual(PPOTrainConfig().lambda_init_conflict, 0.0)
        self.assertEqual(_init_dual_state(PPOTrainConfig()).conflict, 0.0)
        # (a) warm start seeds the conflict channel; other channels stay 0.
        cfg = PPOTrainConfig(lambda_init_conflict=4.0)
        dual = _init_dual_state(cfg)
        self.assertEqual(dual.conflict, 4.0)
        self.assertEqual(dual.quality_normal, 0.0)
        self.assertEqual(dual.battery, 0.0)
        # (b) clamped to the per-channel ceiling; disabled channel stays off.
        capped = _init_dual_state(PPOTrainConfig(lambda_init_conflict=99.0))
        self.assertEqual(capped.conflict, PPOTrainConfig().lambda_max_conflict)
        disabled = _init_dual_state(PPOTrainConfig(lambda_init_conflict=4.0, lambda_max_conflict=0.0))
        self.assertEqual(disabled.conflict, 0.0)
        # (c) the leaky update still moves the warm-started lambda both ways.
        risen = _dual_update(
            dual.conflict, observed_cost=1.0, limit=cfg.conflict_cost_limit,
            cfg=cfg, lambda_max=cfg.lambda_max_conflict, lambda_lr=cfg.lambda_lr_conflict,
        )
        self.assertGreater(risen, 4.0)
        fallen = _dual_update(
            dual.conflict, observed_cost=0.0, limit=cfg.conflict_cost_limit,
            cfg=cfg, lambda_max=cfg.lambda_max_conflict, lambda_lr=cfg.lambda_lr_conflict,
        )
        self.assertLess(fallen, 4.0)

    def test_advantage_normalization_keeps_constant_penalty(self) -> None:
        normalized = _normalize_advantages([-3.0, -2.0, -4.0, -2.5])
        self.assertTrue(all(value < 0.0 for value in normalized))

    def test_bc_aux_weight_decays_to_zero(self) -> None:
        cfg = PPOTrainConfig(bc_aux_weight=0.05, service_prior_decay_episodes=360)
        self.assertAlmostEqual(_scheduled_bc_aux_weight(cfg, 0), 0.05)
        self.assertLess(_scheduled_bc_aux_weight(cfg, 180), 0.05)
        self.assertEqual(_scheduled_bc_aux_weight(cfg, 360), 0.0)
        self.assertEqual(_scheduled_bc_aux_weight(cfg, 5000), 0.0)


if __name__ == "__main__":
    unittest.main()
