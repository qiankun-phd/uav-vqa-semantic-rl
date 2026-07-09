#!/usr/bin/env python
"""Paper II (deep-review P1): flat single-head PPO baseline.

The hybrid action space (discrete service level + 4 continuous resource dims)
is flattened into ONE categorical head: each resource dim is discretized to
``flat_resource_levels`` (default 3 sigmoid-space grid points), so the joint
head has ``num_services * 3**4`` logits.  Reward shaping, Lagrangian duals,
service masks, and the safety projection layer are shared bit-for-bit with the
hybrid path -- the only delta is the actor structure (Lya-HiPPO-style flat
baseline).  ``flat_ppo`` defaults to False so the legacy path is untouched.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from vqa_semcom.config import load_config, resolve_path  # noqa: E402
from vqa_semcom.rl.v19_ppo import (  # noqa: E402
    RESOURCE_KEYS,
    FlatActorCritic,
    FlatPPOPolicy,
    PPOTrainConfig,
    _apply_service_mask,
    _flat_decode,
    _flat_encode,
    _flat_joint_mask,
    _flat_nearest_levels,
    _flat_resource_values,
    load_ppo_policy,
    save_ppo_model,
    train_ppo,
)
from vqa_semcom.rl.v19_resource_env import V19LUTResourceEnv  # noqa: E402
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv  # noqa: E402

LEVELS = (0.15, 0.5, 0.85)


class FlatCodecTest(unittest.TestCase):
    def test_encode_decode_roundtrip_full_space(self) -> None:
        num_levels = len(LEVELS)
        num_services = 4
        grid = num_levels ** len(RESOURCE_KEYS)
        for joint in range(num_services * grid):
            service_idx, level_indices = _flat_decode(joint, num_levels)
            self.assertEqual(_flat_encode(service_idx, level_indices, num_levels), joint)
            self.assertTrue(0 <= service_idx < num_services)
            self.assertEqual(len(level_indices), len(RESOURCE_KEYS))
            for level in level_indices:
                self.assertTrue(0 <= level < num_levels)

    def test_resource_values_map_levels_to_grid(self) -> None:
        self.assertEqual(_flat_resource_values([0, 1, 2, 1], LEVELS), [0.15, 0.5, 0.85, 0.5])

    def test_nearest_levels_snap(self) -> None:
        self.assertEqual(_flat_nearest_levels([0.0, 0.49, 0.7, 1.0], LEVELS), [0, 1, 2, 2])

    def test_joint_mask_blocks_all_grid_cells_of_masked_service(self) -> None:
        grid = len(LEVELS) ** len(RESOURCE_KEYS)
        service_mask = torch.tensor([[True, False, True]])
        joint_mask = _flat_joint_mask(service_mask, grid)
        self.assertEqual(tuple(joint_mask.shape), (1, 3 * grid))
        logits = torch.zeros((1, 3 * grid))
        masked = _apply_service_mask(logits, joint_mask)
        blocked = masked[0, grid : 2 * grid]
        self.assertTrue(bool((blocked < -1e6).all()))
        self.assertTrue(bool((masked[0, :grid] == 0).all()))
        self.assertTrue(bool((masked[0, 2 * grid :] == 0).all()))

    def test_forward_shapes(self) -> None:
        model = FlatActorCritic(obs_dim=10, num_service_actions=4, hidden_size=16, resource_levels=LEVELS)
        joint_logits, value = model(torch.zeros((2, 10)))
        self.assertEqual(tuple(joint_logits.shape), (2, 4 * len(LEVELS) ** len(RESOURCE_KEYS)))
        self.assertEqual(tuple(value.shape), (2,))

    def test_config_defaults_are_legacy(self) -> None:
        cfg = PPOTrainConfig()
        self.assertFalse(cfg.flat_ppo)
        self.assertEqual(tuple(cfg.flat_resource_levels), LEVELS)


class FlatTrainSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs" / "v1_9_snr_lut.yaml")
        cls.lut = load_lut(ROOT / "outputs" / "lut" / "v1_9_snr_semantic_quality_lut.csv")
        tasks = read_csv(resolve_path(cls.cfg["paths"]["tasks_csv"]))
        cls.tasks = filter_tasks_supported_by_lut(tasks, cls.lut)

    def _train(self, **overrides: object):
        env = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=0, tasks_per_episode=3)
        base = dict(
            flat_ppo=True,
            train_episodes=2,
            device="cpu",
            hidden_size=16,
            semantic_reward_mode="semantic_utility",
            risk_aware_constraints=True,
            constrained=True,
        )
        base.update(overrides)
        cfg = PPOTrainConfig(**base)
        model, trace = train_ppo(env, cfg, seed=0)
        return env, cfg, model, trace

    def test_flat_training_smoke_and_policy_action(self) -> None:
        env, cfg, model, trace = self._train()
        self.assertIsInstance(model, FlatActorCritic)
        self.assertEqual(len(trace), 2)
        policy = FlatPPOPolicy(env, model, cfg)
        obs = env.reset(seed=1, options={"policy_name": "flat_test"})
        action = policy.act(obs, deterministic=True)
        self.assertIn("service_level", action)
        for key in RESOURCE_KEYS:
            self.assertIn(key, action)

    def test_flat_training_with_bc_warm_start(self) -> None:
        env, cfg, model, trace = self._train(
            imitation_warm_start=True,
            demo_policy="semantic_greedy",
            demo_episodes=1,
            bc_epochs=1,
        )
        self.assertIsInstance(model, FlatActorCritic)
        self.assertGreaterEqual(float(trace[0].get("demo_samples", 0.0)), 1.0)

    def test_flat_save_load_roundtrip(self) -> None:
        import tempfile

        env, cfg, model, _trace = self._train()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ppo_flat_policy.pt"
            save_ppo_model(path, model, env, cfg)
            policy = load_ppo_policy(path, env, hidden_size=cfg.hidden_size, device="cpu")
            self.assertIsInstance(policy, FlatPPOPolicy)
            obs = env.reset(seed=2, options={"policy_name": "flat_load_test"})
            action = policy.act(obs, deterministic=True)
            self.assertIn("service_level", action)


if __name__ == "__main__":
    unittest.main()
