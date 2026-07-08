"""W3c/E4: pluggable quality backend (lut | persample) for the V1.9 env."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.rl.v19_resource_env import DEFAULT_PERSAMPLE_MODEL, V19LUTResourceEnv
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv

MODEL_PATH = ROOT / DEFAULT_PERSAMPLE_MODEL


class QualityBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs" / "v1_9_snr_lut.yaml")
        cls.lut = load_lut(ROOT / "outputs" / "lut" / "v1_9_snr_semantic_quality_lut.csv")
        tasks = read_csv(resolve_path(cls.cfg["paths"]["tasks_csv"]))
        cls.tasks = filter_tasks_supported_by_lut(tasks, cls.lut)

    def _env(self, backend: str | None) -> V19LUTResourceEnv:
        return V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=0,
                                 tasks_per_episode=3, quality_backend=backend)

    def test_default_backend_is_lut(self) -> None:
        env = self._env(None)
        self.assertEqual(env.quality_backend, "lut")
        self.assertIsNone(env._persample)
        obs = env.reset(seed=0)
        _, _, _, info = env.step(env.candidate_action(1, obs))
        self.assertNotIn("quality_backend", info)

    def test_unknown_backend_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._env("oracle")

    @unittest.skipUnless(MODEL_PATH.exists(), "persample model artefact missing")
    def test_persample_backend_overrides_transmit_quality(self) -> None:
        env = self._env("persample")
        self.assertEqual(env.quality_backend, "persample")
        obs = env.reset(seed=0)
        _, _, _, info = env.step(env.candidate_action(1, obs))
        self.assertEqual(info.get("quality_backend"), "persample")
        mean = float(info["semantic_accuracy_mean"])
        lcb = float(info["semantic_accuracy_lcb"])
        unc = float(info["semantic_uncertainty"])
        self.assertGreaterEqual(mean, 0.0)
        self.assertLessEqual(mean, 1.0)
        self.assertLessEqual(lcb, mean + 1e-9)
        self.assertGreater(unc, 0.0)
        self.assertAlmostEqual(lcb, max(0.0, min(1.0, mean - unc)), places=9)
        self.assertEqual(float(info["answer_accuracy_est"]), lcb)

    @unittest.skipUnless(MODEL_PATH.exists(), "persample model artefact missing")
    def test_persample_backend_leaves_cache_path_untouched(self) -> None:
        env_lut = self._env(None)
        env_ps = self._env("persample")
        obs_l = env_lut.reset(seed=0)
        obs_p = env_ps.reset(seed=0)
        info_l = env_lut.evaluate_action(env_lut.candidate_action(0, obs_l), obs_l)
        info_p = env_ps.evaluate_action(env_ps.candidate_action(0, obs_p), obs_p)
        self.assertNotIn("quality_backend", info_p)
        for key in ("semantic_accuracy_mean", "semantic_accuracy_lcb",
                    "semantic_uncertainty", "cache_eligible", "cache_quality_lcb"):
            self.assertEqual(info_l.get(key), info_p.get(key), key)

    @unittest.skipUnless(MODEL_PATH.exists(), "persample model artefact missing")
    def test_persample_state_features_differ_from_lut(self) -> None:
        env_lut = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=0,
                                    tasks_per_episode=3, state_version="v2")
        env_ps = V19LUTResourceEnv(self.tasks, self.lut, self.cfg, seed=0,
                                   tasks_per_episode=3, state_version="v2",
                                   quality_backend="persample")
        vec_l = env_lut.reset(seed=0)["vector"]
        vec_p = env_ps.reset(seed=0)["vector"]
        self.assertEqual(len(vec_l), len(vec_p))
        self.assertNotEqual(vec_l, vec_p)


if __name__ == "__main__":
    unittest.main()
