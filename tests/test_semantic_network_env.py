from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.sim.multi_uav_env import (
    MultiUAVVQAEnv,
    available_formal_scenarios,
    formal_scenario_specs_markdown,
    scalability_presets,
)
from vqa_semcom.sim.resource_env import LUTEntry


class SemanticNetworkEnvTest(unittest.TestCase):
    def _env(self) -> MultiUAVVQAEnv:
        tasks = [
            {
                "question_type": "presence",
                "question": "Are there vehicles?",
                "risk_level": "normal",
                "epsilon_k": "0.5",
                "tau_k": "20.0",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
            },
            {
                "question_type": "presence",
                "question": "Are there blocked roads?",
                "risk_level": "critical",
                "epsilon_k": "0.6",
                "tau_k": "20.0",
                "view_quality_bin": "medium",
                "freshness_bin": "stale",
            },
        ]
        lut = {}
        for risk in ("normal", "critical"):
            for view in ("poor", "medium", "good"):
                for fresh in ("fresh", "stale", "expired"):
                    for snr in ("0dB", "10dB", "20dB"):
                        lut[("presence", 0, snr, view, fresh, risk)] = LUTEntry(0.45, 0.0)
                        lut[("presence", 1, snr, view, fresh, risk)] = LUTEntry(0.70, 2048.0)
                        lut[("presence", 2, snr, view, fresh, risk)] = LUTEntry(0.90, 300000.0)
        cfg = {
            "bins": {"snr_db": [0, 10, 20], "freshness": ["fresh", "stale", "expired"], "service_levels": [0, 1, 2]},
            "simulation": {"seed": 11, "bandwidth_hz": 1_000_000},
            "multi_uav_env": {
                "scenario": "nominal",
                "enabled_service_levels": [0, 1, 2],
                "enable_service_level_3": False,
                "num_uavs": 2,
                "num_edges": 1,
                "tasks_per_episode": 4,
                "episode_steps": 4,
                "area_spacing_m": 100.0,
            },
        }
        return MultiUAVVQAEnv(tasks, lut, cfg, seed=11)

    def test_formal_scenarios_are_registered(self) -> None:
        expected = {
            "train_nominal",
            "train_mixed_random",
            "test_conflict_heavy",
            "test_interference_heavy",
            "test_cache_heavy",
            "test_mobility_stress",
            "test_unseen_mixed",
        }
        self.assertEqual(set(available_formal_scenarios()), expected)

    def test_semantic_routing_and_utility_are_in_info(self) -> None:
        env = self._env()
        obs = env.reset(seed=11, options={"formal_scenario": "train_nominal"})
        self.assertEqual(obs["formal_scenario"], "train_nominal")
        self.assertIn("semantic_utility_layer", obs["network_layers"])
        _obs, _reward, _done, info = env.step({"service_level": 1, "bandwidth": 1_000_000.0})
        self.assertEqual(info["semantic_service_name"], "semantic_tokens")
        self.assertIn("semantic_evidence_type", info)
        self.assertIn("semantic_utility", info)
        self.assertIn("semantic_efficiency", info)

    def test_graph_observation_schema_exports_nodes_and_edges(self) -> None:
        env = self._env()
        obs = env.reset(seed=11, options={"formal_scenario": "train_nominal"})
        graph = obs["graph"]
        self.assertEqual(graph["schema_version"], "semantic_network_graph_v1")
        self.assertIn("uav", graph["node_sets"])
        self.assertIn("task", graph["node_sets"])
        self.assertIn("edge", graph["node_sets"])
        self.assertGreaterEqual(len(graph["node_sets"]["uav"]), 2)
        self.assertGreaterEqual(len(graph["edge_sets"]["uav_task_link"]), len(graph["node_sets"]["uav"]))
        self.assertIn("uav_task_link", env.graph_observation_schema()["edge_sets"])

    def test_scalability_options_change_network_size_and_load(self) -> None:
        env = self._env()
        obs = env.reset(
            seed=11,
            options={
                "formal_scenario": "train_nominal",
                "num_uavs": "M8",
                "task_arrival": "high",
                "edge_load": "heavy",
            },
        )
        self.assertEqual(len(env.uavs), 8)
        self.assertEqual(len(env.tasks), 40)
        self.assertEqual(obs["scalability_profile"]["uav_count"], "M8")
        self.assertGreaterEqual(env.edges[0].load, 0.45)
        self.assertIn("M6", scalability_presets()["uav_count"])

    def test_roi_service_level_remains_disabled(self) -> None:
        env = self._env()
        env.reset(seed=11, options={"formal_scenario": "test_unseen_mixed"})
        self.assertEqual(env.service_levels(), [0, 1, 2])
        self.assertNotIn(3, env.action_mask()["service_level_allowed"])

    def test_formal_specs_markdown_mentions_all_splits(self) -> None:
        md = formal_scenario_specs_markdown()
        self.assertIn("train_nominal", md)
        self.assertIn("test_unseen_mixed", md)
        self.assertIn("M8", md)


if __name__ == "__main__":
    unittest.main()
