from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.detector.visdrone_yolo import DetectionRecord, degrade_detections_for_channel
from vqa_semcom.snr import (
    channel_bin_from_snr,
    degradation_config,
    parse_snr_bins,
    snr_bin_label,
    snr_db_to_bin_label,
    snr_db_from_label,
)
from vqa_semcom.sim.resource_env import load_lut, run_simulation
from vqa_semcom.vlm.lut import build_lut_from_predictions, write_lut_csv


class V19SNRLUTTest(unittest.TestCase):
    def test_sensed_snr_maps_to_discrete_bin_and_legacy_channel(self) -> None:
        bins = parse_snr_bins([-5, 0, 5, 10, 15, 20])
        self.assertEqual(snr_db_to_bin_label(11.2, bins), "10dB")
        self.assertEqual(snr_bin_label(-5), "-5dB")
        self.assertEqual(snr_db_from_label("15dB"), 15.0)
        self.assertEqual(channel_bin_from_snr(-1.0), "bad")
        self.assertEqual(channel_bin_from_snr(10.0), "medium")
        self.assertEqual(channel_bin_from_snr(20.0), "good")

    def test_low_snr_degradation_is_stronger_than_high_snr(self) -> None:
        cfg = {"vlm": {}}
        low_light = degradation_config("light", "-5dB", cfg)
        high_light = degradation_config("light", "20dB", cfg)
        self.assertGreater(low_light["drop_rate"], high_light["drop_rate"])
        self.assertGreater(low_light["bbox_quantization"], high_light["bbox_quantization"])
        self.assertGreater(low_light["confidence_threshold"], high_light["confidence_threshold"])
        low_image = degradation_config("image", "-5dB", cfg)
        high_image = degradation_config("image", "20dB", cfg)
        self.assertLess(low_image["jpeg_quality"], high_image["jpeg_quality"])
        self.assertLess(low_image["resize_scale"], high_image["resize_scale"])

    def test_detector_semantic_tokens_degrade_with_snr(self) -> None:
        records = [
            DetectionRecord("car", 10, 10, 20, 20, 0.20),
            DetectionRecord("car", 50, 50, 20, 20, 0.92),
        ]
        cfg = {"vlm": {}}
        low = degrade_detections_for_channel(records, "-5dB", cfg, "demo")
        high = degrade_detections_for_channel(records, "20dB", cfg, "demo")
        self.assertLessEqual(len(low), len(high))

    def test_vlm_lut_groups_by_snr_bin(self) -> None:
        rows = [
            {
                "question_type": "presence",
                "service_level": "1",
                "channel_bin": "bad",
                "snr_bin": "0dB",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
                "risk_level": "normal",
                "correct": "False",
                "payload_bytes": "100",
            },
            {
                "question_type": "presence",
                "service_level": "1",
                "channel_bin": "good",
                "snr_bin": "20dB",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
                "risk_level": "normal",
                "correct": "True",
                "payload_bytes": "100",
            },
        ]
        lut = build_lut_from_predictions(rows)
        self.assertEqual({row.snr_bin for row in lut}, {"0dB", "20dB"})
        self.assertEqual(len(lut), 2)

    def test_resource_sim_uses_continuous_snr_for_delay_and_discrete_snr_for_lut(self) -> None:
        task = {
            "question_type": "presence",
            "view_quality_bin": "good",
            "risk_level": "normal",
            "freshness_bin": "fresh",
            "epsilon_k": "0.7",
            "tau_k": "10.0",
        }
        predictions = []
        for service_level in [0, 1, 2]:
            for snr_bin in ["0dB", "10dB", "20dB"]:
                predictions.append(
                    {
                        "question_type": "presence",
                        "service_level": str(service_level),
                        "channel_bin": channel_bin_from_snr(snr_db_from_label(snr_bin)),
                        "snr_bin": snr_bin,
                        "view_quality_bin": "good",
                        "freshness_bin": "fresh",
                        "risk_level": "normal",
                        "correct": "True" if service_level == 2 or snr_bin == "20dB" else "False",
                        "payload_bytes": str({0: 0, 1: 256, 2: 2000}[service_level]),
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = Path(tmp) / "lut.csv"
            write_lut_csv(build_lut_from_predictions(predictions), lut_path)
            lut = load_lut(lut_path)
            cfg = {
                "bins": {
                    "freshness": ["fresh"],
                    "channel": ["bad", "medium", "good"],
                    "snr_db": [0, 10, 20],
                },
                "simulation": {
                    "seed": 3,
                    "tasks_per_episode": 5,
                    "policies": ["always_light", "always_image", "oracle_best_feasible_evidence"],
                    "delay_by_level": {"0": 0.1, "1": 0.2, "2": 0.3},
                    "energy_by_level": {"0": 0.1, "1": 0.2, "2": 0.3},
                    "channel_delay_multiplier": {"bad": 1.0, "medium": 1.0, "good": 1.0},
                    "service_level_order": [0, 1, 2],
                    "use_snr_rate_model": True,
                    "bandwidth_hz": 1000000,
                    "sensed_snr_db_values": [20],
                    "processing_delay_by_level": {"0": 0.0, "1": 0.1, "2": 0.2},
                },
            }
            results = run_simulation([task], lut, cfg, episodes=2)
        by_policy = {result.policy: result for result in results}
        self.assertGreater(by_policy["always_light"].task_success_rate, 0.0)
        self.assertLess(by_policy["always_light"].average_delay, by_policy["always_image"].average_delay)
        self.assertGreaterEqual(by_policy["oracle_best_feasible_evidence"].task_success_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
