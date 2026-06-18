from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.report_v1_8_protocols import collect_quality_rows
from scripts.sweep_v1_8_detector_thresholds import summarize_threshold_rows
from vqa_semcom.config import load_config


class V18ProtocolsTest(unittest.TestCase):
    def test_threshold_summary_keeps_presence_polarity_and_payload(self) -> None:
        rows = [
            {
                "threshold": "0.10",
                "question_type": "presence",
                "presence_polarity": "positive",
                "channel_bin": "good",
                "object_count": "1",
                "target_class": "car",
                "correct": "True",
                "raw_correct": "True",
                "payload_bytes": "1024",
                "transmitted_detector_count": "1",
                "calibrated_detector_count": "1",
            },
            {
                "threshold": "0.10",
                "question_type": "presence",
                "presence_polarity": "negative",
                "channel_bin": "good",
                "object_count": "0",
                "target_class": "bus",
                "correct": "False",
                "raw_correct": "False",
                "payload_bytes": "2048",
                "transmitted_detector_count": "1",
                "calibrated_detector_count": "1",
            },
            {
                "threshold": "0.10",
                "question_type": "counting",
                "presence_polarity": "",
                "channel_bin": "good",
                "object_count": "4",
                "target_class": "car",
                "correct": "True",
                "raw_correct": "False",
                "payload_bytes": "3072",
                "transmitted_detector_count": "2",
                "calibrated_detector_count": "4",
            },
        ]
        summary = summarize_threshold_rows(rows)
        by_key = {(row["group"], row["value"]): row for row in summary}
        self.assertEqual(by_key[("presence_polarity", "positive")]["accuracy"], "1.000000")
        self.assertEqual(by_key[("presence_polarity", "negative")]["accuracy"], "0.000000")
        self.assertEqual(by_key[("question_type", "counting")]["raw_accuracy"], "0.000000")
        self.assertGreater(float(by_key[("overall", "all")]["mean_payload_kb"]), 0.0)

    def test_collect_quality_rows_reads_protocol_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            fields = ["service_level", "question_type", "presence_polarity", "object_count", "correct", "payload_bytes"]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "service_level": "1",
                        "question_type": "presence",
                        "presence_polarity": "positive",
                        "object_count": "1",
                        "correct": "True",
                        "payload_bytes": "1000",
                    }
                )
            rows = collect_quality_rows([("Protocol-X", path)])
            self.assertTrue(rows)
            self.assertTrue(all(row["protocol"] == "Protocol-X" for row in rows))
            self.assertIn("service_level", {row["group"] for row in rows})

    def test_v1_8_roi_config_keeps_full_image_and_adds_roi(self) -> None:
        cfg = load_config("configs/v1_8_roi_baseline.yaml")
        self.assertIn(2, cfg["bins"]["service_levels"])
        self.assertIn(3, cfg["bins"]["service_levels"])
        self.assertIn("always_roi", cfg["simulation"]["policies"])
        self.assertIn("no_roi_greedy", cfg["simulation"]["policies"])
        self.assertEqual(cfg["simulation"]["service_level_order"], [0, 1, 3, 2])


if __name__ == "__main__":
    unittest.main()
