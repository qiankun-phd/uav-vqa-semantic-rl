from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_v1_7_tasks import build_v1_7_tasks
from scripts.run_v1_detector_eval import (
    PREDICTION_FIELDNAMES,
    _calibrated_count,
    _semantic_token_prediction,
)
from scripts.report_v1_6_success_breakdown import _group_rows


class V17QualityCalibrationTest(unittest.TestCase):
    def _cfg(self, root: Path) -> dict:
        return {
            "paths": {"visdrone_val": str(root)},
            "thresholds": {
                "epsilon_normal": 0.65,
                "epsilon_critical": 0.82,
                "tau_normal": 5.0,
                "tau_critical": 3.0,
            },
            "view_quality": {
                "scale_reference": 0.0035,
                "good_score": 0.62,
                "medium_score": 0.30,
                "density_penalty_start": 6.0,
                "density_penalty_per_object": 0.025,
                "min_density_component": 0.55,
            },
        }

    def test_v1_7_tasks_include_negative_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "annotations").mkdir(parents=True)
            (root / "annotations" / "img001.txt").write_text(
                "10,10,20,20,1,4,0,0\n40,10,8,20,1,1,0,0\n",
                encoding="utf-8",
            )
            tasks = build_v1_7_tasks(self._cfg(root))
            negatives = [task for task in tasks if task.get("presence_polarity") == "negative"]
            positives = [task for task in tasks if task.get("presence_polarity") == "positive"]
            self.assertTrue(positives)
            self.assertTrue(negatives)
            present_classes = {task["target_class"] for task in positives}
            self.assertTrue(all(task["answer"] == "no" for task in negatives))
            self.assertTrue(all(task["target_class"] not in present_classes for task in negatives))

    def test_prediction_schema_has_v1_7_decoder_fields(self) -> None:
        required = {
            "presence_polarity",
            "decoder_mode",
            "raw_detector_count",
            "transmitted_detector_count",
            "calibrated_detector_count",
            "raw_decoder_correct",
        }
        self.assertTrue(required.issubset(set(PREDICTION_FIELDNAMES)))

    def test_direct_presence_decoder_follows_detector_count(self) -> None:
        task = {"question_type": "presence", "answer": "yes"}
        pred, norm, correct = _semantic_token_prediction(task, 2, {"vlm": {}})
        self.assertEqual(pred, "yes")
        self.assertEqual(norm, "yes")
        self.assertTrue(correct)
        pred, norm, correct = _semantic_token_prediction(task, 0, {"vlm": {}})
        self.assertEqual(pred, "no")
        self.assertFalse(correct)

    def test_calibrated_count_keeps_raw_count_available(self) -> None:
        ratios = {("car", "bad"): 2.0}
        self.assertEqual(_calibrated_count(3, "car", "bad", ratios), 6)
        self.assertEqual(_calibrated_count(2, "car", "bad", ratios), 2)
        self.assertEqual(_calibrated_count(0, "car", "bad", ratios), 0)
        self.assertEqual(_calibrated_count(3, "bus", "bad", ratios), 3)

    def test_success_breakdown_metrics_are_bounded(self) -> None:
        rows = [
            {
                "service_level": "1",
                "question_type": "presence",
                "risk_level": "normal",
                "channel_bin": "good",
                "correct": "True",
                "epsilon_k": "0.65",
                "tau_k": "5.0",
            },
            {
                "service_level": "1",
                "question_type": "presence",
                "risk_level": "normal",
                "channel_bin": "bad",
                "correct": "False",
                "epsilon_k": "0.65",
                "tau_k": "5.0",
            },
        ]
        for row in rows:
            row["view_quality_bin"] = "good"
            row["freshness_bin"] = "fresh"
        lut = {
            ("presence", "1", "good", "good", "fresh", "normal"): 0.8,
            ("presence", "1", "bad", "good", "fresh", "normal"): 0.5,
        }
        cfg = {
            "simulation": {
                "delay_by_level": {"1": 1.8},
                "channel_delay_multiplier": {"good": 1.0, "bad": 1.0},
            }
        }
        metrics = _group_rows(rows, lut, cfg, ["service_level"])[0]
        for key in ["answer_correctness", "quality_satisfaction", "deadline_satisfaction", "final_success"]:
            value = float(metrics[key])
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
