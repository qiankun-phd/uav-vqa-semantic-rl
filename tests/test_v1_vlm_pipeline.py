from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from vqa_semcom.config import load_config
from vqa_semcom.degradation.channel import degrade_image
from vqa_semcom.evidence.builder import build_degraded_evidence_records, build_lightweight_evidence
from vqa_semcom.data.visdrone import demo_objects
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut, read_csv, run_simulation, write_results
from vqa_semcom.tasks.generate_tasks import generate_tasks
from vqa_semcom.vlm.answer import check_answer, extract_count, normalize_yes_no
from vqa_semcom.vlm.lut import build_lut_from_predictions, write_lut_csv
from vqa_semcom.vlm.model_setup import looks_like_model_dir, model_cache_dir_name, resolve_model_reference


class V1VLMPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(ROOT / "configs" / "v0.yaml")
        self.tasks = [task.__dict__ for task in generate_tasks(demo_objects(), self.cfg)]

    def test_answer_normalizer_handles_yes_no_and_counts(self) -> None:
        self.assertEqual(normalize_yes_no("Yes, there are cars."), "yes")
        self.assertEqual(normalize_yes_no("No objects are visible."), "no")
        self.assertEqual(extract_count("There are 12 cars."), 12)
        self.assertTrue(check_answer("counting", "11", "12", tolerance_ratio=0.10).correct)
        self.assertFalse(check_answer("presence", "no", "yes").correct)

    def test_lightweight_evidence_contains_task_context(self) -> None:
        task = next(t for t in self.tasks if t["question_type"] == "presence")
        evidence = build_lightweight_evidence(task, demo_objects(), "good", self.cfg)
        self.assertIn("Object counts by class", evidence)
        self.assertIn(task["target_class"], evidence)
        self.assertIn("yes", evidence.lower())

    def test_detector_degradation_is_stronger_for_bad_channel(self) -> None:
        objects = demo_objects() * 8
        good = build_degraded_evidence_records(objects, "good", self.cfg, seed=3, image_id="demo")
        bad = build_degraded_evidence_records(objects, "bad", self.cfg, seed=3, image_id="demo")
        self.assertLessEqual(len(bad), len(good))
        self.assertTrue(all(record.bbox_x % 16 == 0 and record.bbox_y % 16 == 0 for record in bad))

    def test_image_degradation_outputs_different_channel_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_path = tmp_path / "sample.jpg"
            Image.new("RGB", (80, 60), "red").save(img_path)
            good = degrade_image(img_path, tmp_path / "out", "good", self.cfg)
            bad = degrade_image(img_path, tmp_path / "out", "bad", self.cfg)
            self.assertTrue(good.exists())
            self.assertTrue(bad.exists())
            self.assertNotEqual(good.read_bytes(), bad.read_bytes())
            self.assertGreater(good.stat().st_size, 0)
            self.assertGreater(bad.stat().st_size, 0)

    def test_payload_bytes_follow_service_levels(self) -> None:
        from scripts.run_v1_vlm_eval import _payload_bytes

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "evidence.jpg"
            image_path.write_bytes(b"image-bytes")
            self.assertEqual(_payload_bytes(0, "cache", "ignored"), 0)
            self.assertEqual(_payload_bytes(1, "lightweight", "abc"), 3)
            self.assertEqual(_payload_bytes(2, "image", str(image_path)), len(b"image-bytes"))

    def test_vlm_lut_aggregation_bounds_accuracy(self) -> None:
        rows = [
            {
                "question_type": "presence",
                "service_level": "1",
                "channel_bin": "good",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
                "risk_level": "normal",
                "correct": "True",
                "payload_bytes": "100",
            },
            {
                "question_type": "presence",
                "service_level": "1",
                "channel_bin": "good",
                "view_quality_bin": "good",
                "freshness_bin": "fresh",
                "risk_level": "normal",
                "correct": "False",
                "payload_bytes": "300",
            },
        ]
        lut = build_lut_from_predictions(rows)
        self.assertEqual(len(lut), 1)
        self.assertEqual(lut[0].sample_count, 2)
        self.assertGreaterEqual(lut[0].expected_accuracy, 0.0)
        self.assertLessEqual(lut[0].expected_accuracy, 1.0)
        self.assertEqual(lut[0].avg_payload_bytes, 200.0)
        self.assertAlmostEqual(lut[0].avg_payload_kb, 200.0 / 1024.0, places=6)

    def test_prediction_schema_columns_are_available(self) -> None:
        from scripts.run_v1_vlm_eval import PREDICTION_FIELDNAMES

        required = {"question", "ground_truth_answer", "service_level", "predicted_answer", "correct", "model_name", "payload_bytes"}
        self.assertTrue(required.issubset(set(PREDICTION_FIELDNAMES)))

    def test_measured_lut_can_drive_resource_simulation(self) -> None:
        prediction_rows = []
        for task in self.tasks:
            if task["question_type"] not in {"presence", "counting"}:
                continue
            for level in [0, 1, 2]:
                for channel in self.cfg["bins"]["channel"]:
                    for freshness in self.cfg["bins"]["freshness"]:
                        prediction_rows.append(
                            {
                                "question_type": task["question_type"],
                                "service_level": str(level),
                                "channel_bin": channel,
                                "view_quality_bin": task["view_quality_bin"],
                                "freshness_bin": freshness,
                                "risk_level": task["risk_level"],
                                "correct": "True" if level == 2 or channel == "good" else "False",
                                "payload_bytes": str({0: 0, 1: 2048, 2: 204800}[level]),
                            }
                        )
        lut_rows = build_lut_from_predictions(prediction_rows)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lut_csv = tmp_path / "lut.csv"
            results_csv = tmp_path / "results.csv"
            results_md = tmp_path / "results.md"
            write_lut_csv(lut_rows, lut_csv)
            lut = load_lut(lut_csv)
            supported = filter_tasks_supported_by_lut([dict((k, str(v)) for k, v in task.items()) for task in self.tasks], lut)
            self.assertTrue(supported)
            cfg = dict(self.cfg)
            cfg["simulation"] = dict(self.cfg["simulation"])
            cfg["simulation"]["policies"] = ["always_cache", "oracle_best_feasible_evidence"]
            results = run_simulation(supported, lut, cfg, episodes=1)
            write_results(results, results_csv, results_md)
            self.assertTrue(results_csv.exists())
            self.assertTrue(results_md.exists())
            self.assertEqual({result.policy for result in results}, {"always_cache", "oracle_best_feasible_evidence"})
            self.assertTrue(all(hasattr(result, "average_payload_kb") for result in results))
            self.assertTrue(all(-1.0 <= result.payload_reduction_vs_always_image <= 1.0 for result in results))
            self.assertIn("payload KB", results_md.read_text(encoding="utf-8"))

    def test_model_reference_prefers_complete_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "qwen_local"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type":"qwen2_vl"}', encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"0" * 2048)
            self.assertTrue(looks_like_model_dir(model_dir))
            ref = resolve_model_reference({"model_name": "Qwen/Qwen2-VL-2B-Instruct", "model_local_path": str(model_dir)})
            self.assertEqual(ref, str(model_dir))
            self.assertEqual(model_cache_dir_name("Qwen/Qwen2-VL-2B-Instruct"), "models--Qwen--Qwen2-VL-2B-Instruct")


if __name__ == "__main__":
    unittest.main()
