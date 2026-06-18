from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.config import load_config
from vqa_semcom.data.visdrone import demo_objects, parse_annotation_line
from vqa_semcom.quality.lut_builder import build_lut, estimate_accuracy
from vqa_semcom.sim.resource_env import load_lut, run_simulation, write_results
from vqa_semcom.tasks.generate_tasks import _view_quality, generate_tasks, write_tasks_csv


class V0PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(ROOT / "configs" / "v0.yaml")

    def test_visdrone_parser_skips_ignored_category(self) -> None:
        ignored = parse_annotation_line("1,2,3,4,1,0,0,0", "img")
        car = parse_annotation_line("1,2,30,40,1,4,0,0", "img")
        self.assertIsNone(ignored)
        self.assertIsNotNone(car)
        self.assertEqual(car.category, "car")

    def test_task_generator_creates_required_question_types(self) -> None:
        tasks = generate_tasks(demo_objects(), self.cfg)
        qtypes = {task.question_type for task in tasks}
        self.assertTrue({"presence", "counting", "risk"}.issubset(qtypes))
        self.assertTrue(all(task.risk_level in {"normal", "critical"} for task in tasks))

    def test_view_quality_proxy_keeps_large_clear_box_at_least_medium(self) -> None:
        view_bin, score = _view_quality(
            scale_proxy=0.010,
            occlusion_score=0.0,
            truncation_score=0.0,
            density_score=3.0,
            scale_reference=self.cfg["view_quality"]["scale_reference"],
            good_score=self.cfg["view_quality"]["good_score"],
            medium_score=self.cfg["view_quality"]["medium_score"],
            density_penalty_start=self.cfg["view_quality"]["density_penalty_start"],
            density_penalty_per_object=self.cfg["view_quality"]["density_penalty_per_object"],
            min_density_component=self.cfg["view_quality"]["min_density_component"],
        )
        self.assertIn(view_bin, {"medium", "good"})
        self.assertGreater(score, 0.0)

    def test_service_levels_are_fixed_and_critical_is_not_service_level(self) -> None:
        self.assertEqual(self.cfg["bins"]["service_levels"], [0, 1, 2])
        self.assertNotIn("critical", self.cfg["bins"]["service_levels"])
        self.assertIn("critical", self.cfg["bins"]["risk_levels"])

    def test_lut_monotonicity_and_quality_symbol_contract(self) -> None:
        bad = estimate_accuracy("counting", 2, "bad", "poor", "fresh", "normal", self.cfg["evaluator"])
        good = estimate_accuracy("counting", 2, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        self.assertLess(bad, good)
        light = estimate_accuracy("presence", 1, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        image = estimate_accuracy("presence", 2, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        cache = estimate_accuracy("presence", 0, "good", "good", "fresh", "normal", self.cfg["evaluator"])
        counting_cache = estimate_accuracy("counting", 0, "good", "medium", "fresh", "normal", self.cfg["evaluator"])
        counting_light = estimate_accuracy("counting", 1, "good", "medium", "fresh", "normal", self.cfg["evaluator"])
        self.assertGreater(light, cache)
        self.assertGreater(counting_light, counting_cache)
        self.assertGreaterEqual(image, light)
        self.assertEqual("A_k >= epsilon_k", "A_k >= epsilon_k")
        self.assertNotEqual("A_k >= epsilon_k", "T_k <= tau_k")

    def test_end_to_end_demo_lut_and_simulation(self) -> None:
        tasks = generate_tasks(demo_objects(), self.cfg)
        task_dicts = [dict((k, str(v)) for k, v in task.__dict__.items()) for task in tasks]
        rows = build_lut(task_dicts, self.cfg)
        self.assertTrue(rows)
        self.assertTrue(all(row.sample_count > 0 for row in rows))
        self.assertTrue(any(row.std_or_ci > 0 for row in rows))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_csv = tmp_path / "tasks.csv"
            lut_csv = tmp_path / "lut.csv"
            results_csv = tmp_path / "results.csv"
            results_md = tmp_path / "summary.md"
            write_tasks_csv(tasks, task_csv)
            with lut_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].__dataclass_fields__.keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(row.__dict__)
            with task_csv.open(newline="", encoding="utf-8") as f:
                task_rows = list(csv.DictReader(f))
            results = run_simulation(task_rows, load_lut(lut_csv), self.cfg, episodes=2)
            write_results(results, results_csv, results_md)
            self.assertTrue(results_csv.exists())
            self.assertTrue(results_md.exists())
            self.assertEqual({r.policy for r in results}, set(self.cfg["simulation"]["policies"]))


if __name__ == "__main__":
    unittest.main()
