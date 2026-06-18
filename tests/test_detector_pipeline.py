from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from vqa_semcom.detector.visdrone_yolo import (
    DetectionRecord,
    build_detector_lightweight_evidence,
    convert_split_to_yolo,
    visdrone_annotation_to_yolo_lines,
)


class DetectorPipelineTest(unittest.TestCase):
    def test_visdrone_to_yolo_skips_ignored_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.jpg"
            ann = root / "sample.txt"
            Image.new("RGB", (100, 50), "white").save(image)
            ann.write_text("10,5,20,10,1,4,0,0\n1,1,10,10,1,0,0,0\n", encoding="utf-8")
            lines = visdrone_annotation_to_yolo_lines(ann, image)
            self.assertEqual(len(lines), 1)
            parts = lines[0].split()
            self.assertEqual(len(parts), 5)
            self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in parts[1:]))

    def test_convert_split_to_yolo_writes_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "train"
            (split / "images").mkdir(parents=True)
            (split / "annotations").mkdir(parents=True)
            Image.new("RGB", (100, 100), "white").save(split / "images" / "img001.jpg")
            (split / "annotations" / "img001.txt").write_text("10,10,20,20,1,1,0,0\n", encoding="utf-8")
            out = Path(tmp) / "yolo"
            count = convert_split_to_yolo(split, out, "train")
            self.assertEqual(count, 1)
            self.assertTrue((out / "labels" / "train" / "img001.txt").exists())
            self.assertTrue((out / "images" / "train" / "img001.jpg").exists())

    def test_detector_evidence_uses_detector_records(self) -> None:
        task = {"image_id": "demo", "question": "How many car objects are in this area?", "target_class": "car"}
        records = [
            DetectionRecord("car", 10, 10, 20, 20, 0.91),
            DetectionRecord("pedestrian", 40, 10, 8, 20, 0.80),
        ]
        cfg = {"vlm": {"light_evidence_degradation": {"good": {}}}}
        evidence = build_detector_lightweight_evidence(task, records, "good", cfg)
        self.assertIn("real object detector", evidence)
        self.assertIn("car:1", evidence)
        self.assertIn("conf=0.91", evidence)


if __name__ == "__main__":
    unittest.main()
