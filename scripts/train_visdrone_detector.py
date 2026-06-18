#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_detector_qwen.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    det_cfg = cfg["detector"]
    data_yaml = resolve_path(det_cfg["dataset_yaml"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO dataset yaml missing: {data_yaml}. Run prepare_visdrone_yolo.py first.")
    from ultralytics import YOLO

    weights = args.weights or det_cfg.get("base_model", "yolov8n.pt")
    epochs = int(args.epochs or det_cfg.get("epochs", 50))
    imgsz = int(args.imgsz or det_cfg.get("imgsz", 640))
    batches = [int(args.batch)] if args.batch else [int(v) for v in det_cfg.get("batch_fallback", [16, 8, 4])]
    project = resolve_path(det_cfg.get("train_project", "outputs/detector"))
    name = det_cfg.get("train_name", "visdrone_yolov8n")
    last_exc: Exception | None = None
    for batch in batches:
        try:
            model = YOLO(weights)
            result = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, project=str(project), name=name, exist_ok=True)
            print(f"trained_detector={project / name / 'weights' / 'best.pt'}")
            print(f"batch={batch} result={result}")
            return 0
        except RuntimeError as exc:
            last_exc = exc
            if "out of memory" not in str(exc).lower() and "cuda" not in str(exc).lower():
                raise
            print(f"batch={batch} failed with CUDA/runtime error, trying smaller batch: {exc}")
    if last_exc is not None:
        raise last_exc
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
