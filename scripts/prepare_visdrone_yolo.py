#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.detector.visdrone_yolo import convert_split_to_yolo, write_class_mapping, write_dataset_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_detector_qwen.yaml")
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    det_cfg = cfg["detector"]
    out_root = resolve_path(det_cfg["yolo_dataset_dir"])
    train_root = resolve_path(cfg["paths"].get("visdrone_train", "data/raw/visdrone/DET/train"))
    val_root = resolve_path(cfg["paths"]["visdrone_val"])
    train_count = convert_split_to_yolo(train_root, out_root, "train", limit_images=args.limit_images) if train_root.exists() else 0
    val_count = convert_split_to_yolo(val_root, out_root, "val", limit_images=args.limit_images)
    write_dataset_yaml(out_root, resolve_path(det_cfg["dataset_yaml"]))
    write_class_mapping(resolve_path(det_cfg["class_mapping_json"]))
    print(f"yolo_dataset_dir={out_root}")
    print(f"train_images={train_count} val_images={val_count}")
    print(f"dataset_yaml={resolve_path(det_cfg['dataset_yaml'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
