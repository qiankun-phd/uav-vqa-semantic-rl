#!/usr/bin/env python3
"""Q1/W1 smoke: exercise all four s3 ROI modes directly (no VLM).

Runs the detector at conf=0.05 over a handful of task images and calls
_build_roi_image until each mode {target_topk, suspect, thumbnail, dual}
has been produced at least once. Prints roi_meta + output path per sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.detector.visdrone_yolo import run_ultralytics_detector
from vqa_semcom.evidence.builder import image_path_for_task, read_tasks_csv

import scripts.run_v1_detector_eval as ev

CHANNEL = "10dB"
WANTED = {"target_topk", "suspect", "thumbnail", "dual"}


def main() -> int:
    out_dir = resolve_path("outputs/vlm/smoke_s3_roi")
    found: dict[str, str] = {}
    det_cache: dict[str, list] = {}

    def try_tasks(cfg_path: str, budget: int = 200) -> None:
        cfg = load_config(cfg_path)
        visdrone_root = resolve_path(cfg["paths"]["visdrone_val"])
        weights = resolve_path(cfg["detector"]["weights_path"])
        conf = float(cfg["detector"]["conf"])
        tasks = read_tasks_csv(resolve_path(cfg["paths"]["tasks_csv"]))
        for task in tasks[:budget]:
            if WANTED <= set(found):
                return
            image_id = task["image_id"]
            if image_id not in det_cache:
                src = image_path_for_task(visdrone_root, image_id)
                det_cache[image_id] = run_ultralytics_detector(src, weights, conf=conf).records
            src = image_path_for_task(visdrone_root, image_id)
            path, meta, mode = ev._build_roi_image(src, det_cache[image_id], task, CHANNEL, out_dir, cfg)
            if mode not in found:
                found[mode] = f"{task['question_type']:>11s} tgt={task.get('target_class','')}/{task.get('target_class_b','')} pol={task.get('presence_polarity','')} | {meta} | {path}"
                print(f"[{mode}] {found[mode]}")

    try_tasks("configs/s3_rician_main.json")
    try_tasks("configs/s3_rician_cmp.json", budget=40)
    missing = WANTED - set(found)
    print(f"\nmodes covered: {sorted(found)} missing: {sorted(missing) if missing else 'NONE'}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
