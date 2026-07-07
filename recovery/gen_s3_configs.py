#!/usr/bin/env python3
"""Q1/W1: configs for the s3 (detector-guided ROI) rician runs.

Based on the v26 SmolVLM rician configs (same model / channel / task sets) so
that s3 rows are directly comparable with the logged v26 s1/s2 rows.
Detector conf floor is lowered to 0.05 so the suspect-mode ROI builder can see
sub-threshold candidate boxes; tau=0.25 keeps the "confident" definition
identical to the s1/s2 baseline detector operating point.
"""
import json
from pathlib import Path

ROI_PARAMS = {
    "conf": 0.05,                    # detector floor (suspect band needs <0.25 boxes)
    "roi_conf_threshold": 0.25,      # tau: confident-detection threshold (= baseline conf)
    "roi_suspect_conf_floor": 0.05,  # suspect band = [floor, tau)
    "roi_top_k": 3,
    "roi_expand_ratio": 1.5,
    "roi_thumbnail_max_side": 512,
}

for tag in ("main", "cmp", "extra"):
    src = Path(f"configs/v26_rician_{tag}.json")
    c = json.loads(src.read_text())
    c["paths"]["vlm_predictions_csv"] = f"outputs/vlm/s3_rician_{tag}_predictions.csv"
    c["paths"]["degraded_image_dir"] = f"outputs/vlm/degraded_images_s3_rician_{tag}"
    c.setdefault("detector", {})["detections_csv"] = f"outputs/detector/s3_rician_{tag}_detections.csv"
    c["detector"].update(ROI_PARAMS)
    out = Path(f"configs/s3_rician_{tag}.json")
    out.write_text(json.dumps(c, indent=2))
    print("wrote", out, "| qtypes=", c["vlm"]["question_types"], "| detector.conf=", c["detector"]["conf"])
