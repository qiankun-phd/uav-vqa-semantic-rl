#!/usr/bin/env python3
"""Generate the three DroneVehicle rician configs (Qwen2-VL-2B, unchanged model so
the numbers are directly comparable to the VisDrone main experiment).

Base = configs/v2_0_ldpc_channel.yaml (rician). Only the paths that must point at
DroneVehicle are overridden; the model / channel / thresholds are left intact.

Image integration: paths.visdrone_val -> data/raw/dronevehicle/val/adapter, whose
images/ subdir is a symlink to val/valimg (image_path_for_task resolves
<visdrone_val>/images/<image_id>.jpg). The 100px white border is NOT cropped.
"""
import json
import sys

sys.path.insert(0, "src")
from vqa_semcom.config import load_config

DV_VAL = "data/raw/dronevehicle/val/adapter"

specs = {
    "main": ("outputs/tasks/dv_tasks.csv", ["presence", "counting"]),
    "cmp": ("outputs/tasks/dv_comparison_tasks.csv", ["comparison"]),
    "extra": ("outputs/tasks/dv_extra_tasks.csv", ["co_presence", "threshold"]),
}

for tag, (tasks_csv, qtypes) in specs.items():
    c = load_config("configs/v2_0_ldpc_channel.yaml")  # rician, Qwen2-VL-2B
    c["paths"]["visdrone_val"] = DV_VAL
    c["paths"]["tasks_csv"] = tasks_csv
    c["paths"]["vlm_predictions_csv"] = f"outputs/vlm/dv_rician_{tag}_predictions.csv"
    c["paths"]["degraded_image_dir"] = f"outputs/vlm/degraded_images_dv_rician_{tag}"
    c["vlm"]["question_types"] = qtypes
    c.setdefault("detector", {})["detections_csv"] = f"outputs/detector/dv_rician_{tag}_detections.csv"
    out = f"configs/dv_rician_{tag}.json"
    json.dump(c, open(out, "w"), indent=2)
    print("wrote", out, "| model=", c["vlm"]["model_name"], "| qtypes=", qtypes,
          "| tasks=", tasks_csv)
