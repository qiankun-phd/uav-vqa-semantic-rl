#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path
from vqa_semcom.data.visdrone import VISDRONE_CATEGORY, load_visdrone_annotations
from vqa_semcom.tasks.generate_tasks import generate_tasks


TASK_FIELDNAMES = [
    "image_id",
    "question_type",
    "question",
    "answer",
    "target_class",
    "object_count",
    "risk_level",
    "epsilon_k",
    "tau_k",
    "view_quality_bin",
    "scale_proxy",
    "occlusion_score",
    "truncation_score",
    "density_score",
    "presence_polarity",
]


def _candidate_classes() -> list[str]:
    excluded = {"ignored", "others"}
    return sorted({name for name in VISDRONE_CATEGORY.values() if name not in excluded})


def _negative_task_from_template(template: dict[str, object], target_class: str) -> dict[str, object]:
    out = dict(template)
    out.update(
        {
            "question_type": "presence",
            "question": f"Are there {target_class} objects in this area?",
            "answer": "no",
            "target_class": target_class,
            "object_count": 0,
            "risk_level": "normal",
            "presence_polarity": "negative",
        }
    )
    return out


def _with_polarity(task: dict[str, object]) -> dict[str, object]:
    out = dict(task)
    if out.get("question_type") == "presence":
        out["presence_polarity"] = "positive" if str(out.get("answer", "")).lower() == "yes" else "negative"
    else:
        out["presence_polarity"] = ""
    return out


def build_v1_7_tasks(cfg: dict, limit_images: int | None = None) -> list[dict[str, object]]:
    objects = load_visdrone_annotations(resolve_path(cfg["paths"]["visdrone_val"]), limit_images=limit_images)
    base_tasks = [_with_polarity(asdict(task)) for task in generate_tasks(objects, cfg)]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for task in base_tasks:
        if task["question_type"] in {"presence", "counting"}:
            grouped[str(task["image_id"])].append(task)

    classes = _candidate_classes()
    out: list[dict[str, object]] = []
    for image_id in sorted(grouped):
        image_tasks = grouped[image_id]
        positives = [task for task in image_tasks if task["question_type"] == "presence" and task["answer"] == "yes"]
        countings = [task for task in image_tasks if task["question_type"] == "counting"]
        present_classes = {str(task["target_class"]) for task in positives}
        missing_classes = [name for name in classes if name not in present_classes]
        negatives = [
            _negative_task_from_template(positives[idx % max(1, len(positives))], missing_classes[idx])
            for idx in range(min(len(positives), len(missing_classes)))
        ] if positives else []

        max_len = max(len(positives), len(negatives), len(countings))
        for idx in range(max_len):
            if idx < len(positives):
                out.append(positives[idx])
            if idx < len(negatives):
                out.append(negatives[idx])
            if idx < len(countings):
                out.append(countings[idx])
    return out


def write_tasks_csv(rows: list[dict[str, object]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TASK_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_7_quality_calibrated.yaml")
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = build_v1_7_tasks(cfg, limit_images=args.limit_images)
    out_path = resolve_path(cfg["paths"]["tasks_csv"])
    write_tasks_csv(rows, out_path)
    positives = sum(1 for row in rows if row.get("presence_polarity") == "positive")
    negatives = sum(1 for row in rows if row.get("presence_polarity") == "negative")
    countings = sum(1 for row in rows if row.get("question_type") == "counting")
    print(f"tasks_csv={out_path}")
    print(f"tasks={len(rows)} positive_presence={positives} negative_presence={negatives} counting={countings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
