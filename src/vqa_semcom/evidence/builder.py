from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vqa_semcom.config import resolve_path
from vqa_semcom.data.visdrone import VisDroneObject, parse_annotation_line


@dataclass(frozen=True)
class EvidenceRecord:
    category: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    confidence: float
    occlusion: int
    truncation: int


def read_tasks_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_path_for_task(visdrone_root: Path, image_id: str) -> Path:
    for suffix in [".jpg", ".png", ".jpeg"]:
        candidate = visdrone_root / "images" / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return visdrone_root / "images" / f"{image_id}.jpg"


def load_objects_for_image(visdrone_root: Path, image_id: str) -> list[VisDroneObject]:
    ann_path = visdrone_root / "annotations" / f"{image_id}.txt"
    if not ann_path.exists():
        return []
    objects: list[VisDroneObject] = []
    with ann_path.open(encoding="utf-8") as f:
        for line in f:
            obj = parse_annotation_line(line, image_id=image_id)
            if obj is not None:
                objects.append(obj)
    return objects


def load_objects_by_image(visdrone_root: Path, image_ids: Iterable[str]) -> dict[str, list[VisDroneObject]]:
    return {image_id: load_objects_for_image(visdrone_root, image_id) for image_id in sorted(set(image_ids))}


def select_vlm_tasks(
    tasks: list[dict[str, str]],
    question_types: set[str],
    limit_images: int | None = None,
    max_tasks: int | None = None,
    max_tasks_per_image: int = 4,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        if task.get("question_type") in question_types:
            grouped[task["image_id"]].append(task)
    selected: list[dict[str, str]] = []
    for image_idx, image_id in enumerate(sorted(grouped)):
        if limit_images is not None and image_idx >= limit_images:
            break
        for task in grouped[image_id][:max_tasks_per_image]:
            selected.append(task)
            if max_tasks is not None and len(selected) >= max_tasks:
                return selected
    return selected


def _rng_for(*parts: str, seed: int = 0) -> random.Random:
    raw = "|".join([str(seed), *parts])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _pseudo_confidence(obj: VisDroneObject, image_id: str, channel_bin: str, seed: int) -> float:
    rng = _rng_for(image_id, obj.category, str(obj.bbox_x), str(obj.bbox_y), channel_bin, seed=seed)
    quality_penalty = 0.08 * obj.occlusion + 0.05 * obj.truncation
    return max(0.05, min(0.99, 0.92 - quality_penalty - 0.18 * rng.random()))


def _quantize(value: float, step: int) -> int:
    if step <= 1:
        return int(round(value))
    return int(round(value / step) * step)


def degrade_objects_for_light_evidence(
    objects: list[VisDroneObject],
    channel_bin: str,
    cfg: dict,
    seed: int = 0,
    image_id: str = "",
) -> list[VisDroneObject]:
    deg_cfg = cfg.get("vlm", {}).get("light_evidence_degradation", {}).get(channel_bin, {})
    drop_rate = float(deg_cfg.get("drop_rate", 0.0))
    rng = _rng_for(image_id, channel_bin, seed=str(seed))
    kept = [obj for obj in objects if rng.random() >= drop_rate]
    return kept or objects[:1]


def build_degraded_evidence_records(
    objects: list[VisDroneObject],
    channel_bin: str,
    cfg: dict,
    seed: int = 0,
    image_id: str = "",
) -> list[EvidenceRecord]:
    deg_cfg = cfg.get("vlm", {}).get("light_evidence_degradation", {}).get(channel_bin, {})
    drop_rate = float(deg_cfg.get("drop_rate", 0.0))
    corrupt_rate = float(deg_cfg.get("class_corrupt_rate", 0.0))
    quantization = int(deg_cfg.get("bbox_quantization", 1))
    confidence_threshold = float(deg_cfg.get("confidence_threshold", 0.0))
    rng = _rng_for(image_id, channel_bin, "records", seed=seed)
    records: list[EvidenceRecord] = []
    for obj in objects:
        confidence = _pseudo_confidence(obj, image_id, channel_bin, seed)
        if confidence < confidence_threshold or rng.random() < drop_rate:
            continue
        category = obj.category
        if corrupt_rate > 0 and rng.random() < corrupt_rate:
            category = "unknown"
        records.append(
            EvidenceRecord(
                category=category,
                bbox_x=_quantize(obj.bbox_x, quantization),
                bbox_y=_quantize(obj.bbox_y, quantization),
                bbox_w=max(1, _quantize(obj.bbox_w, quantization)),
                bbox_h=max(1, _quantize(obj.bbox_h, quantization)),
                confidence=round(confidence, 3),
                occlusion=obj.occlusion,
                truncation=obj.truncation,
            )
        )
    if records:
        return records
    fallback = sorted(objects, key=lambda obj: _pseudo_confidence(obj, image_id, channel_bin, seed), reverse=True)[:1]
    return [
        EvidenceRecord(
            category=obj.category,
            bbox_x=_quantize(obj.bbox_x, quantization),
            bbox_y=_quantize(obj.bbox_y, quantization),
            bbox_w=max(1, _quantize(obj.bbox_w, quantization)),
            bbox_h=max(1, _quantize(obj.bbox_h, quantization)),
            confidence=round(_pseudo_confidence(obj, image_id, channel_bin, seed), 3),
            occlusion=obj.occlusion,
            truncation=obj.truncation,
        )
        for obj in fallback
    ]


def build_lightweight_evidence(task: dict[str, str], objects: list[VisDroneObject], channel_bin: str, cfg: dict) -> str:
    seed = int(cfg.get("vlm", {}).get("seed", 0))
    degraded = build_degraded_evidence_records(objects, channel_bin, cfg, seed=seed, image_id=task["image_id"])
    class_counts = Counter(obj.category for obj in degraded)
    target = task.get("target_class", "")
    target_objects = [obj for obj in degraded if obj.category == target]
    box_summaries = []
    for obj in target_objects[:8]:
        box_summaries.append(
            f"{obj.category}(x={obj.bbox_x}, y={obj.bbox_y}, w={obj.bbox_w}, h={obj.bbox_h}, "
            f"conf={obj.confidence:.2f}, occ={obj.occlusion}, trunc={obj.truncation})"
        )
    count_text = ", ".join(f"{name}:{count}" for name, count in sorted(class_counts.items())) or "none"
    boxes_text = "; ".join(box_summaries) if box_summaries else "no target boxes retained"
    return (
        "You are given lightweight UAV semantic evidence extracted from VisDrone annotations. "
        "Answer the question using only this evidence. "
        f"Image id: {task['image_id']}. Object counts by class: {count_text}. "
        f"Target class: {target}. Target box summaries: {boxes_text}. "
        "Reply with only 'yes' or 'no' for presence questions, or a single integer for counting questions."
    )


def build_vlm_prompt(task: dict[str, str], evidence_text: str | None = None) -> str:
    instruction = (
        "Answer the visual question for a UAV emergency inspection scene. "
        "For presence questions, reply only yes or no. For counting questions, reply only one integer."
    )
    if evidence_text:
        return f"{instruction}\n\n{evidence_text}\n\nQuestion: {task['question']}"
    return f"{instruction}\n\nQuestion: {task['question']}"
