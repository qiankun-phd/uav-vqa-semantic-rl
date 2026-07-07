from __future__ import annotations

import csv
import json
import shutil
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from vqa_semcom.config import ensure_parent
from vqa_semcom.data.visdrone import VISDRONE_CATEGORY, parse_annotation_line
from vqa_semcom.snr import degradation_config

DETECTOR_CATEGORY_IDS = [idx for idx in sorted(VISDRONE_CATEGORY) if idx != 0]
CATEGORY_TO_YOLO = {category_id: yolo_id for yolo_id, category_id in enumerate(DETECTOR_CATEGORY_IDS)}
YOLO_TO_CATEGORY = {yolo_id: VISDRONE_CATEGORY[category_id] for category_id, yolo_id in CATEGORY_TO_YOLO.items()}


@dataclass(frozen=True)
class DetectionRecord:
    category: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    confidence: float


@dataclass(frozen=True)
class DetectorResult:
    records: list[DetectionRecord]
    latency_sec: float
    model_name: str


def class_mapping_payload() -> dict[str, object]:
    return {
        "source": "VisDrone DET category ids, excluding ignored regions only",
        "category_to_yolo": {str(k): v for k, v in CATEGORY_TO_YOLO.items()},
        "yolo_to_category": {str(k): v for k, v in YOLO_TO_CATEGORY.items()},
        "names": [YOLO_TO_CATEGORY[idx] for idx in sorted(YOLO_TO_CATEGORY)],
    }


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _image_path_for_annotation(split_root: Path, image_id: str) -> Path | None:
    for suffix in [".jpg", ".jpeg", ".png"]:
        candidate = split_root / "images" / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def visdrone_annotation_to_yolo_lines(annotation_path: Path, image_path: Path) -> list[str]:
    width, height = _image_size(image_path)
    lines: list[str] = []
    with annotation_path.open(encoding="utf-8") as f:
        for raw in f:
            obj = parse_annotation_line(raw, image_id=annotation_path.stem, image_width=width, image_height=height)
            if obj is None or obj.category_id not in CATEGORY_TO_YOLO:
                continue
            x_center = (obj.bbox_x + obj.bbox_w / 2.0) / max(1, width)
            y_center = (obj.bbox_y + obj.bbox_h / 2.0) / max(1, height)
            box_w = obj.bbox_w / max(1, width)
            box_h = obj.bbox_h / max(1, height)
            vals = [x_center, y_center, box_w, box_h]
            if not all(0.0 <= value <= 1.0 for value in vals):
                vals = [min(1.0, max(0.0, value)) for value in vals]
            if vals[2] <= 0 or vals[3] <= 0:
                continue
            lines.append(f"{CATEGORY_TO_YOLO[obj.category_id]} " + " ".join(f"{value:.6f}" for value in vals))
    return lines


def _link_or_copy(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert_split_to_yolo(split_root: Path, out_root: Path, split_name: str, limit_images: int | None = None) -> int:
    ann_dir = split_root / "annotations"
    if not ann_dir.exists():
        raise FileNotFoundError(f"VisDrone annotations not found: {ann_dir}")
    image_out = out_root / "images" / split_name
    label_out = out_root / "labels" / split_name
    converted = 0
    for ann_idx, ann_path in enumerate(sorted(ann_dir.glob("*.txt"))):
        if limit_images is not None and ann_idx >= limit_images:
            break
        image_path = _image_path_for_annotation(split_root, ann_path.stem)
        if image_path is None:
            continue
        yolo_lines = visdrone_annotation_to_yolo_lines(ann_path, image_path)
        if not yolo_lines:
            continue
        out_image_path = image_out / image_path.name
        out_label_path = label_out / f"{image_path.stem}.txt"
        _link_or_copy(image_path.resolve(), out_image_path)
        ensure_parent(out_label_path)
        out_label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        converted += 1
    return converted


def write_dataset_yaml(out_root: Path, yaml_path: Path) -> None:
    names = [YOLO_TO_CATEGORY[idx] for idx in sorted(YOLO_TO_CATEGORY)]
    ensure_parent(yaml_path)
    lines = [
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_class_mapping(path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(class_mapping_payload(), indent=2), encoding="utf-8")


def load_detector_classes(path: Path | None = None) -> dict[int, str]:
    if path is None or not path.exists():
        return dict(YOLO_TO_CATEGORY)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in payload.get("yolo_to_category", {}).items()}


def run_ultralytics_detector(image_path: Path, weights_path: Path, conf: float = 0.25, imgsz: int = 640) -> DetectorResult:
    if not weights_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {weights_path}")
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    start = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    latency = time.perf_counter() - start
    records: list[DetectionRecord] = []
    if results:
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for box, score, cls_id in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = box
                class_id = int(cls_id)
                category = str(names.get(class_id, YOLO_TO_CATEGORY.get(class_id, "unknown")))
                records.append(
                    DetectionRecord(
                        category=category,
                        bbox_x=max(0, int(round(x1))),
                        bbox_y=max(0, int(round(y1))),
                        bbox_w=max(1, int(round(x2 - x1))),
                        bbox_h=max(1, int(round(y2 - y1))),
                        confidence=round(float(score), 4),
                    )
                )
    return DetectorResult(records=records, latency_sec=latency, model_name=str(weights_path))


def _stable_score(*parts: str) -> float:
    import hashlib

    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _degrade_detections_ldpc(records: Iterable[DetectionRecord], channel_bin: str, cfg: dict, image_id: str) -> list[DetectionRecord]:
    """Token transmission over the LDPC-coded fading link: lost frames (prob=FER)
    are dropped (header lost) or garbled (payload lost: class -> unknown, box jitter)."""
    from vqa_semcom.degradation.digital_link import fer_for

    fer = fer_for(cfg, channel_bin)
    out: list[DetectionRecord] = []
    for idx, record in enumerate(records):
        if _stable_score(image_id, channel_bin, record.category, str(idx), "frame") >= fer:
            out.append(record)
            continue
        if _stable_score(image_id, channel_bin, record.category, str(idx), "mode") < 0.5:
            continue  # frame header lost -> detection dropped
        jx = 0.7 + 0.6 * _stable_score(image_id, channel_bin, str(idx), "jx")
        jy = 0.7 + 0.6 * _stable_score(image_id, channel_bin, str(idx), "jy")
        out.append(
            DetectionRecord(
                category="unknown",
                bbox_x=int(record.bbox_x * jx),
                bbox_y=int(record.bbox_y * jy),
                bbox_w=max(1, int(record.bbox_w * jx)),
                bbox_h=max(1, int(record.bbox_h * jy)),
                confidence=record.confidence,
            )
        )
    return out


def degrade_detections_for_channel(records: Iterable[DetectionRecord], channel_bin: str, cfg: dict, image_id: str) -> list[DetectionRecord]:
    if (cfg.get("vlm", {}) or {}).get("channel_model") == "ldpc_fading":
        return _degrade_detections_ldpc(records, channel_bin, cfg, image_id)
    deg_cfg = degradation_config("light", channel_bin, cfg)
    drop_rate = float(deg_cfg.get("drop_rate", 0.0))
    corrupt_rate = float(deg_cfg.get("class_corrupt_rate", 0.0))
    quantization = int(deg_cfg.get("bbox_quantization", 1))
    confidence_threshold = float(deg_cfg.get("confidence_threshold", 0.0))
    out: list[DetectionRecord] = []
    for idx, record in enumerate(records):
        if record.confidence < confidence_threshold:
            continue
        if _stable_score(image_id, channel_bin, record.category, str(idx), "drop") < drop_rate:
            continue
        category = record.category
        if corrupt_rate > 0 and _stable_score(image_id, channel_bin, record.category, str(idx), "corrupt") < corrupt_rate:
            category = "unknown"
        q = max(1, quantization)
        out.append(
            DetectionRecord(
                category=category,
                bbox_x=int(round(record.bbox_x / q) * q),
                bbox_y=int(round(record.bbox_y / q) * q),
                bbox_w=max(1, int(round(record.bbox_w / q) * q)),
                bbox_h=max(1, int(round(record.bbox_h / q) * q)),
                confidence=record.confidence,
            )
        )
    return out


def build_detector_lightweight_evidence(task: dict[str, str], records: list[DetectionRecord], channel_bin: str, cfg: dict) -> str:
    degraded = degrade_detections_for_channel(records, channel_bin, cfg, task["image_id"])
    class_counts = Counter(record.category for record in degraded)
    target = task.get("target_class", "")
    target_records = [record for record in degraded if record.category == target]
    target_count = len(target_records)
    total_count = len(degraded)
    confidences = [record.confidence for record in target_records]
    mean_conf = statistics.mean(confidences) if confidences else 0.0
    top_conf = max(confidences) if confidences else 0.0
    target_area = sum(record.bbox_w * record.bbox_h for record in target_records)
    all_area = sum(record.bbox_w * record.bbox_h for record in degraded)
    density_hint = "none"
    if target_count >= 20:
        density_hint = "very dense target detections"
    elif target_count >= 8:
        density_hint = "dense target detections"
    elif target_count >= 3:
        density_hint = "several target detections"
    elif target_count >= 1:
        density_hint = "few target detections"
    box_summaries = []
    for record in sorted(target_records, key=lambda item: item.confidence, reverse=True)[:12]:
        box_summaries.append(
            f"{record.category}(x={record.bbox_x}, y={record.bbox_y}, w={record.bbox_w}, h={record.bbox_h}, conf={record.confidence:.2f})"
        )
    count_text = ", ".join(f"{name}:{count}" for name, count in sorted(class_counts.items())) or "none"
    boxes_text = "; ".join(box_summaries) if box_summaries else "no target boxes detected"
    return (
        "You are given lightweight UAV semantic tokens produced by a real object detector, not ground-truth annotations.\n"
        "Use only these semantic tokens. Do not infer objects that are not listed.\n"
        f"image_id: {task['image_id']}\n"
        f"question_type: {task.get('question_type', '')}\n"
        f"target_class: {target}\n"
        f"detector_target_count: {target_count}\n"
        f"detector_total_count: {total_count}\n"
        f"detector_counts_by_class: {count_text}\n"
        f"target_mean_confidence: {mean_conf:.3f}\n"
        f"target_top_confidence: {top_conf:.3f}\n"
        f"target_box_area_sum: {target_area}\n"
        f"all_box_area_sum: {all_area}\n"
        f"density_hint: {density_hint}\n"
        f"top_target_boxes: {boxes_text}\n"
        "Answer format: yes/no for presence; a single integer for counting. "
        "For counting questions, the best answer is detector_target_count unless the tokens clearly say otherwise."
    )


def write_detections_csv(rows: list[dict[str, str]], path: Path) -> None:
    ensure_parent(path)
    fieldnames = ["image_id", "category", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence", "detector_model", "detector_latency_sec"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
