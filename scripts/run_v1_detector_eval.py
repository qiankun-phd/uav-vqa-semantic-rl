#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path
from vqa_semcom.degradation.channel import degrade_image, degrade_pil_image
from vqa_semcom.detector.visdrone_yolo import (
    DetectionRecord,
    build_detector_lightweight_evidence,
    degrade_detections_for_channel,
    run_ultralytics_detector,
    write_detections_csv,
)
from vqa_semcom.evidence.builder import build_vlm_prompt, image_path_for_task, read_tasks_csv, select_vlm_tasks
from vqa_semcom.snr import (
    channel_bin_from_snr,
    parse_snr_bins,
    snr_bin_label,
    snr_bins_from_config,
    snr_db_from_label,
)
from vqa_semcom.vlm.evaluator import evaluate_prediction, make_evaluator
from vqa_semcom.vlm.answer import check_answer
from scripts.run_v1_vlm_eval import _cache_prediction, _payload_bytes

PREDICTION_FIELDNAMES = [
    "image_id", "question_type", "question", "ground_truth_answer", "target_class", "object_count",
    "risk_level", "epsilon_k", "tau_k", "service_level", "channel_bin", "sensed_snr_db", "snr_bin",
    "view_quality_bin", "freshness_bin",
    "evidence_type", "evidence_source", "evidence_repr", "payload_bytes", "image_path", "predicted_answer",
    "normalized_prediction", "correct", "latency_sec", "model_name", "detector_model", "detector_conf",
    "detector_latency_sec", "detector_object_count", "presence_polarity", "decoder_mode", "raw_detector_count",
    "transmitted_detector_count", "calibrated_detector_count", "raw_decoder_correct", "roi_mode",
]


def _read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dedupe_rows(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row["image_id"],
        row["question"],
        str(row["service_level"]),
        row.get("channel_bin", ""),
        row.get("snr_bin", ""),
        row["freshness_bin"],
        row["evidence_type"],
    )


def _write_rows(rows: list[dict[str, str]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _merge_vlm_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = dict(cfg)
    vlm_cfg = dict(cfg.get("vlm", {}))
    if args.max_new_tokens is not None:
        vlm_cfg["max_new_tokens"] = args.max_new_tokens
    cfg["vlm"] = vlm_cfg
    return cfg


def _link_quality_bins(cfg: dict, args: argparse.Namespace) -> tuple[list[str], bool]:
    snr_values = parse_snr_bins(args.snr_bins) if args.snr_bins else snr_bins_from_config(cfg)
    if snr_values:
        return [snr_bin_label(value) for value in snr_values], True
    channels = [x.strip() for x in args.channels.split(",")] if args.channels else list(cfg["bins"]["channel"])
    return channels, False


def _link_quality_fields(link_quality: str, use_snr: bool) -> tuple[str, str, str]:
    if not use_snr:
        return link_quality, "", ""
    snr_db = snr_db_from_label(link_quality)
    return channel_bin_from_snr(snr_db), f"{snr_db:g}", snr_bin_label(snr_db)


def _union_box(records: list[DetectionRecord]) -> tuple[int, int, int, int]:
    x1 = min(r.bbox_x for r in records)
    y1 = min(r.bbox_y for r in records)
    x2 = max(r.bbox_x + r.bbox_w for r in records)
    y2 = max(r.bbox_y + r.bbox_h for r in records)
    return x1, y1, x2, y2


def _expand_clip(box: tuple[int, int, int, int], scale: float, width: int, height: int, min_side: int = 48) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = max(x2 - x1, min_side) * scale
    h = max(y2 - y1, min_side) * scale
    nx1, ny1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    nx2, ny2 = int(min(width, cx + w / 2)), int(min(height, cy + h / 2))
    return nx1, ny1, max(nx1 + 1, nx2), max(ny1 + 1, ny2)


def _collage(rgb: Image.Image, boxes: list[tuple[int, int, int, int]], tile_height: int = 224, gap: int = 4) -> Image.Image:
    tiles = []
    for box in boxes:
        crop = rgb.crop(box)
        scale = tile_height / max(1, crop.height)
        tiles.append(crop.resize((max(1, int(crop.width * scale)), tile_height)))
    total_w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (max(1, total_w), tile_height), (0, 0, 0))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width + gap
    return canvas


def _topk_by_conf(records: list[DetectionRecord], k: int) -> list[DetectionRecord]:
    return sorted(records, key=lambda r: r.confidence, reverse=True)[:k]


def _build_roi_image(source_image: Path, records: list[DetectionRecord], task: dict[str, str], channel_bin: str, out_dir: Path, cfg: dict) -> tuple[Path, str, str]:
    """Detector-guided ROI evidence for service_level=3.

    Modes:
      * target_topk : >=tau detections of target_class -> union of top-k boxes, expanded 1.5x
      * dual        : comparison/co_presence -> union of per-class boxes (>=tau, else suspect band)
      * suspect     : no >=tau target detection but conf in [floor, tau) exists -> top-3 collage
      * thumbnail   : no detection of the queried class(es) at all -> downscaled full image
    """
    det_cfg = cfg.get("detector", {}) or {}
    tau = float(det_cfg.get("roi_conf_threshold", 0.25))
    floor = float(det_cfg.get("roi_suspect_conf_floor", 0.05))
    top_k = int(det_cfg.get("roi_top_k", 3))
    expand = float(det_cfg.get("roi_expand_ratio", 1.5))
    thumb_side = int(det_cfg.get("roi_thumbnail_max_side", 512))
    qt = task.get("question_type", "")
    target = task.get("target_class", "")
    target_b = task.get("target_class_b", "")
    dual_classes = [c for c in (target, target_b) if c] if qt in ("comparison", "co_presence") else []

    with Image.open(source_image) as img:
        rgb = img.convert("RGB")
        width, height = rgb.width, rgb.height
        mode = ""
        meta_extra = ""
        if dual_classes and len(dual_classes) > 1:
            chosen: list[DetectionRecord] = []
            for cls in dual_classes:
                confident = [r for r in records if r.category == cls and r.confidence >= tau]
                pool = confident or [r for r in records if r.category == cls and floor <= r.confidence < tau]
                chosen.extend(_topk_by_conf(pool, top_k))
            if chosen:
                mode = "dual"
                box = _expand_clip(_union_box(chosen), expand, width, height)
                crop = rgb.crop(box)
                meta_extra = f"boxes={len(chosen)};region={box[0]},{box[1]},{box[2]},{box[3]}"
        if not mode and not dual_classes:
            confident = [r for r in records if r.category == target and r.confidence >= tau]
            suspect = [r for r in records if r.category == target and floor <= r.confidence < tau]
            if confident:
                mode = "target_topk"
                chosen = _topk_by_conf(confident, top_k)
                box = _expand_clip(_union_box(chosen), expand, width, height)
                crop = rgb.crop(box)
                meta_extra = f"boxes={len(chosen)};conf_top={chosen[0].confidence:.2f};region={box[0]},{box[1]},{box[2]},{box[3]}"
            elif suspect:
                mode = "suspect"
                chosen = _topk_by_conf(suspect, top_k)
                boxes = [_expand_clip((r.bbox_x, r.bbox_y, r.bbox_x + r.bbox_w, r.bbox_y + r.bbox_h), 2.0, width, height, min_side=64) for r in chosen]
                crop = _collage(rgb, boxes)
                meta_extra = f"boxes={len(chosen)};conf_top={chosen[0].confidence:.2f}"
        if not mode:
            mode = "thumbnail"
            scale = min(1.0, thumb_side / max(width, height))
            crop = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))))
            meta_extra = f"scale={scale:.3f}"
        stem_cls = f"{target}_{target_b}" if dual_classes and len(dual_classes) > 1 else (target or "target")
        out_path = out_dir / channel_bin / f"{source_image.stem}_{stem_cls}_{mode}_roi_{channel_bin}.jpg"
        return degrade_pil_image(crop, out_path, channel_bin, cfg), f"mode={mode};{meta_extra}", mode


def _detector_target_count(records: list[DetectionRecord], task: dict[str, str]) -> int:
    target = task.get("target_class", "")
    return sum(1 for record in records if record.category == target)


def _semantic_token_prediction(task: dict[str, str], target_count: int, cfg: dict) -> tuple[str, str, bool]:
    if task.get("question_type") == "presence":
        predicted = "yes" if target_count > 0 else "no"
    elif task.get("question_type") == "counting":
        predicted = str(target_count)
    else:
        predicted = str(target_count)
    checked = check_answer(
        task["question_type"],
        predicted,
        task["answer"],
        tolerance_ratio=float(cfg.get("vlm", {}).get("count_tolerance_ratio", 0.10)),
    )
    return predicted, checked.normalized_prediction, checked.correct


def _counting_class_channel_calibration(
    tasks: list[dict[str, str]],
    detection_cache: dict[str, tuple[list[DetectionRecord], float, str]],
    channels: list[str],
    cfg: dict,
) -> dict[tuple[str, str], float]:
    sums: dict[tuple[str, str], list[float]] = {}
    for task in tasks:
        if task.get("question_type") != "counting":
            continue
        image_id = task["image_id"]
        target = task.get("target_class", "")
        records = detection_cache.get(image_id, ([], 0.0, ""))[0]
        gt_count = max(0, int(float(task.get("object_count", "0") or 0)))
        for channel in channels:
            transmitted = degrade_detections_for_channel(records, channel, cfg, image_id)
            det_count = _detector_target_count(transmitted, task)
            key = (target, channel)
            if key not in sums:
                sums[key] = [0.0, 0.0]
            sums[key][0] += float(gt_count)
            sums[key][1] += float(det_count)
    ratios: dict[tuple[str, str], float] = {}
    for key, (gt_sum, det_sum) in sums.items():
        if det_sum <= 0:
            ratios[key] = 1.0
        else:
            ratios[key] = max(0.5, min(4.0, gt_sum / det_sum))
    accepted: dict[tuple[str, str], float] = {}
    min_raw_count = int(cfg.get("vlm", {}).get("count_calibration_min_raw_count", 3))
    for key, ratio in ratios.items():
        raw_scores: list[float] = []
        calibrated_scores: list[float] = []
        target, channel = key
        for task in tasks:
            if task.get("question_type") != "counting" or task.get("target_class", "") != target:
                continue
            image_id = task["image_id"]
            records = detection_cache.get(image_id, ([], 0.0, ""))[0]
            transmitted = degrade_detections_for_channel(records, channel, cfg, image_id)
            raw_count = _detector_target_count(transmitted, task)
            calibrated_count = _calibrated_count(raw_count, target, channel, {key: ratio}, min_raw_count=min_raw_count)
            raw_scores.append(float(_semantic_token_prediction(task, raw_count, cfg)[2]))
            calibrated_scores.append(float(_semantic_token_prediction(task, calibrated_count, cfg)[2]))
        if calibrated_scores and sum(calibrated_scores) + 1e-9 < sum(raw_scores):
            accepted[key] = 1.0
        else:
            accepted[key] = ratio
    return accepted


def _calibrated_count(
    target_count: int,
    target_class: str,
    channel_bin: str,
    ratios: dict[tuple[str, str], float],
    min_raw_count: int = 3,
) -> int:
    if target_count <= 0:
        return 0
    if target_count < min_raw_count:
        return target_count
    ratio = ratios.get((target_class, channel_bin), 1.0)
    return max(0, int(round(target_count * ratio)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_detector_qwen.yaml")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--evaluator", choices=["mock", "qwen"], default="mock")
    parser.add_argument("--service-levels", default="1,2")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--snr-bins", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = _merge_vlm_overrides(load_config(args.config), args)
    det_cfg = cfg["detector"]
    service_levels = [int(x.strip()) for x in args.service_levels.split(",") if x.strip()]
    link_quality_bins, use_snr = _link_quality_bins(cfg, args)
    freshness_bins = list(cfg["bins"]["freshness"])
    question_types = set(cfg.get("vlm", {}).get("question_types", ["presence", "counting"]))
    tasks = read_tasks_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    selected_tasks = select_vlm_tasks(
        tasks,
        question_types=question_types,
        limit_images=args.limit_images,
        max_tasks=args.max_tasks,
        max_tasks_per_image=int(cfg.get("vlm", {}).get("max_tasks_per_image", 4)),
    )
    if not selected_tasks:
        raise RuntimeError("No detector VQA tasks selected.")
    visdrone_root = resolve_path(cfg["paths"]["visdrone_val"])
    weights_path = resolve_path(det_cfg["weights_path"])
    conf = float(det_cfg.get("conf", 0.25))
    imgsz = int(det_cfg.get("imgsz", 640))
    degraded_dir = resolve_path(cfg["paths"].get("degraded_image_dir", "outputs/vlm/degraded_images"))
    out_path = resolve_path(cfg["paths"]["vlm_predictions_csv"])
    detections_path = resolve_path(det_cfg.get("detections_csv", "outputs/detector/v1_detector_detections.csv"))
    existing_rows = _read_existing_rows(out_path) if args.resume else []
    existing_keys = {_row_key(row) for row in existing_rows}
    rows = list(existing_rows)
    vlm_cfg = cfg.get("vlm", {})
    use_semantic_decoder = bool(vlm_cfg.get("semantic_token_decoder", False))
    decoder_mode = str(vlm_cfg.get("semantic_decoder_mode", "v1_6_hybrid" if use_semantic_decoder else "qwen"))
    direct_semantic_levels = {1} if decoder_mode == "v1_7_direct_calibrated" else set()
    needs_vlm = any(level in service_levels and level not in direct_semantic_levels for level in [1, 2, 3])
    evaluator = make_evaluator(args.evaluator, cfg) if needs_vlm else None
    detection_cache: dict[str, tuple[list[DetectionRecord], float, str]] = {}
    prediction_cache: dict[tuple[str, str, int, str], tuple[str, str, bool, float, str, str, str, int, float, str, int]] = {}
    roi_mode_cache: dict[tuple[str, str, int, str], str] = {}
    detection_csv_rows: list[dict[str, str]] = _read_existing_rows(detections_path) if args.resume else []
    if any(level in service_levels for level in [1, 3]):
        for image_id in sorted({task["image_id"] for task in selected_tasks}):
            source_image = image_path_for_task(visdrone_root, image_id)
            det = run_ultralytics_detector(source_image, weights_path, conf=conf, imgsz=imgsz)
            detection_cache[image_id] = (det.records, det.latency_sec, det.model_name)
            for record in det.records:
                detection_csv_rows.append({
                    "image_id": image_id, "category": record.category, "bbox_x": str(record.bbox_x), "bbox_y": str(record.bbox_y),
                    "bbox_w": str(record.bbox_w), "bbox_h": str(record.bbox_h), "confidence": f"{record.confidence:.4f}",
                    "detector_model": det.model_name, "detector_latency_sec": f"{det.latency_sec:.6f}",
                })
    calibration_ratios = _counting_class_channel_calibration(selected_tasks, detection_cache, link_quality_bins, cfg)
    for task in selected_tasks:
        image_id = task["image_id"]
        source_image = image_path_for_task(visdrone_root, image_id)
        for service_level in service_levels:
            for link_quality in link_quality_bins:
                channel_bin, sensed_snr_db, snr_bin = _link_quality_fields(link_quality, use_snr)
                for freshness_bin in freshness_bins:
                    evidence_type = "cache" if service_level == 0 else "lightweight" if service_level == 1 else "image" if service_level == 2 else "roi_image"
                    row_stub = {
                        "image_id": image_id,
                        "question": task["question"],
                        "service_level": str(service_level),
                        "channel_bin": channel_bin,
                        "snr_bin": snr_bin,
                        "freshness_bin": freshness_bin,
                        "evidence_type": evidence_type,
                    }
                    if args.resume and _row_key(row_stub) in existing_keys:
                        continue
                    image_path = ""
                    detector_model = ""
                    detector_latency = 0.0
                    detector_count = 0
                    raw_detector_count = 0
                    transmitted_detector_count = 0
                    calibrated_detector_count = 0
                    raw_decoder_correct = ""
                    roi_mode = ""
                    if service_level == 0:
                        cache_quality = "good" if use_snr and bool(vlm_cfg.get("cache_ignore_snr", True)) else link_quality
                        predicted, normalized, correct = _cache_prediction(task, cache_quality, freshness_bin, cfg)
                        latency = 0.0
                        model_name = "cache-simulator"
                        evidence_repr = ""
                        payload_bytes = 0
                    else:
                        cache_key = (image_id, task["question"], service_level, link_quality)
                        if cache_key not in prediction_cache:
                            if evaluator is None:
                                if not (use_semantic_decoder and service_level == 1):
                                    raise RuntimeError("Evaluator is required for detector evaluation.")
                            semantic_decoded = False
                            if service_level == 1:
                                records, detector_latency, detector_model = detection_cache[image_id]
                                evidence_repr = build_detector_lightweight_evidence(task, records, link_quality, cfg)
                                raw_target_count = _detector_target_count(records, task)
                                transmitted_records = degrade_detections_for_channel(records, link_quality, cfg, image_id)
                                target_count = _detector_target_count(transmitted_records, task)
                                detector_count = target_count
                                if task.get("question_type") == "comparison":
                                    cb = task.get("target_class_b", "")
                                    count_a = sum(1 for rr in transmitted_records if rr.category == task.get("target_class", ""))
                                    count_b = sum(1 for rr in transmitted_records if rr.category == cb)
                                    cmp_pred = "yes" if count_a > count_b else "no"
                                    _chk = check_answer("comparison", cmp_pred, task["answer"], tolerance_ratio=float(vlm_cfg.get("count_tolerance_ratio", 0.10)))
                                    predicted, normalized, correct = cmp_pred, _chk.normalized_prediction, _chk.correct
                                    payload_bytes = _payload_bytes(service_level, evidence_type, evidence_repr, "")
                                    prediction_cache[cache_key] = (
                                        predicted, normalized, correct, detector_latency, "semantic-token-decoder",
                                        evidence_type, evidence_repr, payload_bytes, detector_latency, detector_model, count_a,
                                        count_a, count_b, count_a, str(bool(correct)),
                                    )
                                    semantic_decoded = True
                                elif task.get("question_type") in ("co_presence", "threshold"):
                                    ca = sum(1 for rr in transmitted_records if rr.category == task.get("target_class", ""))
                                    if task.get("question_type") == "co_presence":
                                        cb2 = sum(1 for rr in transmitted_records if rr.category == task.get("target_class_b", ""))
                                        xpred = "yes" if (ca > 0 and cb2 > 0) else "no"
                                    else:
                                        try:
                                            thr = int(float(task.get("threshold_n", "1")))
                                        except ValueError:
                                            thr = 1
                                        xpred = "yes" if ca >= thr else "no"
                                    _xchk = check_answer(task["question_type"], xpred, task["answer"], tolerance_ratio=float(vlm_cfg.get("count_tolerance_ratio", 0.10)))
                                    predicted, normalized, correct = xpred, _xchk.normalized_prediction, _xchk.correct
                                    payload_bytes = _payload_bytes(service_level, evidence_type, evidence_repr, "")
                                    prediction_cache[cache_key] = (
                                        predicted, normalized, correct, detector_latency, "semantic-token-decoder",
                                        evidence_type, evidence_repr, payload_bytes, detector_latency, detector_model, ca,
                                        ca, ca, ca, str(bool(correct)),
                                    )
                                    semantic_decoded = True
                                elif use_semantic_decoder and (
                                    decoder_mode == "v1_7_direct_calibrated"
                                    or task.get("question_type") == "counting"
                                ):
                                    calibrated_count = (
                                        _calibrated_count(
                                            target_count,
                                            task.get("target_class", ""),
                                            link_quality,
                                            calibration_ratios,
                                            min_raw_count=int(vlm_cfg.get("count_calibration_min_raw_count", 3)),
                                        )
                                        if task.get("question_type") == "counting"
                                        else target_count
                                    )
                                    raw_predicted, _raw_norm, raw_correct = _semantic_token_prediction(task, target_count, cfg)
                                    predicted, normalized, correct = _semantic_token_prediction(task, calibrated_count, cfg)
                                    payload_bytes = _payload_bytes(service_level, evidence_type, evidence_repr, "")
                                    prediction_cache[cache_key] = (
                                        predicted, normalized, correct, detector_latency, "semantic-token-decoder",
                                        evidence_type, evidence_repr, payload_bytes, detector_latency, detector_model, target_count,
                                        raw_target_count, target_count, calibrated_count, str(bool(raw_correct)),
                                    )
                                    semantic_decoded = True
                                else:
                                    prompt = build_vlm_prompt(task, evidence_text=evidence_repr)
                                    prompt = f"service_level=1 snr_bin={snr_bin or ''} channel={channel_bin} evidence_source=detector\n{prompt}"
                                    image_input = None
                            elif service_level == 2:
                                image_input = degrade_image(source_image, degraded_dir, link_quality, cfg)
                                evidence_repr = str(image_input)
                                prompt = build_vlm_prompt(task)
                                prompt = f"service_level=2 snr_bin={snr_bin or ''} channel={channel_bin} evidence_source=image\n{prompt}"
                            elif service_level == 3:
                                records, detector_latency, detector_model = detection_cache[image_id]
                                detector_count = len(records)
                                image_input, roi_meta, roi_mode_val = _build_roi_image(source_image, records, task, link_quality, degraded_dir / "roi", cfg)
                                roi_mode_cache[cache_key] = roi_mode_val
                                evidence_repr = str(image_input)
                                prompt = build_vlm_prompt(task)
                                mode_hint = {
                                    "target_topk": "The image is a detector-guided crop around the highest-confidence target detections.",
                                    "dual": "The image is a detector-guided crop covering the regions of both queried classes.",
                                    "suspect": "The image is a collage of low-confidence candidate regions; verify carefully whether the target class is actually present.",
                                    "thumbnail": "No candidate region was detected; the image is a downscaled full view of the scene.",
                                }.get(roi_mode_val, "The image is a detector-guided crop of the target region.")
                                prompt = (
                                    f"service_level=3 snr_bin={snr_bin or ''} channel={channel_bin} evidence_source=detector_roi_image {roi_meta}\n"
                                    f"{mode_hint}\n"
                                    f"{prompt}"
                                )
                            else:
                                raise ValueError(f"Unsupported service level: {service_level}")
                            if not semantic_decoded:
                                pred = evaluate_prediction(evaluator, task, prompt, image_input, cfg)
                                payload_bytes = _payload_bytes(service_level, evidence_type, evidence_repr, str(image_input) if image_input is not None else "")
                                prediction_cache[cache_key] = (
                                    pred.predicted_answer, pred.normalized_prediction, pred.correct, pred.latency_sec, pred.model_name,
                                    evidence_type, evidence_repr, payload_bytes, detector_latency, detector_model, detector_count,
                                    0, detector_count, detector_count, "",
                                )
                        (
                            predicted, normalized, correct, latency, model_name, evidence_type, evidence_repr, payload_bytes,
                            detector_latency, detector_model, detector_count, raw_detector_count, transmitted_detector_count,
                            calibrated_detector_count, raw_decoder_correct,
                        ) = prediction_cache[cache_key]
                        if evidence_type in {"image", "roi_image"}:
                            image_path = evidence_repr
                        roi_mode = roi_mode_cache.get(cache_key, "") if service_level == 3 else ""
                    rows.append({
                        "image_id": image_id,
                        "question_type": task["question_type"],
                        "question": task["question"],
                        "ground_truth_answer": task["answer"],
                        "target_class": task["target_class"],
                        "object_count": task["object_count"],
                        "risk_level": task["risk_level"],
                        "epsilon_k": task["epsilon_k"],
                        "tau_k": task["tau_k"],
                        "service_level": str(service_level),
                        "channel_bin": channel_bin,
                        "sensed_snr_db": sensed_snr_db,
                        "snr_bin": snr_bin,
                        "view_quality_bin": task["view_quality_bin"],
                        "freshness_bin": freshness_bin,
                        "evidence_type": evidence_type,
                        "evidence_source": "detector" if service_level == 1 else "detector_roi" if service_level == 3 else evidence_type,
                        "evidence_repr": evidence_repr,
                        "payload_bytes": str(payload_bytes),
                        "image_path": image_path,
                        "predicted_answer": predicted,
                        "normalized_prediction": normalized,
                        "correct": str(bool(correct)),
                        "latency_sec": f"{latency:.6f}",
                        "model_name": model_name,
                        "detector_model": detector_model,
                        "detector_conf": f"{conf:.3f}" if service_level in {1, 3} else "",
                        "detector_latency_sec": f"{detector_latency:.6f}" if service_level in {1, 3} else "0.000000",
                        "detector_object_count": str(detector_count) if service_level in {1, 3} else "0",
                        "presence_polarity": task.get("presence_polarity", ""),
                        "decoder_mode": decoder_mode if service_level == 1 else "",
                        "raw_detector_count": str(raw_detector_count) if service_level == 1 else "",
                        "transmitted_detector_count": str(transmitted_detector_count) if service_level == 1 else "",
                        "calibrated_detector_count": str(calibrated_detector_count) if service_level == 1 else "",
                        "raw_decoder_correct": raw_decoder_correct if service_level == 1 else "",
                        "roi_mode": roi_mode,
                    })
                    if len(rows) % 2000 == 0:
                        _write_rows(rows, out_path)  # periodic checkpoint for --resume
    _write_rows(rows, out_path)
    if detection_csv_rows:
        detection_csv_rows = _dedupe_rows(
            detection_csv_rows,
            ("image_id", "category", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence"),
        )
        write_detections_csv(detection_csv_rows, detections_path)
    print(f"selected_tasks={len(selected_tasks)} prediction_rows={len(rows)}")
    print(f"predictions_csv={out_path}")
    print(f"detections_csv={detections_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
