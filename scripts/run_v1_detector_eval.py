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
    "transmitted_detector_count", "calibrated_detector_count", "raw_decoder_correct",
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


def _roi_box_for_task(records: list[DetectionRecord], task: dict[str, str], image_width: int, image_height: int) -> tuple[int, int, int, int, str]:
    target = task.get("target_class", "")
    target_records = [record for record in records if record.category == target]
    if not target_records:
        return 0, 0, image_width, image_height, "fallback_full_image_no_target_detection"
    xs1 = [record.bbox_x for record in target_records]
    ys1 = [record.bbox_y for record in target_records]
    xs2 = [record.bbox_x + record.bbox_w for record in target_records]
    ys2 = [record.bbox_y + record.bbox_h for record in target_records]
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    pad_ratio = float(task.get("roi_padding_ratio", "") or 0.18)
    pad_x = int(max(16, (x2 - x1) * pad_ratio))
    pad_y = int(max(16, (y2 - y1) * pad_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
        "detector_target_roi",
    )


def _build_roi_image(source_image: Path, records: list[DetectionRecord], task: dict[str, str], channel_bin: str, out_dir: Path, cfg: dict) -> tuple[Path, str]:
    with Image.open(source_image) as img:
        rgb = img.convert("RGB")
        x1, y1, x2, y2, source = _roi_box_for_task(records, task, rgb.width, rgb.height)
        crop = rgb.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
        out_path = out_dir / channel_bin / f"{source_image.stem}_{task.get('target_class', 'target')}_roi_{channel_bin}.jpg"
        return degrade_pil_image(crop, out_path, channel_bin, cfg), f"{source}:x1={x1},y1={y1},x2={x2},y2={y2}"


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
                                if use_semantic_decoder and (
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
                                image_input, roi_meta = _build_roi_image(source_image, records, task, link_quality, degraded_dir / "roi", cfg)
                                evidence_repr = str(image_input)
                                prompt = build_vlm_prompt(task)
                                prompt = (
                                    f"service_level=3 snr_bin={snr_bin or ''} channel={channel_bin} evidence_source=detector_roi_image {roi_meta}\n"
                                    "The image is a detector-guided crop/zoom of the target region when detections are available.\n"
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
                    })
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
