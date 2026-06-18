#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.run_v1_detector_eval import (  # noqa: E402
    _calibrated_count,
    _counting_class_channel_calibration,
    _detector_target_count,
    _semantic_token_prediction,
)
from vqa_semcom.config import ensure_parent, load_config, resolve_path  # noqa: E402
from vqa_semcom.detector.visdrone_yolo import (  # noqa: E402
    DetectionRecord,
    build_detector_lightweight_evidence,
    degrade_detections_for_channel,
    run_ultralytics_detector,
)
from vqa_semcom.evidence.builder import image_path_for_task, read_tasks_csv, select_vlm_tasks  # noqa: E402


PREDICTION_FIELDS = [
    "threshold",
    "image_id",
    "question_type",
    "presence_polarity",
    "target_class",
    "object_count",
    "channel_bin",
    "view_quality_bin",
    "risk_level",
    "raw_detector_count",
    "transmitted_detector_count",
    "calibrated_detector_count",
    "predicted_answer",
    "correct",
    "raw_correct",
    "payload_bytes",
]

SUMMARY_FIELDS = [
    "threshold",
    "group",
    "value",
    "samples",
    "accuracy",
    "raw_accuracy",
    "mean_payload_kb",
    "mean_gt_count",
    "mean_raw_count",
    "mean_calibrated_count",
]


def _float(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        return float(value if value not in {"", None} else default)
    except (TypeError, ValueError):
        return default


def _truth(value: str | bool | int | float | None) -> float:
    return 1.0 if str(value).lower() in {"1", "true", "yes"} else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _gt_count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 9:
        return "4-9"
    return ">=10"


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_threshold_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        threshold = f"{_float(row.get('threshold')):.2f}"
        grouped[(threshold, "overall", "all")].append(row)
        grouped[(threshold, "question_type", row.get("question_type", ""))].append(row)
        grouped[(threshold, "channel_bin", row.get("channel_bin", ""))].append(row)
        if row.get("question_type") == "presence":
            grouped[(threshold, "presence_polarity", row.get("presence_polarity") or "unknown")].append(row)
        if row.get("question_type") == "counting":
            grouped[(threshold, "gt_count_bin", _gt_count_bin(int(_float(row.get("object_count")))))].append(row)
            grouped[(threshold, "target_class", row.get("target_class", ""))].append(row)

    out: list[dict[str, str]] = []
    for (threshold, group, value), group_rows in sorted(grouped.items()):
        payloads = [_float(row.get("payload_bytes")) / 1024.0 for row in group_rows]
        gt_counts = [_float(row.get("object_count")) for row in group_rows]
        raw_counts = [_float(row.get("transmitted_detector_count")) for row in group_rows]
        calibrated_counts = [_float(row.get("calibrated_detector_count")) for row in group_rows]
        out.append(
            {
                "threshold": threshold,
                "group": group,
                "value": value,
                "samples": str(len(group_rows)),
                "accuracy": f"{_mean([_truth(row.get('correct')) for row in group_rows]):.6f}",
                "raw_accuracy": f"{_mean([_truth(row.get('raw_correct')) for row in group_rows]):.6f}",
                "mean_payload_kb": f"{_mean(payloads):.6f}",
                "mean_gt_count": f"{_mean(gt_counts):.6f}",
                "mean_raw_count": f"{_mean(raw_counts):.6f}",
                "mean_calibrated_count": f"{_mean(calibrated_counts):.6f}",
            }
        )
    return out


def _detect_for_threshold(
    tasks: list[dict[str, str]],
    visdrone_root: Path,
    weights_path: Path,
    conf: float,
    imgsz: int,
) -> dict[str, tuple[list[DetectionRecord], float, str]]:
    cache: dict[str, tuple[list[DetectionRecord], float, str]] = {}
    for image_id in sorted({task["image_id"] for task in tasks}):
        image_path = image_path_for_task(visdrone_root, image_id)
        result = run_ultralytics_detector(image_path, weights_path, conf=conf, imgsz=imgsz)
        cache[image_id] = (result.records, result.latency_sec, result.model_name)
    return cache


def build_threshold_prediction_rows(
    tasks: list[dict[str, str]],
    detection_cache: dict[str, tuple[list[DetectionRecord], float, str]],
    channels: list[str],
    cfg: dict,
    threshold: float,
) -> list[dict[str, str]]:
    ratios = _counting_class_channel_calibration(tasks, detection_cache, channels, cfg)
    min_raw_count = int(cfg.get("vlm", {}).get("count_calibration_min_raw_count", 3))
    rows: list[dict[str, str]] = []
    for task in tasks:
        image_id = task["image_id"]
        records = detection_cache.get(image_id, ([], 0.0, ""))[0]
        raw_count = _detector_target_count(records, task)
        for channel in channels:
            transmitted = degrade_detections_for_channel(records, channel, cfg, image_id)
            transmitted_count = _detector_target_count(transmitted, task)
            calibrated_count = (
                _calibrated_count(
                    transmitted_count,
                    task.get("target_class", ""),
                    channel,
                    ratios,
                    min_raw_count=min_raw_count,
                )
                if task.get("question_type") == "counting"
                else transmitted_count
            )
            predicted, _norm, correct = _semantic_token_prediction(task, calibrated_count, cfg)
            _raw_predicted, _raw_norm, raw_correct = _semantic_token_prediction(task, transmitted_count, cfg)
            evidence = build_detector_lightweight_evidence(task, records, channel, cfg)
            rows.append(
                {
                    "threshold": f"{threshold:.2f}",
                    "image_id": image_id,
                    "question_type": task.get("question_type", ""),
                    "presence_polarity": task.get("presence_polarity", ""),
                    "target_class": task.get("target_class", ""),
                    "object_count": task.get("object_count", "0"),
                    "channel_bin": channel,
                    "view_quality_bin": task.get("view_quality_bin", ""),
                    "risk_level": task.get("risk_level", ""),
                    "raw_detector_count": str(raw_count),
                    "transmitted_detector_count": str(transmitted_count),
                    "calibrated_detector_count": str(calibrated_count),
                    "predicted_answer": predicted,
                    "correct": str(bool(correct)),
                    "raw_correct": str(bool(raw_correct)),
                    "payload_bytes": str(len(evidence.encode("utf-8"))),
                }
            )
    return rows


def _write_markdown(path: Path, summary_rows: list[dict[str, str]], prediction_rows: list[dict[str, str]]) -> None:
    lines = [
        "# V1.8 Detector Threshold Sweep",
        "",
        "This sweep evaluates service level `s=1` semantic-token transmission under different detector confidence thresholds.",
        "The goal is to expose the operating-point tradeoff among positive presence recall, negative presence precision, counting accuracy, and lightweight payload.",
        "",
        "## Summary By Threshold",
        "",
        "| threshold | overall acc | presence acc | positive presence | negative presence | counting acc | payload KB | samples |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {(row["threshold"], row["group"], row["value"]): row for row in summary_rows}
    thresholds = sorted({row["threshold"] for row in summary_rows}, key=float)
    for threshold in thresholds:
        overall = by_key.get((threshold, "overall", "all"), {})
        presence = by_key.get((threshold, "question_type", "presence"), {})
        positive = by_key.get((threshold, "presence_polarity", "positive"), {})
        negative = by_key.get((threshold, "presence_polarity", "negative"), {})
        counting = by_key.get((threshold, "question_type", "counting"), {})
        lines.append(
            f"| {threshold} | {overall.get('accuracy', '-')} | {presence.get('accuracy', '-')} | "
            f"{positive.get('accuracy', '-')} | {negative.get('accuracy', '-')} | "
            f"{counting.get('accuracy', '-')} | {overall.get('mean_payload_kb', '-')} | {overall.get('samples', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Counting By GT Count Bin",
            "",
            "| threshold | count bin | samples | calibrated acc | raw acc | mean GT | mean raw count | mean calibrated count |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        if row["group"] != "gt_count_bin":
            continue
        lines.append(
            f"| {row['threshold']} | {row['value']} | {row['samples']} | {row['accuracy']} | "
            f"{row['raw_accuracy']} | {row['mean_gt_count']} | {row['mean_raw_count']} | {row['mean_calibrated_count']} |"
        )
    lines.extend(["", f"Prediction rows: {len(prediction_rows)}", ""])
    ensure_parent(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(path: Path, summary_rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    by_key = {(row["threshold"], row["group"], row["value"]): row for row in summary_rows}
    thresholds = sorted({row["threshold"] for row in summary_rows}, key=float)
    xs = [float(x) for x in thresholds]
    series = {
        "overall": [float(by_key.get((t, "overall", "all"), {}).get("accuracy", 0.0)) for t in thresholds],
        "positive presence": [float(by_key.get((t, "presence_polarity", "positive"), {}).get("accuracy", 0.0)) for t in thresholds],
        "negative presence": [float(by_key.get((t, "presence_polarity", "negative"), {}).get("accuracy", 0.0)) for t in thresholds],
        "counting": [float(by_key.get((t, "question_type", "counting"), {}).get("accuracy", 0.0)) for t in thresholds],
    }
    ensure_parent(path)
    plt.figure(figsize=(7.2, 4.2))
    for label, ys in series.items():
        plt.plot(xs, ys, marker="o", linewidth=2, label=label)
    plt.xlabel("Detector confidence threshold")
    plt.ylabel("Semantic-token answer accuracy")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_7_quality_calibrated.yaml")
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--out-csv", default="outputs/reports/v1_8_detector_threshold_sweep.csv")
    parser.add_argument("--out-predictions-csv", default="outputs/reports/v1_8_detector_threshold_predictions.csv")
    parser.add_argument("--out-md", default="outputs/reports/v1_8_detector_threshold_sweep.md")
    parser.add_argument("--out-figure", default="outputs/figures/v1_8_detector_threshold_sweep.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    channels = [item.strip() for item in args.channels.split(",")] if args.channels else list(cfg["bins"]["channel"])
    question_types = set(cfg.get("vlm", {}).get("question_types", ["presence", "counting"]))
    tasks = read_tasks_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    selected_tasks = select_vlm_tasks(
        tasks,
        question_types=question_types,
        limit_images=args.limit_images,
        max_tasks=args.max_tasks,
        max_tasks_per_image=int(cfg.get("vlm", {}).get("max_tasks_per_image", 6)),
    )
    if not selected_tasks:
        raise RuntimeError("No tasks selected for V1.8 detector threshold sweep.")

    det_cfg = cfg["detector"]
    visdrone_root = resolve_path(cfg["paths"]["visdrone_val"])
    weights_path = resolve_path(det_cfg["weights_path"])
    imgsz = int(det_cfg.get("imgsz", 640))
    prediction_rows: list[dict[str, str]] = []
    for threshold in thresholds:
        detection_cache = _detect_for_threshold(selected_tasks, visdrone_root, weights_path, threshold, imgsz)
        prediction_rows.extend(build_threshold_prediction_rows(selected_tasks, detection_cache, channels, cfg, threshold))

    summary_rows = summarize_threshold_rows(prediction_rows)
    out_csv = resolve_path(args.out_csv)
    out_predictions = resolve_path(args.out_predictions_csv)
    out_md = resolve_path(args.out_md)
    out_figure = resolve_path(args.out_figure)
    _write_csv(out_predictions, prediction_rows, PREDICTION_FIELDS)
    _write_csv(out_csv, summary_rows, SUMMARY_FIELDS)
    _write_markdown(out_md, summary_rows, prediction_rows)
    _write_plot(out_figure, summary_rows)
    print(f"selected_tasks={len(selected_tasks)} thresholds={len(thresholds)} rows={len(prediction_rows)}")
    print(f"threshold_sweep_csv={out_csv}")
    print(f"threshold_sweep_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
