#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except ValueError:
        return default


def _correct(row: dict[str, str]) -> float:
    return 1.0 if str(row.get("correct", "")).lower() == "true" else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _error_bin(abs_error: int) -> str:
    if abs_error == 0:
        return "0"
    if abs_error <= 2:
        return "1-2"
    if abs_error <= 5:
        return "3-5"
    return ">5"


def _gt_count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 9:
        return "4-9"
    return ">=10"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _unique_count_tasks(tasks: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    selected = []
    for task in tasks:
        if task.get("question_type") != "counting":
            continue
        key = (task.get("image_id"), task.get("target_class"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(task)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_detector_qwen.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tasks = _read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    predictions = _read_csv(resolve_path(cfg["paths"]["vlm_predictions_csv"]))
    detections_path = resolve_path(cfg["detector"]["detections_csv"])
    detections = _read_csv(detections_path) if detections_path.exists() else []

    diagnostic_md = resolve_path(
        cfg["paths"].get("detector_diagnostic_md", "outputs/reports/v1_detector_diagnostics.md")
    )
    diagnostic_csv = resolve_path(
        cfg["paths"].get("detector_diagnostic_csv", "outputs/reports/v1_detector_count_diagnostics.csv")
    )

    det_counts: Counter[tuple[str, str]] = Counter()
    det_conf: dict[tuple[str, str], list[float]] = defaultdict(list)
    for det in detections:
        key = (det.get("image_id", ""), det.get("category", ""))
        det_counts[key] += 1
        det_conf[key].append(_float(det.get("confidence")))

    count_rows: list[dict[str, str]] = []
    per_class_errors: dict[str, list[int]] = defaultdict(list)
    for task in _unique_count_tasks(tasks):
        image_id = task.get("image_id", "")
        target_class = task.get("target_class", "")
        gt_count = _int(task.get("object_count"))
        detected_count = int(det_counts[(image_id, target_class)])
        count_error = detected_count - gt_count
        abs_error = abs(count_error)
        per_class_errors[target_class].append(abs_error)
        count_rows.append(
            {
                "image_id": image_id,
                "target_class": target_class,
                "gt_count": str(gt_count),
                "detected_count": str(detected_count),
                "count_error": str(count_error),
                "abs_count_error": str(abs_error),
                "error_bin": _error_bin(abs_error),
                "zero_detection": str(detected_count == 0),
                "under_count": str(detected_count < gt_count),
                "over_count": str(detected_count > gt_count),
                "mean_confidence": f"{_mean(det_conf.get((image_id, target_class), [])):.6f}",
            }
        )

    service1_rows = [row for row in predictions if row.get("service_level") == "1"]
    vqa_by_error_bin: dict[str, list[float]] = defaultdict(list)
    vqa_by_gt_count_bin: dict[str, list[float]] = defaultdict(list)
    vqa_by_channel: dict[str, list[float]] = defaultdict(list)
    detector_failure_rows: list[dict[str, str]] = []
    for row in service1_rows:
        gt_count = _int(row.get("object_count"))
        detector_count = int(det_counts[(row.get("image_id", ""), row.get("target_class", ""))])
        abs_error = abs(detector_count - gt_count)
        bin_name = _error_bin(abs_error)
        vqa_by_error_bin[bin_name].append(_correct(row))
        if row.get("question_type") == "counting":
            vqa_by_gt_count_bin[_gt_count_bin(gt_count)].append(_correct(row))
        vqa_by_channel[row.get("channel_bin", "")].append(_correct(row))
        if row.get("question_type") == "counting" and _correct(row) < 1.0:
            detector_failure_rows.append(
                {
                    "image_id": row.get("image_id", ""),
                    "channel_bin": row.get("channel_bin", ""),
                    "view_quality_bin": row.get("view_quality_bin", ""),
                    "target_class": row.get("target_class", ""),
                    "gt_count": str(gt_count),
                    "detector_target_count": str(detector_count),
                    "transmitted_detector_count": row.get("transmitted_detector_count", ""),
                    "calibrated_detector_count": row.get("calibrated_detector_count", ""),
                    "abs_detector_count_error": str(abs_error),
                    "question": row.get("question", ""),
                    "ground_truth_answer": row.get("ground_truth_answer", ""),
                    "predicted_answer": row.get("predicted_answer", ""),
                    "correct": row.get("correct", ""),
                }
            )

    count_abs_errors = [_int(row["abs_count_error"]) for row in count_rows]
    zero_detection_rate = _mean([1.0 if row["zero_detection"] == "True" else 0.0 for row in count_rows])
    under_rate = _mean([1.0 if row["under_count"] == "True" else 0.0 for row in count_rows])
    over_rate = _mean([1.0 if row["over_count"] == "True" else 0.0 for row in count_rows])
    mae = _mean([float(value) for value in count_abs_errors])

    _write_csv(
        diagnostic_csv,
        [
            "image_id",
            "target_class",
            "gt_count",
            "detected_count",
            "count_error",
            "abs_count_error",
            "error_bin",
            "zero_detection",
            "under_count",
            "over_count",
            "mean_confidence",
        ],
        count_rows,
    )
    _write_csv(
        diagnostic_csv.with_name(diagnostic_csv.stem + "_vqa_failures.csv"),
        [
            "image_id",
            "channel_bin",
            "view_quality_bin",
            "target_class",
            "gt_count",
            "detector_target_count",
            "transmitted_detector_count",
            "calibrated_detector_count",
            "abs_detector_count_error",
            "question",
            "ground_truth_answer",
            "predicted_answer",
            "correct",
        ],
        detector_failure_rows,
    )

    lines = [
        "# Detector-Qwen Diagnostic Report",
        "",
        "This report checks whether detector-generated semantic tokens are reliable enough for VQA service control.",
        "Ground-truth annotations are used only for evaluation; the `s=1` Qwen input is detector output.",
        "",
        "## Detector Count Quality",
        "",
        f"- count diagnostic rows: `{len(count_rows)}`",
        f"- raw detections: `{len(detections)}`",
        f"- count MAE: `{mae:.3f}`",
        f"- zero-detection rate: `{zero_detection_rate:.3f}`",
        f"- under-count ratio: `{under_rate:.3f}`",
        f"- over-count ratio: `{over_rate:.3f}`",
        "",
        "## Per-class Count Error",
        "",
        "| class | samples | count MAE |",
        "|---|---:|---:|",
    ]
    for cls, errors in sorted(per_class_errors.items(), key=lambda item: (-_mean([float(v) for v in item[1]]), item[0])):
        lines.append(f"| {cls} | {len(errors)} | {_mean([float(v) for v in errors]):.3f} |")
    lines.extend([
        "",
        "## Detector Error vs Qwen Correctness for Semantic-token Evidence",
        "",
        "| detector count error bin | Qwen accuracy | samples |",
        "|---|---:|---:|",
    ])
    for bin_name in ["0", "1-2", "3-5", ">5"]:
        values = vqa_by_error_bin.get(bin_name, [])
        lines.append(f"| {bin_name} | {_mean(values):.3f} | {len(values)} |")
    lines.extend([
        "",
        "## Counting Correctness by GT Count Bin",
        "",
        "| GT count bin | semantic-token accuracy | samples |",
        "|---|---:|---:|",
    ])
    for bin_name in ["1", "2-3", "4-9", ">=10"]:
        values = vqa_by_gt_count_bin.get(bin_name, [])
        lines.append(f"| {bin_name} | {_mean(values):.3f} | {len(values)} |")
    lines.extend([
        "",
        "## Semantic-token Qwen Accuracy by Channel",
        "",
        "| channel | accuracy | samples |",
        "|---|---:|---:|",
    ])
    for channel in ["bad", "medium", "good"]:
        values = vqa_by_channel.get(channel, [])
        lines.append(f"| {channel} | {_mean(values):.3f} | {len(values)} |")
    lines.extend(["", "## Counting Failure Examples", ""])
    for row in sorted(detector_failure_rows, key=lambda item: -_int(item["abs_detector_count_error"]))[:15]:
        lines.append(
            f"- `{row['image_id']}` channel={row['channel_bin']} class={row['target_class']} "
            f"GT={row['gt_count']} detector={row['detector_target_count']} "
            f"transmitted={row['transmitted_detector_count']} calibrated={row['calibrated_detector_count']} "
            f"Pred=`{row['predicted_answer']}` Q: {row['question']}"
        )
    if not detector_failure_rows:
        lines.append("- No detector semantic-token counting failures found.")
    lines.append("")
    ensure_parent(diagnostic_md)
    diagnostic_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"detector_diagnostic_md={diagnostic_md}")
    print(f"detector_diagnostic_csv={diagnostic_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
