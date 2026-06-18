#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _correct(row: dict[str, str], field: str = "correct") -> float:
    return 1.0 if str(row.get(field, "")).lower() == "true" else 0.0


def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except ValueError:
        return default


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


def _add_group(rows: list[dict[str, str]], group_name: str, group_value: str, out: list[dict[str, str]]) -> None:
    correct = [_correct(row) for row in rows]
    raw_correct = [_correct(row, "raw_decoder_correct") for row in rows if row.get("raw_decoder_correct") != ""]
    raw_counts = [_int(row.get("transmitted_detector_count")) for row in rows if row.get("transmitted_detector_count") != ""]
    calibrated_counts = [_int(row.get("calibrated_detector_count")) for row in rows if row.get("calibrated_detector_count") != ""]
    gt_counts = [_int(row.get("object_count")) for row in rows]
    out.append(
        {
            "group": group_name,
            "value": group_value,
            "samples": str(len(rows)),
            "accuracy": f"{_mean(correct):.6f}",
            "raw_decoder_accuracy": f"{_mean(raw_correct):.6f}" if raw_correct else "",
            "mean_gt_count": f"{_mean([float(x) for x in gt_counts]):.6f}" if gt_counts else "",
            "mean_raw_count": f"{_mean([float(x) for x in raw_counts]):.6f}" if raw_counts else "",
            "mean_calibrated_count": f"{_mean([float(x) for x in calibrated_counts]):.6f}" if calibrated_counts else "",
        }
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent(path)
    fieldnames = [
        "group",
        "value",
        "samples",
        "accuracy",
        "raw_decoder_accuracy",
        "mean_gt_count",
        "mean_raw_count",
        "mean_calibrated_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_7_quality_calibrated.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    predictions = _read_csv(resolve_path(cfg["paths"]["vlm_predictions_csv"]))
    if not predictions:
        raise RuntimeError("No predictions found for V1.7 quality report.")
    service1 = [row for row in predictions if row.get("service_level") == "1"]
    report_rows: list[dict[str, str]] = []

    by_qtype: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_presence: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_count_bin: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_channel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in service1:
        by_qtype[row.get("question_type", "")].append(row)
        by_channel[row.get("channel_bin", "")].append(row)
        if row.get("question_type") == "presence":
            by_presence[row.get("presence_polarity", "unknown") or "unknown"].append(row)
        if row.get("question_type") == "counting":
            by_count_bin[_gt_count_bin(_int(row.get("object_count")))].append(row)
            by_class[row.get("target_class", "")].append(row)

    _add_group(service1, "service1", "all", report_rows)
    for key in sorted(by_qtype):
        _add_group(by_qtype[key], "question_type", key, report_rows)
    for key in ["positive", "negative", "unknown"]:
        if key in by_presence:
            _add_group(by_presence[key], "presence_polarity", key, report_rows)
    for key in ["1", "2-3", "4-9", ">=10"]:
        if key in by_count_bin:
            _add_group(by_count_bin[key], "gt_count_bin", key, report_rows)
    for key in ["bad", "medium", "good"]:
        if key in by_channel:
            _add_group(by_channel[key], "channel_bin", key, report_rows)
    for key, rows in sorted(by_class.items()):
        _add_group(rows, "target_class", key, report_rows)

    md_path = resolve_path(cfg["paths"].get("v1_7_quality_report_md", "outputs/reports/v1_7_quality_report.md"))
    csv_path = resolve_path(cfg["paths"].get("v1_7_quality_report_csv", "outputs/reports/v1_7_quality_report.csv"))
    _write_csv(csv_path, report_rows)
    lines = [
        "# V1.7 Semantic Quality Calibration Report",
        "",
        "This report focuses on `s=1` detector semantic-token decoding.",
        "Presence includes both positive and negative questions; counting reports raw and calibrated detector-count decoding.",
        "",
        "| group | value | samples | accuracy | raw decoder accuracy | mean GT | mean raw count | mean calibrated count |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['group']} | {row['value']} | {row['samples']} | {row['accuracy']} | "
            f"{row['raw_decoder_accuracy'] or '-'} | {row['mean_gt_count'] or '-'} | "
            f"{row['mean_raw_count'] or '-'} | {row['mean_calibrated_count'] or '-'} |"
        )
    lines.append("")
    ensure_parent(md_path)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"v1_7_quality_report_md={md_path}")
    print(f"v1_7_quality_report_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
