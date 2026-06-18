#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path  # noqa: E402


POLICY_FIELDS = [
    "protocol",
    "role",
    "policy",
    "task_success_rate",
    "average_accuracy",
    "average_delay",
    "average_energy",
    "average_payload_kb",
    "payload_reduction_vs_always_image",
    "quality_violation_rate",
    "deadline_violation_rate",
    "service_level_0_ratio",
    "service_level_1_ratio",
    "service_level_2_ratio",
    "service_level_3_ratio",
]

QUALITY_FIELDS = [
    "protocol",
    "group",
    "value",
    "samples",
    "accuracy",
    "mean_payload_kb",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        return float(value if value not in {"", None} else default)
    except (TypeError, ValueError):
        return default


def _truth(value: str | bool | int | float | None) -> float:
    return 1.0 if str(value).lower() in {"1", "true", "yes"} else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _normal_ci(rate: float, n: int) -> tuple[float, float]:
    n = max(1, n)
    width = 1.96 * math.sqrt(max(0.0, rate * (1.0 - rate)) / n)
    return max(0.0, rate - width), min(1.0, rate + width)


def _format_ci(rate: float, n: int) -> str:
    lo, hi = _normal_ci(rate, n)
    return f"{rate:.3f} [{lo:.3f}, {hi:.3f}]"


def _protocol_config_rows(config_path: str, protocol: str, role: str) -> tuple[list[dict[str, str]], Path, Path]:
    cfg = load_config(config_path)
    sim_path = resolve_path(cfg["paths"]["sim_results_csv"])
    pred_path = resolve_path(cfg["paths"]["vlm_predictions_csv"])
    rows: list[dict[str, str]] = []
    for row in _read_csv(sim_path):
        item = {field: row.get(field, "") for field in POLICY_FIELDS}
        item["protocol"] = protocol
        item["role"] = role
        rows.append(item)
    return rows, sim_path, pred_path


def collect_policy_rows(
    protocol_a_config: str,
    protocol_b_config: str,
    roi_config: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, Path]]:
    rows: list[dict[str, str]] = []
    paths: dict[str, Path] = {}
    a_rows, a_sim, a_pred = _protocol_config_rows(protocol_a_config, "Protocol-A", "operational positive-query")
    b_rows, b_sim, b_pred = _protocol_config_rows(protocol_b_config, "Protocol-B", "balanced presence calibration")
    rows.extend(a_rows)
    rows.extend(b_rows)
    paths["Protocol-A sim"] = a_sim
    paths["Protocol-A predictions"] = a_pred
    paths["Protocol-B sim"] = b_sim
    paths["Protocol-B predictions"] = b_pred
    if roi_config:
        roi_rows, roi_sim, roi_pred = _protocol_config_rows(roi_config, "Protocol-B+ROI", "balanced calibration with crop/tile image evidence")
        rows.extend(roi_rows)
        paths["Protocol-B+ROI sim"] = roi_sim
        paths["Protocol-B+ROI predictions"] = roi_pred
    return rows, paths


def _gt_count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 9:
        return "4-9"
    return ">=10"


def _add_quality_group(protocol: str, group: str, value: str, rows: list[dict[str, str]], out: list[dict[str, str]]) -> None:
    if not rows:
        return
    out.append(
        {
            "protocol": protocol,
            "group": group,
            "value": value,
            "samples": str(len(rows)),
            "accuracy": f"{_mean([_truth(row.get('correct')) for row in rows]):.6f}",
            "mean_payload_kb": f"{_mean([_float(row.get('payload_bytes')) / 1024.0 for row in rows]):.6f}",
        }
    )


def collect_quality_rows(protocol_predictions: list[tuple[str, Path]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for protocol, path in protocol_predictions:
        rows = _read_csv(path)
        if not rows:
            continue
        by_service = defaultdict(list)
        by_qtype = defaultdict(list)
        by_presence = defaultdict(list)
        by_count_bin = defaultdict(list)
        for row in rows:
            by_service[row.get("service_level", "")].append(row)
            by_qtype[row.get("question_type", "")].append(row)
            if row.get("question_type") == "presence":
                by_presence[row.get("presence_polarity", "") or "unknown"].append(row)
            if row.get("question_type") == "counting":
                by_count_bin[_gt_count_bin(int(_float(row.get("object_count"))))].append(row)
        _add_quality_group(protocol, "all", "all", rows, out)
        for key in sorted(by_service, key=lambda x: int(x) if str(x).isdigit() else 99):
            _add_quality_group(protocol, "service_level", key, by_service[key], out)
        for key in sorted(by_qtype):
            _add_quality_group(protocol, "question_type", key, by_qtype[key], out)
        for key in ["positive", "negative", "unknown"]:
            _add_quality_group(protocol, "presence_polarity", key, by_presence.get(key, []), out)
        for key in ["1", "2-3", "4-9", ">=10"]:
            _add_quality_group(protocol, "gt_count_bin", key, by_count_bin.get(key, []), out)
    return out


def _pick_policy(rows: list[dict[str, str]], protocol: str, policy: str) -> dict[str, str] | None:
    for row in rows:
        if row["protocol"] == protocol and row["policy"] == policy:
            return row
    return None


def _metric(row: dict[str, str] | None, field: str) -> str:
    if row is None:
        return "-"
    raw = row.get(field, "")
    if raw == "":
        return "-"
    try:
        return f"{float(raw):.3f}"
    except ValueError:
        return raw


def _write_plot(path: Path, policy_rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = []
    xs = []
    ys = []
    for row in policy_rows:
        if row["policy"] not in {"always_cache", "always_light", "always_image", "always_roi", "greedy_min_sufficient_evidence", "oracle_best_feasible_evidence"}:
            continue
        labels.append(f"{row['protocol']}\n{row['policy'].replace('_', ' ')}")
        xs.append(_float(row.get("average_payload_kb")))
        ys.append(_float(row.get("task_success_rate")))
    if not labels:
        return
    ensure_parent(path)
    plt.figure(figsize=(9.0, 5.0))
    plt.scatter(xs, ys, s=60)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("Average payload (KB/task)")
    plt.ylabel("Task success rate")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _write_markdown(
    path: Path,
    policy_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
    threshold_summary: list[dict[str, str]],
    source_paths: dict[str, Path],
) -> None:
    lines = [
        "# V1.8 Protocol Comparison Report",
        "",
        "V1.8 reports two evaluation protocols instead of treating all results as one task distribution.",
        "",
        "- Protocol-A: operational positive-query setting. This corresponds to V1.6 and is used for the main semantic-communication claim: task-aware evidence selection reduces payload while keeping useful VQA service quality.",
        "- Protocol-B: balanced presence calibration. This corresponds to V1.7 and adds negative presence questions, so a lower success rate should be interpreted as stricter calibration, not system regression.",
        "- Protocol-B+ROI: optional fairness check that adds detector-guided ROI/crop image evidence while keeping full-image evidence as a baseline.",
        "",
        "## Policy Tables",
        "",
    ]
    for protocol in sorted({row["protocol"] for row in policy_rows}):
        lines.append(f"### {protocol}")
        lines.append("")
        lines.append("| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality viol. | deadline viol. | s0 | s1 | s2 | s3 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in [item for item in policy_rows if item["protocol"] == protocol]:
            lines.append(
                f"| {row['policy']} | {_metric(row, 'task_success_rate')} | {_metric(row, 'average_accuracy')} | "
                f"{_metric(row, 'average_delay')} | {_metric(row, 'average_energy')} | {_metric(row, 'average_payload_kb')} | "
                f"{_metric(row, 'payload_reduction_vs_always_image')} | {_metric(row, 'quality_violation_rate')} | "
                f"{_metric(row, 'deadline_violation_rate')} | {_metric(row, 'service_level_0_ratio')} | "
                f"{_metric(row, 'service_level_1_ratio')} | {_metric(row, 'service_level_2_ratio')} | {_metric(row, 'service_level_3_ratio')} |"
            )
        lines.append("")
    lines.extend(["## Main Reading", ""])
    a_greedy = _pick_policy(policy_rows, "Protocol-A", "greedy_min_sufficient_evidence")
    b_greedy = _pick_policy(policy_rows, "Protocol-B", "greedy_min_sufficient_evidence")
    lines.append(
        f"- Protocol-A greedy success is {_metric(a_greedy, 'task_success_rate')} with "
        f"{_metric(a_greedy, 'average_payload_kb')} KB/task and "
        f"{_metric(a_greedy, 'payload_reduction_vs_always_image')} payload reduction vs full-image."
    )
    lines.append(
        f"- Protocol-B greedy success is {_metric(b_greedy, 'task_success_rate')}; this includes negative presence and therefore tests detector false negatives/false positives more strictly."
    )
    lines.append("- The comparison should be framed as operational efficiency vs calibration robustness, not V1.7 being worse than V1.6.")
    lines.extend(["", "## Quality Breakdown", ""])
    lines.append("| protocol | group | value | samples | accuracy | payload KB |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in quality_rows:
        if row["group"] in {"all", "service_level", "question_type", "presence_polarity", "gt_count_bin"}:
            lines.append(
                f"| {row['protocol']} | {row['group']} | {row['value']} | {row['samples']} | {row['accuracy']} | {row['mean_payload_kb']} |"
            )
    if threshold_summary:
        lines.extend(["", "## Detector Threshold Sweep", ""])
        lines.append("| threshold | overall | positive presence | negative presence | counting | payload KB |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        by_key = {(row["threshold"], row["group"], row["value"]): row for row in threshold_summary}
        thresholds = sorted({row["threshold"] for row in threshold_summary}, key=float)
        for threshold in thresholds:
            overall = by_key.get((threshold, "overall", "all"), {})
            positive = by_key.get((threshold, "presence_polarity", "positive"), {})
            negative = by_key.get((threshold, "presence_polarity", "negative"), {})
            counting = by_key.get((threshold, "question_type", "counting"), {})
            lines.append(
                f"| {threshold} | {overall.get('accuracy', '-')} | {positive.get('accuracy', '-')} | "
                f"{negative.get('accuracy', '-')} | {counting.get('accuracy', '-')} | {overall.get('mean_payload_kb', '-')} |"
            )
    lines.extend(["", "## Source Files", ""])
    for label, src in source_paths.items():
        lines.append(f"- {label}: `{src}`")
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-a-config", default="configs/v1_6_semantic_decoder_hybrid.yaml")
    parser.add_argument("--protocol-b-config", default="configs/v1_7_quality_calibrated.yaml")
    parser.add_argument("--roi-config", default=None)
    parser.add_argument("--threshold-summary-csv", default="outputs/reports/v1_8_detector_threshold_sweep.csv")
    parser.add_argument("--out-md", default="outputs/reports/v1_8_protocol_comparison.md")
    parser.add_argument("--out-policy-csv", default="outputs/reports/v1_8_protocol_policy_table.csv")
    parser.add_argument("--out-quality-csv", default="outputs/reports/v1_8_protocol_quality_table.csv")
    parser.add_argument("--out-figure", default="outputs/figures/v1_8_protocol_success_payload.png")
    args = parser.parse_args()

    policy_rows, source_paths = collect_policy_rows(args.protocol_a_config, args.protocol_b_config, args.roi_config)
    prediction_paths = [
        ("Protocol-A", source_paths["Protocol-A predictions"]),
        ("Protocol-B", source_paths["Protocol-B predictions"]),
    ]
    if "Protocol-B+ROI predictions" in source_paths:
        prediction_paths.append(("Protocol-B+ROI", source_paths["Protocol-B+ROI predictions"]))
    quality_rows = collect_quality_rows(prediction_paths)
    threshold_summary = _read_csv(resolve_path(args.threshold_summary_csv))

    policy_csv = resolve_path(args.out_policy_csv)
    quality_csv = resolve_path(args.out_quality_csv)
    out_md = resolve_path(args.out_md)
    out_figure = resolve_path(args.out_figure)
    _write_csv(policy_csv, policy_rows, POLICY_FIELDS)
    _write_csv(quality_csv, quality_rows, QUALITY_FIELDS)
    _write_markdown(out_md, policy_rows, quality_rows, threshold_summary, source_paths)
    _write_plot(out_figure, policy_rows)
    print(f"policy_rows={len(policy_rows)} quality_rows={len(quality_rows)}")
    print(f"protocol_report={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
