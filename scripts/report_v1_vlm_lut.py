#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return _read_csv(path) if path.exists() else []


def _mean(values: list[float]) -> str:
    return f"{statistics.mean(values):.3f}" if values else "-"


def _mean_float(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _correct_value(row: dict[str, str]) -> float:
    return 1.0 if str(row.get("correct", "")).lower() == "true" else 0.0


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit() or ch == "-")
    if digits in {"", "-"}:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _bootstrap_mean_ci(values: list[float], rounds: int = 1000, seed: int = 2026) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(rounds):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    low = estimates[max(0, int(0.025 * rounds))]
    high = estimates[min(rounds - 1, int(0.975 * rounds))]
    return low, high


def _accuracy_ci_text(values: list[float]) -> str:
    if not values:
        return "-"
    low, high = _bootstrap_mean_ci(values)
    return f"{statistics.mean(values):.3f} [{low:.3f}, {high:.3f}]"


def _payload(row: dict[str, str]) -> float:
    try:
        return max(0.0, float(row.get("payload_bytes", "") or 0.0))
    except ValueError:
        return 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    return sorted_values[min(len(sorted_values) - 1, int(0.95 * len(sorted_values)))]


def _write_companion_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _service_name(level: str) -> str:
    return {
        "0": "cache answer",
        "1": "detector semantic tokens",
        "2": "raw visual evidence",
        "3": "detector ROI crop image",
    }.get(str(level), f"s={level}")


def _counting_tolerance(gt: int) -> int:
    return max(1, int(round(0.1 * max(0, gt))))


def _failure_taxonomy(predictions: list[dict[str, str]]) -> tuple[Counter[str], list[dict[str, str]]]:
    taxonomy: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for row in predictions:
        if _correct_value(row) >= 1.0:
            continue
        qtype = row.get("question_type", "")
        level = row.get("service_level", "")
        gt_text = str(row.get("ground_truth_answer", "")).strip().lower()
        pred_text = str(row.get("normalized_prediction") or row.get("predicted_answer", "")).strip().lower()
        label = "other_failure"
        if qtype == "presence":
            gt_yes = gt_text in {"yes", "true", "1"}
            pred_yes = pred_text in {"yes", "true", "1"}
            if gt_yes and not pred_yes:
                label = "presence_false_negative"
            elif not gt_yes and pred_yes:
                label = "presence_false_positive"
        elif qtype == "counting":
            gt = _int_or_none(row.get("ground_truth_answer"))
            pred = _int_or_none(row.get("normalized_prediction") or row.get("predicted_answer"))
            if pred is None:
                label = "counting_invalid_answer"
            elif gt is not None:
                tol = _counting_tolerance(gt)
                if pred < gt - tol:
                    label = "counting_undercount"
                elif pred > gt + tol:
                    label = "counting_overcount"
        if level == "1":
            label = f"detector_semantic_{label}"
        elif level == "2":
            label = f"image_vlm_{label}"
        elif level == "3":
            label = f"roi_image_vlm_{label}"
        taxonomy[label] += 1
        rows.append(
            {
                "failure_type": label,
                "image_id": row.get("image_id", ""),
                "service_level": level,
                "channel_bin": row.get("channel_bin", ""),
                "view_quality_bin": row.get("view_quality_bin", ""),
                "question_type": qtype,
                "question": row.get("question", ""),
                "ground_truth_answer": row.get("ground_truth_answer", ""),
                "predicted_answer": row.get("predicted_answer", ""),
            }
        )
    return taxonomy, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    predictions = _read_csv(resolve_path(cfg["paths"]["vlm_predictions_csv"]))
    lut = _read_csv(resolve_path(cfg["paths"]["vlm_lut_csv"]))
    if not predictions or not lut:
        raise RuntimeError("V1 predictions and LUT must be built before reporting.")

    report_path = resolve_path(cfg["paths"]["vlm_report_md"])
    report_dir = report_path.parent
    sim_rows = _read_csv_if_exists(resolve_path(cfg["paths"].get("sim_results_csv", "outputs/sim/v1_qwen_results.csv")))

    correct_values = [_correct_value(row) for row in predictions]
    model_names = sorted({row["model_name"] for row in predictions})
    measured_models = [name for name in model_names if name not in {"mock-vlm", "cache-simulator"}]
    latency_by_evidence: dict[str, list[float]] = defaultdict(list)
    payload_by_evidence: dict[str, list[float]] = defaultdict(list)
    payload_by_service_channel: dict[tuple[str, str], list[float]] = defaultdict(list)
    rows_by_service: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows_by_service_qtype: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        latency_by_evidence[row["evidence_type"]].append(float(row.get("latency_sec", 0.0) or 0.0))
        payload = _payload(row)
        payload_by_evidence[row["evidence_type"]].append(payload)
        payload_by_service_channel[(row["service_level"], row["channel_bin"])].append(payload)
        rows_by_service[row["service_level"]].append(row)
        rows_by_service_qtype[(row["service_level"], row["question_type"])].append(row)
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in predictions:
        grouped[(row["question_type"], row["service_level"], row["channel_bin"], row["view_quality_bin"])].append(
            _correct_value(row)
        )
    qtypes = sorted({row["question_type"] for row in predictions})
    levels = sorted({row["service_level"] for row in predictions}, key=int)
    channels = ["bad", "medium", "good"]
    views = ["poor", "medium", "good"]
    lines = [
        "# V1 VLM-measured Semantic Quality Report",
        "",
        "This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.",
        "It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.",
        "",
        f"- prediction rows: `{len(predictions)}`",
        f"- LUT rows: `{len(lut)}`",
        f"- overall measured accuracy: `{statistics.mean(correct_values):.3f}`",
        f"- model names: `{', '.join(model_names)}`",
        f"- real VLM present: `{'yes' if measured_models else 'no'}`",
        "",
        "## Latency and Payload Summary",
        "",
        "| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    payload_summary_rows = []
    for evidence_type in sorted(latency_by_evidence):
        latencies = latency_by_evidence[evidence_type]
        payloads = payload_by_evidence[evidence_type]
        mean_payload_kb = statistics.mean(payloads) / 1024.0 if payloads else 0.0
        p95_payload_kb = _p95(payloads) / 1024.0
        lines.append(
            f"| {evidence_type} | {statistics.mean(latencies):.3f} | {_p95(latencies):.3f} | "
            f"{mean_payload_kb:.3f} | {p95_payload_kb:.3f} | {len(latencies)} |"
        )
        payload_summary_rows.append(
            {
                "evidence_type": evidence_type,
                "mean_latency_sec": f"{statistics.mean(latencies):.6f}",
                "p95_latency_sec": f"{_p95(latencies):.6f}",
                "mean_payload_kb": f"{mean_payload_kb:.6f}",
                "p95_payload_kb": f"{p95_payload_kb:.6f}",
                "samples": str(len(latencies)),
            }
        )
    lines.extend(["", "## Payload by Service and Channel", ""])
    lines.append("| service | channel | mean payload (KB) | p95 payload (KB) | samples |")
    lines.append("|---:|---|---:|---:|---:|")
    payload_service_rows = []
    for level in levels:
        for channel in channels:
            values = payload_by_service_channel.get((level, channel), [])
            mean_payload_kb = statistics.mean(values) / 1024.0 if values else 0.0
            p95_payload_kb = _p95(values) / 1024.0
            lines.append(f"| {level} | {channel} | {mean_payload_kb:.3f} | {p95_payload_kb:.3f} | {len(values)} |")
            payload_service_rows.append(
                {
                    "service_level": level,
                    "channel_bin": channel,
                    "mean_payload_kb": f"{mean_payload_kb:.6f}",
                    "p95_payload_kb": f"{p95_payload_kb:.6f}",
                    "samples": str(len(values)),
                }
            )
    semantic_vs_image_rows = []
    lines.extend([
        "",
        "## Semantic-token vs Raw-image Baseline",
        "",
        "Here `s=1` is detector-generated semantic-token transmission: classes, boxes, counts, and confidence summaries. "
        "`s=2` is raw visual evidence transmission: degraded full-image bytes sent to Qwen-VL. "
        "`s=3` is detector-guided ROI/crop image transmission: zoomed target-region bytes sent to Qwen-VL. "
        "The comparison therefore tests whether task-aware semantic communication can reduce payload while preserving answer correctness.",
        "",
        "| service | evidence baseline | accuracy with 95% CI | mean payload (KB) | p95 payload (KB) | mean latency (s) | samples |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for level in levels:
        rows = rows_by_service[level]
        accuracies = [_correct_value(row) for row in rows]
        payloads = [_payload(row) for row in rows]
        latencies = [float(row.get("latency_sec", 0.0) or 0.0) for row in rows]
        mean_payload_kb = _mean_float(payloads) / 1024.0
        p95_payload_kb = _p95(payloads) / 1024.0
        mean_latency = _mean_float(latencies)
        lines.append(
            f"| {level} | {_service_name(level)} | {_accuracy_ci_text(accuracies)} | "
            f"{mean_payload_kb:.3f} | {p95_payload_kb:.3f} | {mean_latency:.3f} | {len(rows)} |"
        )
        low, high = _bootstrap_mean_ci(accuracies)
        semantic_vs_image_rows.append(
            {
                "service_level": level,
                "evidence_baseline": _service_name(level),
                "accuracy": f"{_mean_float(accuracies):.6f}",
                "accuracy_ci_low": f"{low:.6f}",
                "accuracy_ci_high": f"{high:.6f}",
                "mean_payload_kb": f"{mean_payload_kb:.6f}",
                "p95_payload_kb": f"{p95_payload_kb:.6f}",
                "mean_latency_sec": f"{mean_latency:.6f}",
                "samples": str(len(rows)),
            }
        )
    lines.extend(["", "### Accuracy by Service and Task Type", ""])
    lines.append("| service | evidence baseline | question type | accuracy with 95% CI | samples |")
    lines.append("|---:|---|---|---:|---:|")
    accuracy_ci_rows = []
    for level in levels:
        for qtype in qtypes:
            rows = rows_by_service_qtype.get((level, qtype), [])
            accuracies = [_correct_value(row) for row in rows]
            low, high = _bootstrap_mean_ci(accuracies)
            lines.append(f"| {level} | {_service_name(level)} | {qtype} | {_accuracy_ci_text(accuracies)} | {len(rows)} |")
            accuracy_ci_rows.append(
                {
                    "service_level": level,
                    "evidence_baseline": _service_name(level),
                    "question_type": qtype,
                    "accuracy": f"{_mean_float(accuracies):.6f}",
                    "accuracy_ci_low": f"{low:.6f}",
                    "accuracy_ci_high": f"{high:.6f}",
                    "samples": str(len(rows)),
                }
            )
    lines.extend([
        "",
        "## Prediction Distribution",
        "",
        "| field | value | count |",
        "|---|---|---:|",
    ])
    for field in ["question_type", "service_level", "channel_bin", "freshness_bin", "view_quality_bin", "evidence_type"]:
        for key, count in sorted(Counter(row[field] for row in predictions).items()):
            lines.append(f"| {field} | {key} | {count} |")
    lines.extend(["", "## Accuracy by Service, Channel, and View", ""])
    for qtype in qtypes:
        lines.append(f"### {qtype}")
        lines.append("")
        lines.append("| service | channel | poor view | medium view | good view |")
        lines.append("|---:|---|---:|---:|---:|")
        for level in levels:
            for channel in channels:
                vals = [_mean(grouped.get((qtype, level, channel, view), [])) for view in views]
                lines.append(f"| {level} | {channel} | {vals[0]} | {vals[1]} | {vals[2]} |")
        lines.append("")
    if sim_rows:
        lines.extend(["## Paper Table: Resource Policies", ""])
        lines.append("| policy | success | accuracy | delay | energy | payload KB | payload reduction | cache | light | image | roi | success 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sim_rows:
            tasks = max(1, int(float(row.get("tasks", 1) or 1)))
            success = float(row["task_success_rate"])
            half_width = 1.96 * math.sqrt(max(0.0, success * (1.0 - success)) / tasks)
            lines.append(
                f"| {row['policy']} | {success:.3f} | {float(row['average_accuracy']):.3f} | "
                f"{float(row['average_delay']):.3f} | {float(row['average_energy']):.3f} | "
                f"{float(row.get('average_payload_kb', 0.0)):.3f} | {float(row.get('payload_reduction_vs_always_image', 0.0)):.3f} | "
                f"{float(row.get('service_level_0_ratio', 0.0)):.3f} | {float(row.get('service_level_1_ratio', 0.0)):.3f} | "
                f"{float(row.get('service_level_2_ratio', 0.0)):.3f} | {float(row.get('service_level_3_ratio', 0.0)):.3f} | "
                f"[{max(0.0, success - half_width):.3f}, {min(1.0, success + half_width):.3f}] |"
            )
        lines.append("")
    taxonomy, taxonomy_rows = _failure_taxonomy(predictions)
    lines.extend(["## Failure Taxonomy", ""])
    lines.append("| failure type | count | share among failures |")
    lines.append("|---|---:|---:|")
    total_failures = max(1, sum(taxonomy.values()))
    for label, count in taxonomy.most_common():
        lines.append(f"| {label} | {count} | {count / total_failures:.3f} |")
    if not taxonomy:
        lines.append("| no_failure | 0 | 0.000 |")
    lines.append("")
    failures = []
    seen_failures = set()
    for row in predictions:
        key = (row["image_id"], row["service_level"], row["channel_bin"], row["question"], row["predicted_answer"])
        if row["correct"].lower() != "true" and row["service_level"] in {"1", "2", "3"} and key not in seen_failures:
            failures.append(row)
            seen_failures.add(key)
    lines.extend(["## Example Failures", ""])
    for row in failures[:12]:
        payload_kb = _payload(row) / 1024.0
        lines.append(
            f"- `{row['image_id']}` s={row['service_level']} channel={row['channel_bin']} "
            f"payload={payload_kb:.2f}KB Q: {row['question']} GT: `{row['ground_truth_answer']}` Pred: `{row['predicted_answer']}`"
        )
    if not failures:
        lines.append("- No failures found in this run.")
    lines.append("")
    ensure_parent(report_path)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _write_companion_csv(
        report_dir / "v1_qwen_payload_by_evidence.csv",
        ["evidence_type", "mean_latency_sec", "p95_latency_sec", "mean_payload_kb", "p95_payload_kb", "samples"],
        payload_summary_rows,
    )
    _write_companion_csv(
        report_dir / "v1_qwen_payload_by_service_channel.csv",
        ["service_level", "channel_bin", "mean_payload_kb", "p95_payload_kb", "samples"],
        payload_service_rows,
    )
    _write_companion_csv(
        report_dir / "v1_qwen_semantic_vs_image_baseline.csv",
        [
            "service_level",
            "evidence_baseline",
            "accuracy",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "mean_payload_kb",
            "p95_payload_kb",
            "mean_latency_sec",
            "samples",
        ],
        semantic_vs_image_rows,
    )
    _write_companion_csv(
        report_dir / "v1_qwen_accuracy_ci_by_service_task.csv",
        ["service_level", "evidence_baseline", "question_type", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "samples"],
        accuracy_ci_rows,
    )
    _write_companion_csv(
        report_dir / "v1_qwen_failure_taxonomy.csv",
        [
            "failure_type",
            "image_id",
            "service_level",
            "channel_bin",
            "view_quality_bin",
            "question_type",
            "question",
            "ground_truth_answer",
            "predicted_answer",
        ],
        taxonomy_rows,
    )
    print(f"vlm_report_md={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
