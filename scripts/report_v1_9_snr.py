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
from vqa_semcom.snr import snr_db_from_label


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return _read_csv(path) if path.exists() else []


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_correct(row: dict[str, str]) -> float:
    return 1.0 if str(row.get("correct", "")).lower() == "true" else 0.0


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _snr_key(label: str) -> float:
    try:
        return snr_db_from_label(label)
    except ValueError:
        return 1e9


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _mean_text(values: list[float]) -> str:
    return f"{_mean(values):.3f}" if values else "-"


def _service_name(level: str) -> str:
    return {
        "0": "cache answer",
        "1": "detector semantic tokens",
        "2": "raw image evidence",
        "3": "detector ROI image",
    }.get(str(level), f"s={level}")


def _plot_curve(rows: list[dict[str, str]], out_path: Path, y_key: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["service_level"]].append((float(row["snr_db"]), float(row[y_key])))
    if not grouped:
        return
    ensure_parent(out_path)
    plt.figure(figsize=(6.2, 3.8))
    for service, values in sorted(grouped.items(), key=lambda item: int(item[0])):
        values = sorted(values)
        plt.plot(
            [x for x, _y in values],
            [y for _x, y in values],
            marker="o",
            label=f"s={service}: {_service_name(service)}",
        )
    plt.xlabel("Sensed SNR bin (dB)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_9_snr_lut.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    predictions_path = resolve_path(cfg["paths"]["vlm_predictions_csv"])
    lut_path = resolve_path(cfg["paths"]["vlm_lut_csv"])
    predictions = _read_csv(predictions_path)
    lut = _read_csv(lut_path)
    if not predictions or not lut:
        raise RuntimeError("V1.9 predictions and SNR LUT must be built before reporting.")

    report_path = resolve_path(cfg["paths"]["vlm_report_md"])
    report_dir = report_path.parent
    figures_dir = resolve_path(cfg["paths"].get("paper_figures_dir", "outputs/figures/v1_9_snr"))
    sim_rows = _read_csv_if_exists(resolve_path(cfg["paths"].get("sim_results_csv", "outputs/sim/v1_9_snr_resource_results.csv")))

    snr_labels = sorted({row.get("snr_bin", "") for row in predictions if row.get("snr_bin", "")}, key=_snr_key)
    services = sorted({row["service_level"] for row in predictions}, key=int)
    qtypes = sorted({row["question_type"] for row in predictions})
    model_names = sorted({row.get("model_name", "") for row in predictions})
    measured_models = [name for name in model_names if name not in {"mock-vlm", "cache-simulator", "semantic-token-decoder"}]

    by_service_snr: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_qtype_service_snr: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        snr = row.get("snr_bin", "")
        by_service_snr[(row["service_level"], snr)].append(row)
        by_qtype_service_snr[(row["question_type"], row["service_level"], snr)].append(row)

    accuracy_rows: list[dict[str, str]] = []
    payload_rows: list[dict[str, str]] = []
    for service in services:
        for snr in snr_labels:
            rows = by_service_snr.get((service, snr), [])
            acc = [_is_correct(row) for row in rows]
            payloads = [_float(row, "payload_bytes") / 1024.0 for row in rows]
            accuracy_rows.append(
                {
                    "service_level": service,
                    "service_name": _service_name(service),
                    "snr_bin": snr,
                    "snr_db": f"{_snr_key(snr):g}",
                    "accuracy": f"{_mean(acc):.6f}",
                    "samples": str(len(rows)),
                }
            )
            payload_rows.append(
                {
                    "service_level": service,
                    "service_name": _service_name(service),
                    "snr_bin": snr,
                    "snr_db": f"{_snr_key(snr):g}",
                    "mean_payload_kb": f"{_mean(payloads):.6f}",
                    "samples": str(len(rows)),
                }
            )

    qtype_rows: list[dict[str, str]] = []
    for qtype in qtypes:
        for service in services:
            for snr in snr_labels:
                rows = by_qtype_service_snr.get((qtype, service, snr), [])
                qtype_rows.append(
                    {
                        "question_type": qtype,
                        "service_level": service,
                        "snr_bin": snr,
                        "snr_db": f"{_snr_key(snr):g}",
                        "accuracy": f"{_mean([_is_correct(row) for row in rows]):.6f}",
                        "samples": str(len(rows)),
                    }
                )

    service_snr_path = report_dir / "v1_9_snr_accuracy_by_snr.csv"
    payload_snr_path = report_dir / "v1_9_snr_payload_by_snr.csv"
    qtype_snr_path = report_dir / "v1_9_snr_task_accuracy_by_snr.csv"
    _write_csv(service_snr_path, list(accuracy_rows[0].keys()) if accuracy_rows else ["service_level"], accuracy_rows)
    _write_csv(payload_snr_path, list(payload_rows[0].keys()) if payload_rows else ["service_level"], payload_rows)
    _write_csv(qtype_snr_path, list(qtype_rows[0].keys()) if qtype_rows else ["question_type"], qtype_rows)
    _plot_curve(accuracy_rows, figures_dir / "v1_9_answer_accuracy_vs_snr.png", "accuracy", "Answer accuracy")
    _plot_curve(payload_rows, figures_dir / "v1_9_payload_vs_snr.png", "mean_payload_kb", "Mean payload (KB)")

    cache_accuracy_by_snr = [
        float(row["accuracy"])
        for row in accuracy_rows
        if row["service_level"] == "0" and int(row["samples"]) > 0
    ]
    cache_spread = max(cache_accuracy_by_snr) - min(cache_accuracy_by_snr) if cache_accuracy_by_snr else 0.0

    lines = [
        "# V1.9 SNR-Calibrated Semantic Quality Report",
        "",
        "This report calibrates task-conditioned answer correctness against sensed SNR bins. "
        "It does not assume a channel model such as AWGN or Rayleigh; SNR is used only as the sensed link-quality variable.",
        "",
        f"- prediction rows: `{len(predictions)}`",
        f"- LUT rows: `{len(lut)}`",
        f"- SNR bins: `{', '.join(snr_labels)}`",
        f"- model names: `{', '.join(model_names)}`",
        f"- real VLM present: `{'yes' if measured_models else 'no'}`",
        f"- cache accuracy spread across SNR: `{cache_spread:.6f}`",
        "",
        "## Answer Accuracy vs Sensed SNR",
        "",
        "| service | evidence | " + " | ".join(snr_labels) + " |",
        "|---:|---|" + "|".join(["---:" for _ in snr_labels]) + "|",
    ]
    for service in services:
        cells = []
        for snr in snr_labels:
            rows = by_service_snr.get((service, snr), [])
            cells.append(_mean_text([_is_correct(row) for row in rows]))
        lines.append(f"| {service} | {_service_name(service)} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Payload vs Sensed SNR", ""])
    lines.append("| service | evidence | " + " | ".join(snr_labels) + " |")
    lines.append("|---:|---|" + "|".join(["---:" for _ in snr_labels]) + "|")
    for service in services:
        cells = []
        for snr in snr_labels:
            rows = by_service_snr.get((service, snr), [])
            cells.append(_mean_text([_float(row, "payload_bytes") / 1024.0 for row in rows]))
        lines.append(f"| {service} | {_service_name(service)} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Task Breakdown", ""])
    lines.append("| question type | service | " + " | ".join(snr_labels) + " |")
    lines.append("|---|---:|" + "|".join(["---:" for _ in snr_labels]) + "|")
    for qtype in qtypes:
        for service in services:
            cells = []
            for snr in snr_labels:
                rows = by_qtype_service_snr.get((qtype, service, snr), [])
                cells.append(_mean_text([_is_correct(row) for row in rows]))
            lines.append(f"| {qtype} | {service} | " + " | ".join(cells) + " |")

    if sim_rows:
        lines.extend(["", "## Resource Simulation", ""])
        lines.append("| policy | success | accuracy | delay | energy | payload KB | payload reduction |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in sim_rows:
            lines.append(
                f"| {row.get('policy', '')} | {float(row.get('task_success_rate', 0.0)):.3f} | "
                f"{float(row.get('average_accuracy', 0.0)):.3f} | {float(row.get('average_delay', 0.0)):.3f} | "
                f"{float(row.get('average_energy', 0.0)):.3f} | {float(row.get('average_payload_kb', 0.0)):.3f} | "
                f"{float(row.get('payload_reduction_vs_always_image', 0.0)):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            f"- accuracy by SNR: `{service_snr_path}`",
            f"- payload by SNR: `{payload_snr_path}`",
            f"- task accuracy by SNR: `{qtype_snr_path}`",
            "",
            "Interpretation: online resource allocation should use continuous sensed SNR for transmission delay, "
            "then map it to the nearest `snr_bin` for LUT-based answer accuracy.",
        ]
    )
    ensure_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={report_path}")
    print(f"accuracy_by_snr={service_snr_path}")
    print(f"payload_by_snr={payload_snr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
