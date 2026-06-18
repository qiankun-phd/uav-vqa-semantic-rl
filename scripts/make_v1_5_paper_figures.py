#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _correct(row: dict[str, str]) -> float:
    return 1.0 if str(row.get("correct", "")).lower() == "true" else 0.0


def _payload_kb(row: dict[str, str]) -> float:
    try:
        return max(0.0, float(row.get("payload_bytes", "") or 0.0)) / 1024.0
    except ValueError:
        return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _service_name(level: str) -> str:
    return {
        "0": "cache",
        "1": "semantic tokens",
        "2": "full image",
        "3": "ROI crop",
    }.get(str(level), f"s={level}")


def _failure_label(row: dict[str, str]) -> str:
    if _correct(row) >= 1.0:
        return ""
    qtype = row.get("question_type", "")
    level = row.get("service_level", "")
    pred = str(row.get("normalized_prediction") or row.get("predicted_answer", "")).strip().lower()
    gt = str(row.get("ground_truth_answer", "")).strip().lower()
    label = "other"
    if qtype == "presence":
        gt_yes = gt in {"yes", "true", "1"}
        pred_yes = pred in {"yes", "true", "1"}
        if gt_yes and not pred_yes:
            label = "presence false negative"
        elif not gt_yes and pred_yes:
            label = "presence false positive"
    elif qtype == "counting":
        try:
            gt_i = int(float(gt))
            pred_i = int(float(pred))
            if pred_i < gt_i:
                label = "counting undercount"
            elif pred_i > gt_i:
                label = "counting overcount"
            else:
                label = "counting invalid/tolerance"
        except ValueError:
            label = "counting invalid"
    return f"{_service_name(level)}: {label}"


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_5_detector_qwen_roi.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    predictions = _read_csv(resolve_path(cfg["paths"]["vlm_predictions_csv"]))
    sim_rows = _read_csv(resolve_path(cfg["paths"]["sim_results_csv"]))
    if not predictions:
        raise RuntimeError("prediction CSV is required before making figures")

    fig_dir = resolve_path(cfg["paths"].get("paper_figures_dir", "outputs/figures/v1_5"))
    table_dir = resolve_path(cfg["paths"].get("paper_tables_dir", "outputs/reports/v1_5_tables"))
    ensure_parent(fig_dir / "dummy.txt")
    ensure_parent(table_dir / "dummy.txt")

    by_service: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_service_qtype: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_service_channel: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    failures = Counter()
    for row in predictions:
        service = row["service_level"]
        by_service[service].append(row)
        by_service_qtype[(service, row["question_type"])].append(row)
        by_service_channel[(service, row["channel_bin"])].append(row)
        label = _failure_label(row)
        if label:
            failures[label] += 1

    service_rows = []
    for service, rows in sorted(by_service.items(), key=lambda item: int(item[0])):
        service_rows.append(
            {
                "service_level": service,
                "evidence": _service_name(service),
                "accuracy": f"{_mean([_correct(row) for row in rows]):.6f}",
                "mean_payload_kb": f"{_mean([_payload_kb(row) for row in rows]):.6f}",
                "mean_latency_sec": f"{_mean([float(row.get('latency_sec', 0.0) or 0.0) for row in rows]):.6f}",
                "samples": str(len(rows)),
            }
        )
    _write_csv(table_dir / "service_accuracy_payload_latency.csv", service_rows)

    qtype_rows = []
    for (service, qtype), rows in sorted(by_service_qtype.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        qtype_rows.append(
            {
                "service_level": service,
                "evidence": _service_name(service),
                "question_type": qtype,
                "accuracy": f"{_mean([_correct(row) for row in rows]):.6f}",
                "samples": str(len(rows)),
            }
        )
    _write_csv(table_dir / "accuracy_by_service_task.csv", qtype_rows)

    channel_rows = []
    for (service, channel), rows in sorted(by_service_channel.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        channel_rows.append(
            {
                "service_level": service,
                "evidence": _service_name(service),
                "channel": channel,
                "accuracy": f"{_mean([_correct(row) for row in rows]):.6f}",
                "mean_payload_kb": f"{_mean([_payload_kb(row) for row in rows]):.6f}",
                "samples": str(len(rows)),
            }
        )
    _write_csv(table_dir / "accuracy_payload_by_service_channel.csv", channel_rows)

    failure_rows = [
        {"failure_type": label, "count": str(count)}
        for label, count in failures.most_common(20)
    ]
    _write_csv(table_dir / "failure_taxonomy_top20.csv", failure_rows)
    _write_csv(table_dir / "resource_policy_summary.csv", sim_rows)

    plt = _load_matplotlib()
    labels = [row["evidence"] for row in service_rows]
    acc = [float(row["accuracy"]) for row in service_rows]
    payload = [float(row["mean_payload_kb"]) for row in service_rows]
    plt.figure(figsize=(6.2, 4.0))
    plt.scatter(payload, acc, s=80)
    for x, y, label in zip(payload, acc, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Mean payload (KB)")
    plt.ylabel("Answer accuracy")
    plt.title("Accuracy-payload tradeoff")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_payload_tradeoff.png", dpi=220)
    plt.close()

    if sim_rows:
        policies = [row["policy"] for row in sim_rows]
        cache = [float(row.get("service_level_0_ratio", 0.0)) for row in sim_rows]
        light = [float(row.get("service_level_1_ratio", 0.0)) for row in sim_rows]
        image = [float(row.get("service_level_2_ratio", 0.0)) for row in sim_rows]
        roi = [float(row.get("service_level_3_ratio", 0.0)) for row in sim_rows]
        x = range(len(policies))
        plt.figure(figsize=(8.2, 4.0))
        plt.bar(x, cache, label="cache")
        plt.bar(x, light, bottom=cache, label="semantic tokens")
        bottom2 = [a + b for a, b in zip(cache, light)]
        plt.bar(x, roi, bottom=bottom2, label="ROI crop")
        bottom3 = [a + b for a, b in zip(bottom2, roi)]
        plt.bar(x, image, bottom=bottom3, label="full image")
        plt.xticks(list(x), policies, rotation=25, ha="right", fontsize=8)
        plt.ylabel("Selection ratio")
        plt.title("Service level selection ratio")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "service_selection_ratio.png", dpi=220)
        plt.close()

    channels = ["bad", "medium", "good"]
    plt.figure(figsize=(6.5, 4.0))
    for service in sorted(by_service, key=int):
        ys = []
        for channel in channels:
            rows = by_service_channel.get((service, channel), [])
            ys.append(_mean([_correct(row) for row in rows]))
        plt.plot(channels, ys, marker="o", label=_service_name(service))
    plt.xlabel("Channel quality")
    plt.ylabel("Answer accuracy")
    plt.title("Accuracy under channel quality")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_vs_channel.png", dpi=220)
    plt.close()

    if failure_rows:
        top = failure_rows[:10]
        plt.figure(figsize=(8.4, 4.2))
        plt.barh([row["failure_type"] for row in reversed(top)], [int(row["count"]) for row in reversed(top)])
        plt.xlabel("Failure count")
        plt.title("Failure taxonomy")
        plt.tight_layout()
        plt.savefig(fig_dir / "failure_taxonomy.png", dpi=220)
        plt.close()

    print(f"paper_figures_dir={fig_dir}")
    print(f"paper_tables_dir={table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
