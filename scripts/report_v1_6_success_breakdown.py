#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _correct(row: dict[str, str]) -> float:
    return 1.0 if str(row.get("correct", "")).lower() == "true" else 0.0


def _service_name(level: str) -> str:
    return {
        "0": "cache",
        "1": "semantic-token decoder",
        "2": "full image Qwen",
        "3": "ROI crop Qwen",
    }.get(str(level), f"s={level}")


def _delay_for(row: dict[str, str], cfg: dict) -> float:
    delay_by_level = {str(k): float(v) for k, v in cfg["simulation"]["delay_by_level"].items()}
    channel_delay = {str(k): float(v) for k, v in cfg["simulation"]["channel_delay_multiplier"].items()}
    return delay_by_level.get(str(row["service_level"]), delay_by_level.get("2", 3.4)) * channel_delay.get(row["channel_bin"], 1.0)


def _lut_lookup(lut: dict[tuple[str, str, str, str, str, str], float], row: dict[str, str]) -> float:
    key = (
        row["question_type"],
        str(row["service_level"]),
        row["channel_bin"],
        row["view_quality_bin"],
        row["freshness_bin"],
        row["risk_level"],
    )
    return lut.get(key, _correct(row))


def _group_rows(predictions: list[dict[str, str]], lut: dict[tuple[str, str, str, str, str, str], float], cfg: dict, keys: list[str]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for values, rows in sorted(grouped.items()):
        correctness = [_correct(row) for row in rows]
        quality = []
        deadline = []
        success = []
        for row in rows:
            q_ok = _lut_lookup(lut, row) >= float(row["epsilon_k"])
            d_ok = _delay_for(row, cfg) <= float(row["tau_k"])
            quality.append(float(q_ok))
            deadline.append(float(d_ok))
            success.append(float(q_ok and d_ok))
        record = {key: value for key, value in zip(keys, values)}
        if "service_level" in record:
            record["service_name"] = _service_name(record["service_level"])
        record.update(
            {
                "samples": str(len(rows)),
                "answer_correctness": f"{_mean(correctness):.6f}",
                "quality_satisfaction": f"{_mean(quality):.6f}",
                "deadline_satisfaction": f"{_mean(deadline):.6f}",
                "final_success": f"{_mean(success):.6f}",
            }
        )
        out.append(record)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_6_semantic_decoder.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    predictions = _read_csv(resolve_path(cfg["paths"]["vlm_predictions_csv"]))
    lut_rows = _read_csv(resolve_path(cfg["paths"]["vlm_lut_csv"]))
    if not predictions:
        raise RuntimeError("prediction CSV is required before success breakdown")
    lut = {
        (
            row["question_type"],
            str(row["service_level"]),
            row["channel_bin"],
            row["view_quality_bin"],
            row["freshness_bin"],
            row["risk_level"],
        ): float(row["expected_accuracy"])
        for row in lut_rows
    }
    out_csv = resolve_path(cfg["paths"].get("success_breakdown_csv", "outputs/reports/v1_6_success_breakdown.csv"))
    out_md = resolve_path(cfg["paths"].get("success_breakdown_md", "outputs/reports/v1_6_success_breakdown.md"))
    rows = []
    for group_name, keys in [
        ("by_service", ["service_level"]),
        ("by_service_task", ["service_level", "question_type"]),
        ("by_service_risk", ["service_level", "risk_level"]),
        ("by_service_channel", ["service_level", "channel_bin"]),
        ("by_service_task_risk", ["service_level", "question_type", "risk_level"]),
    ]:
        for row in _group_rows(predictions, lut, cfg, keys):
            row = {"group": group_name, **row}
            rows.append(row)
    fieldnames = [
        "group", "service_level", "service_name", "question_type", "risk_level", "channel_bin",
        "samples", "answer_correctness", "quality_satisfaction", "deadline_satisfaction", "final_success",
    ]
    normalized_rows = [{key: row.get(key, "") for key in fieldnames} for row in rows]
    ensure_parent(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    lines = [
        "# V1.6 Success Breakdown",
        "",
        "This report separates answer correctness, semantic quality satisfaction, deadline satisfaction, and final task success.",
        "Quality satisfaction uses the measured LUT accuracy `A_k >= epsilon_k`; deadline satisfaction uses the configured delay proxy.",
        "",
    ]
    for group_name in ["by_service", "by_service_task", "by_service_risk", "by_service_channel", "by_service_task_risk"]:
        group_rows = [row for row in normalized_rows if row["group"] == group_name]
        lines.extend([f"## {group_name}", ""])
        visible_cols = [col for col in fieldnames if col != "group" and any(row.get(col, "") for row in group_rows)]
        lines.append("| " + " | ".join(visible_cols) + " |")
        lines.append("|" + "|".join("---" for _ in visible_cols) + "|")
        for row in group_rows:
            lines.append("| " + " | ".join(row.get(col, "") for col in visible_cols) + " |")
        lines.append("")
    ensure_parent(out_md)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"success_breakdown_md={out_md}")
    print(f"success_breakdown_csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
