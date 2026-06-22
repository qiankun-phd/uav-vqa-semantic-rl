#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, resolve_path
from vqa_semcom.semantic.utility import SemanticUtilityModel


DEFAULT_EXAMPLES = [
    ("presence", 0, "-5dB", "good", "fresh", "normal"),
    ("presence", 1, "20dB", "good", "fresh", "normal"),
    ("presence", 2, "0dB", "medium", "fresh", "critical"),
    ("counting", 1, "10dB", "good", "stale", "normal"),
    ("counting", 2, "15dB", "poor", "expired", "critical"),
]


def _format_row(example: tuple[str, int, str, str, str, str], model: SemanticUtilityModel) -> dict[str, str]:
    task_type, service_level, snr_bin, view_quality, freshness, risk = example
    estimate = model.U_sem(task_type, service_level, snr_bin, view_quality, freshness, risk)
    return {
        "task_type": task_type,
        "service_level": str(service_level),
        "snr_bin": snr_bin,
        "view": view_quality,
        "freshness": freshness,
        "risk": risk,
        "accuracy_mean": f"{estimate.accuracy_mean:.3f}",
        "accuracy_lcb": f"{estimate.accuracy_lcb:.3f}",
        "payload_kb": f"{estimate.payload_kb:.3f}",
        "uncertainty": f"{estimate.uncertainty:.3f}",
        "sample_count": str(estimate.sample_count),
    }


def _markdown_table(rows: list[dict[str, str]]) -> list[str]:
    columns = [
        "task_type",
        "service_level",
        "snr_bin",
        "view",
        "freshness",
        "risk",
        "accuracy_mean",
        "accuracy_lcb",
        "payload_kb",
        "uncertainty",
        "sample_count",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---" for _ in columns]) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    return lines


def write_report(rows: list[dict[str, str]], output_path: Path, utility_csv: Path) -> None:
    ensure_parent(output_path)
    lines = [
        "# Semantic Utility API Examples",
        "",
        f"Utility CSV: `{utility_csv}`",
        "",
        "These examples call:",
        "",
        "```python",
        "U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)",
        "# -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count",
        "```",
        "",
        "For RL/resource control, `accuracy_lcb` is the conservative QoS estimate. `accuracy_mean` remains useful for reporting expected answer correctness.",
        "",
    ]
    lines.extend(_markdown_table(rows))
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `s=0` cache has near-zero payload and does not depend on current SNR.",
            "- `s=1` semantic tokens usually have much lower payload than image evidence.",
            "- `s=2` image evidence may increase payload substantially, so it should be selected only when the utility gain justifies the cost.",
            "- High `uncertainty` or low `sample_count` should make RL treat the estimate conservatively.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the calibrated semantic utility API for representative examples.")
    parser.add_argument("--utility-csv", default="outputs/lut/v1_9_semantic_utility_with_ci.csv")
    parser.add_argument("--output-md", default="outputs/reports/semantic_utility_api_examples.md")
    args = parser.parse_args()

    utility_csv = resolve_path(args.utility_csv)
    output_md = resolve_path(args.output_md)
    model = SemanticUtilityModel.from_csv(utility_csv)
    rows = [_format_row(example, model) for example in DEFAULT_EXAMPLES]
    write_report(rows, output_md, utility_csv)

    for line in _markdown_table(rows):
        print(line)
    print(f"\nwrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

