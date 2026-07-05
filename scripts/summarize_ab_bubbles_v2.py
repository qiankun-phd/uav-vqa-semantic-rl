#!/usr/bin/env python
"""Aggregate the BUBBLES A/B v2 chain and evaluate the pre-registered criteria.

Criteria (docs_spec/V19_Design_Review_2026-07.md):
  1. A2 conflict rate <= 0.10
  2. A1 semantic success - A2 semantic success <= 0.03
  3. A2 cache ratio (service level 0) <= 0.30
  4. B1 conflict rate - A2 conflict rate >= 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ARM_POLICY = {
    "A1": "ppo",
    "A2": "ppo",
    "B1": "ppo",
    "B2": "ppo",
    "Cgreedy": "semantic_greedy",
    "Ccache": "always_cache",
}

METRICS = [
    "semantic_success_rate",
    "task_success_rate",
    "airspace_conflict_rate",
    "utm_conflict_violation_rate",
    "service_level_0_ratio",
    "service_level_1_ratio",
    "service_level_2_ratio",
    "deadline_violation_rate",
    "average_accuracy",
]


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def collect(root: Path) -> dict[str, dict[str, float]]:
    per_arm: dict[str, list[dict[str, float]]] = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or "_" not in run_dir.name:
            continue
        arm = run_dir.name.rsplit("_", 1)[0]
        policy = ARM_POLICY.get(arm)
        if policy is None:
            continue
        results = run_dir / "v1_9_resource_alloc_results.csv"
        if not results.exists():
            continue
        with results.open() as handle:
            for row in csv.DictReader(handle):
                if str(row.get("policy", "")) == policy:
                    per_arm.setdefault(arm, []).append({key: float(row.get(key, 0.0) or 0.0) for key in METRICS})
    out: dict[str, dict[str, float]] = {}
    for arm, rows in per_arm.items():
        out[arm] = {key: round(_mean([row[key] for row in rows]), 6) for key in METRICS}
        out[arm]["runs"] = float(len(rows))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/rl/ab_bubbles_v2")
    args = parser.parse_args()
    root = Path(args.root)
    summary = collect(root)
    lines = ["# BUBBLES A/B v2 summary", ""]
    lines.append("| arm | runs | " + " | ".join(METRICS) + " |")
    lines.append("|" + "---|" * (len(METRICS) + 2))
    for arm in ["Cgreedy", "Ccache", "A2", "B1", "B2", "A1"]:
        row = summary.get(arm)
        if row is None:
            lines.append(f"| {arm} | 0 | " + " | ".join(["-"] * len(METRICS)) + " |")
            continue
        lines.append(f"| {arm} | {int(row['runs'])} | " + " | ".join(f"{row[key]:.4f}" for key in METRICS) + " |")
    lines.append("")

    verdicts: list[tuple[str, bool | None, str]] = []
    a1, a2, b1 = summary.get("A1"), summary.get("A2"), summary.get("B1")
    if a2:
        conflict = a2["airspace_conflict_rate"]
        verdicts.append(("A2 conflict rate <= 0.10", conflict <= 0.10, f"{conflict:.4f}"))
        cache = a2["service_level_0_ratio"]
        verdicts.append(("A2 cache ratio <= 0.30", cache <= 0.30, f"{cache:.4f}"))
    if a1 and a2:
        drop = a1["semantic_success_rate"] - a2["semantic_success_rate"]
        verdicts.append(("A1 semSucc - A2 semSucc <= 0.03", drop <= 0.03, f"{drop:+.4f}"))
    if b1 and a2:
        delta = b1["airspace_conflict_rate"] - a2["airspace_conflict_rate"]
        verdicts.append(("B1 conflict - A2 conflict >= 0.05", delta >= 0.05, f"{delta:+.4f}"))

    lines.append("## Criteria")
    lines.append("")
    for name, ok, value in verdicts:
        status = "PASS" if ok else "FAIL"
        lines.append(f"- [{status}] {name} (observed {value})")
    if not verdicts:
        lines.append("- (insufficient arms completed)")
    lines.append("")

    out_md = root / "ab_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "ab_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
