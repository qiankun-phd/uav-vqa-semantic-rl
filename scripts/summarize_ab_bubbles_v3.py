#!/usr/bin/env python
"""Aggregate the BUBBLES A/B v3 chain (dual evaluation conditions) and check criteria.

v3 evaluation fixes relative to summarize_ab_bubbles_v2.py:
  * ``average_accuracy`` is the PRIMARY accuracy metric.  Under the all-critical
    ``utm_conflict`` condition every task carries epsilon_critical=0.82, which no
    policy reaches, so ``semantic_success_rate`` saturates at 0 for every arm and
    carries no signal there (it stays meaningful under ``nominal``).
  * Dual conditions per arm: ``<ARM>_<seed>`` dirs hold the peak
    (scenario=utm_conflict) evaluation; ``<ARM>_<seed>_nom`` dirs hold the
    nominal (bubbles_daily arrivals, no background intents) evaluation of the
    SAME checkpoint.
  * ``lambda_conflict`` end-of-training values are extracted per seed from
    ``ppo_lambda_trace.csv``.

Criteria (pre-registered for the v3 calibration round):
  1. B2 conflict - A2 conflict >= 0.05   (the dual channel is load-bearing)
  2. A2 conflict <= 0.15                 (pulled by conflict_cost_limit=0.08)
  3. A2 cache ratio <= 0.35
  4. A2 average_accuracy >= A1 average_accuracy - 0.02
  5. B1 conflict - A2 conflict >= 0.05   (slow-head effect survives recalibration)
  6. nominal A2 semantic success >= 0.92 and task success >= 0.30
     (reference: v2 nominal smoke levels semSucc 0.92-0.94 / success 0.07-0.30)
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

ARM_ORDER = ["Cgreedy", "Ccache", "A2", "B2", "B1", "A1"]

METRICS = [
    "average_accuracy",
    "airspace_conflict_rate",
    "utm_conflict_violation_rate",
    "semantic_success_rate",
    "task_success_rate",
    "service_level_0_ratio",
    "service_level_1_ratio",
    "service_level_2_ratio",
    "deadline_violation_rate",
]


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _parse_run_dir(name: str) -> tuple[str, str, str] | None:
    """Return (arm, seed, condition) for ``ARM_SEED`` or ``ARM_SEED_nom``."""
    condition = "peak"
    if name.endswith("_nom"):
        condition = "nominal"
        name = name[: -len("_nom")]
    if "_" not in name:
        return None
    arm, seed = name.rsplit("_", 1)
    if arm not in ARM_POLICY:
        return None
    return arm, seed, condition


def _final_lambda_conflict(run_dir: Path) -> float | None:
    trace = run_dir / "ppo_lambda_trace.csv"
    if not trace.exists():
        return None
    last: dict[str, str] | None = None
    with trace.open() as handle:
        for row in csv.DictReader(handle):
            last = row
    if not last or "lambda_conflict" not in last:
        return None
    try:
        return float(last["lambda_conflict"])
    except (TypeError, ValueError):
        return None


def collect(root: Path) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    per_arm: dict[str, dict[str, list[dict[str, float]]]] = {}
    lambdas: dict[str, dict[str, float]] = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = _parse_run_dir(run_dir.name)
        if parsed is None:
            continue
        arm, seed, condition = parsed
        results = run_dir / "v1_9_resource_alloc_results.csv"
        if not results.exists():
            continue
        policy = ARM_POLICY[arm]
        with results.open() as handle:
            for row in csv.DictReader(handle):
                if str(row.get("policy", "")) == policy:
                    per_arm.setdefault(condition, {}).setdefault(arm, []).append(
                        {key: float(row.get(key, 0.0) or 0.0) for key in METRICS}
                    )
        if condition == "peak" and policy == "ppo":
            lam = _final_lambda_conflict(run_dir)
            if lam is not None:
                lambdas.setdefault(arm, {})[seed] = round(lam, 4)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for condition, arms in per_arm.items():
        out[condition] = {}
        for arm, rows in arms.items():
            out[condition][arm] = {key: round(_mean([row[key] for row in rows]), 6) for key in METRICS}
            out[condition][arm]["runs"] = float(len(rows))
    return out, lambdas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/rl/ab_bubbles_v3")
    args = parser.parse_args()
    root = Path(args.root)
    summary, lambdas = collect(root)

    lines = ["# BUBBLES A/B v3 summary", ""]
    lines.append(
        "> Primary accuracy metric: `average_accuracy`. `semantic_success_rate` saturates"
    )
    lines.append(
        "> to 0 for every arm under the all-critical `utm_conflict` condition"
    )
    lines.append(
        "> (epsilon_critical=0.82 is unreachable there), so it is reported for"
    )
    lines.append("> completeness and only carries signal under `nominal`.")
    lines.append("")
    for condition, title in (("peak", "Peak condition (scenario=utm_conflict)"), ("nominal", "Nominal condition (bubbles_daily arrivals, no background intents)")):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| arm | runs | " + " | ".join(METRICS) + " |")
        lines.append("|" + "---|" * (len(METRICS) + 2))
        arms = summary.get(condition, {})
        for arm in ARM_ORDER:
            row = arms.get(arm)
            if row is None:
                lines.append(f"| {arm} | 0 | " + " | ".join(["-"] * len(METRICS)) + " |")
                continue
            lines.append(
                f"| {arm} | {int(row['runs'])} | " + " | ".join(f"{row[key]:.4f}" for key in METRICS) + " |"
            )
        lines.append("")

    lines.append("## Final lambda_conflict per seed (end of training)")
    lines.append("")
    if lambdas:
        for arm in ARM_ORDER:
            if arm in lambdas:
                per_seed = ", ".join(f"seed{seed}={value:.3f}" for seed, value in sorted(lambdas[arm].items()))
                lines.append(f"- {arm}: {per_seed}")
    else:
        lines.append("- (no lambda traces found)")
    lines.append("")

    peak = summary.get("peak", {})
    nominal = summary.get("nominal", {})
    a1, a2 = peak.get("A1"), peak.get("A2")
    b1, b2 = peak.get("B1"), peak.get("B2")
    a2_nom = nominal.get("A2")

    verdicts: list[tuple[str, bool, str]] = []
    if a2 and b2:
        delta = b2["airspace_conflict_rate"] - a2["airspace_conflict_rate"]
        verdicts.append(("PRIMARY: B2 conflict - A2 conflict >= 0.05 (dual channel load-bearing)", delta >= 0.05, f"{delta:+.4f}"))
    if a2:
        conflict = a2["airspace_conflict_rate"]
        verdicts.append(("A2 conflict rate <= 0.15", conflict <= 0.15, f"{conflict:.4f}"))
        cache = a2["service_level_0_ratio"]
        verdicts.append(("A2 cache ratio <= 0.35", cache <= 0.35, f"{cache:.4f}"))
    if a1 and a2:
        margin = a2["average_accuracy"] - (a1["average_accuracy"] - 0.02)
        verdicts.append(("A2 average_accuracy >= A1 average_accuracy - 0.02", margin >= 0.0, f"margin {margin:+.4f}"))
    if b1 and a2:
        delta = b1["airspace_conflict_rate"] - a2["airspace_conflict_rate"]
        verdicts.append(("B1 conflict - A2 conflict >= 0.05 (slow-head effect)", delta >= 0.05, f"{delta:+.4f}"))
    if a2_nom:
        sem = a2_nom["semantic_success_rate"]
        succ = a2_nom["task_success_rate"]
        verdicts.append(("nominal A2 semantic success >= 0.92", sem >= 0.92, f"{sem:.4f}"))
        verdicts.append(("nominal A2 task success >= 0.30", succ >= 0.30, f"{succ:.4f}"))

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
    payload = {"conditions": summary, "lambda_conflict_final": lambdas}
    (root / "ab_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
