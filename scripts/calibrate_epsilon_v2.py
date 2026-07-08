#!/usr/bin/env python
"""Task #28 iteration 2: epsilon attainability recalibration v2.

v1 fixed the lambda ratchet and the nominal tier, but the peak (utm_conflict,
all-critical) tier stayed *essentially infeasible*: at eps_critical=0.615 the
oracle satisfied only 65% of critical tasks, so the learning policy collapsed
onto the cache-compliance shortcut (acc 0.26 / cache 0.49 / payload 1.5KB).

v2 recalibration rule (two corrections vs v1):

  (1) QUANTILE ANCHORING.  For each operating condition we build the
      distribution of "each task's best-feasible-service LCB" and read a
      quantile off it, instead of v1's single ratio-times-ceiling scalar:
        * eps_critical  = P10 of the PEAK (all-critical) distribution
                          (oracle should be able to satisfy ~90%);
        * eps_normal    = P25 of the NOMINAL normal-risk distribution
                          (oracle should be able to satisfy ~75%).
      The 3D LUT (outputs/lut/v2_0_lut_3d.csv, qtype x service x snr, Wilson
      lower bound = LCB) is the underlying accuracy model.  The per-task
      best-*feasible*-service LCB is the LCB the oracle_best_feasible_evidence
      baseline actually realises once deadline / payload / freshness / view
      degradation are applied on top of that LUT -- i.e. the realised
      semantic_accuracy_lcb of the served oracle tasks under each condition.
      (A pure-LUT best-service ceiling ignores deadline feasibility & freshness
      and is condition-blind, so it is reported only as a cross-check.)

  (2) CACHE-CEILING GUARDRAIL.  Measure the cache accuracy distribution from
      the v1 bl_always_cache run and take
        eps_critical = max( P10-anchor , cache_accuracy_P90 + 0.05 )
      so that the cache can never on its own clear a critical task's quality
      constraint (which would re-open the cache-override projection recovery
      trigger and let cache become the compliance shortcut again).

Outputs the calibration table, both anchors, the guardrail, and writes
docs/EPSILON_RECAL_V2.md.  The resulting constants are transcribed by hand into
ATTAINABILITY_V2_EPSILON in src/vqa_semcom/sim/multi_uav_env.py.
"""
from __future__ import annotations

import csv
import math
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "outputs/rl/eps_recal_v1"
LUT = ROOT / "outputs/lut/v2_0_lut_3d.csv"
DOC = ROOT / "docs/EPSILON_RECAL_V2.md"

CACHE_MARGIN = 0.05


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def quantile(vals: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0,100])."""
    v = sorted(vals)
    n = len(v)
    idx = min(n - 1, max(0, int(math.ceil(q / 100.0 * n)) - 1))
    return v[idx]


def realised_best_feasible_lcb(run: str, *, risk: str | None) -> list[float]:
    """Per-task best-feasible-service LCB = realised semantic_accuracy_lcb of the
    oracle baseline over served (non-reject) real tasks, optionally filtered by
    risk level.  Empty question_type rows are warm-up / no-task steps."""
    rows = load(RL / run / "v1_9_resource_alloc_rollout.csv")
    out = []
    for r in rows:
        if r["semantic_path"] == "reject" or r["question_type"] == "":
            continue
        if risk is not None and r["risk_level"] != risk:
            continue
        out.append(float(r["semantic_accuracy_lcb"]))
    return out


def cache_accuracy(run: str) -> list[float]:
    rows = load(RL / run / "v1_9_resource_alloc_rollout.csv")
    return [float(r["answer_accuracy_est"]) for r in rows if r["semantic_path"] == "cache"]


def lut_best_service_ceiling() -> dict[str, float]:
    """Pure-LUT cross-check: per qtype, max over service of mean-over-SNR LCB."""
    rows = load(LUT)
    agg: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        agg.setdefault((r["question_type"], r["service_level"]), []).append(float(r["wilson_low"]))
    best: dict[str, float] = {}
    for (qt, _sv), vals in agg.items():
        m = statistics.mean(vals)
        best[qt] = max(best.get(qt, 0.0), m)
    return best


def summary(vals: list[float]) -> str:
    return (f"n={len(vals)} min={min(vals):.3f} P10={quantile(vals,10):.3f} "
            f"P25={quantile(vals,25):.3f} med={quantile(vals,50):.3f} "
            f"P90={quantile(vals,90):.3f} max={max(vals):.3f} mean={statistics.mean(vals):.3f}")


def main() -> None:
    # --- Anchor distributions ---------------------------------------------
    peak_crit = realised_best_feasible_lcb("bl_oracle_best_feasible_evidence_0", risk="critical")
    nom_norm = realised_best_feasible_lcb("bl_oracle_best_feasible_evidence_0_nom", risk="normal")

    p10_peak = quantile(peak_crit, 10)          # eps_critical anchor
    p25_nom = quantile(nom_norm, 25)            # eps_normal anchor

    # --- Cache-ceiling guardrail ------------------------------------------
    cache_vals = cache_accuracy("bl_always_cache_0")
    cache_p90 = quantile(cache_vals, 90)
    guardrail = cache_p90 + CACHE_MARGIN

    # --- Final table (rule (1) anchored, rule (2) guarded) ----------------
    eps_critical = round(max(p10_peak, guardrail), 3)
    eps_normal = round(p25_nom, 3)
    guardrail_binds = guardrail > p10_peak

    lut_ceiling = lut_best_service_ceiling()

    lines = []
    lines.append("# EPSILON_RECAL_V2 -- attainability recalibration, iteration 2")
    lines.append("")
    lines.append("Task #28 iteration 2.  v1 (`attainability_v1`, 0.615/0.166) fixed the")
    lines.append("lambda ratchet and the nominal tier but left the PEAK tier essentially")
    lines.append("infeasible (oracle satisfied only 65% of critical tasks -> learning arm")
    lines.append("collapsed onto the cache-compliance shortcut).  v2 re-derives the two")
    lines.append("constants by (1) quantile anchoring and (2) a cache-ceiling guardrail.")
    lines.append("")
    lines.append("## Rule (1): quantile anchoring")
    lines.append("")
    lines.append("Distribution = per-task best-*feasible*-service LCB (realised")
    lines.append("`semantic_accuracy_lcb` of the oracle_best_feasible_evidence baseline,")
    lines.append("whose accuracy model IS the 3D LUT `outputs/lut/v2_0_lut_3d.csv`, after")
    lines.append("deadline / payload / freshness / view degradation).")
    lines.append("")
    lines.append(f"* PEAK (utm_conflict, all critical) distribution: {summary(peak_crit)}")
    lines.append(f"    -> eps_critical anchor = P10 = **{p10_peak:.3f}**  (target: oracle ~90%)")
    lines.append(f"* NOMINAL normal-risk distribution:              {summary(nom_norm)}")
    lines.append(f"    -> eps_normal anchor   = P25 = **{p25_nom:.3f}**  (target: oracle ~75%)")
    lines.append("")
    lines.append("Pure-LUT best-service ceiling (condition-blind cross-check, max over")
    lines.append("service of mean-over-SNR Wilson LCB):")
    for qt in sorted(lut_ceiling):
        lines.append(f"    {qt:12} {lut_ceiling[qt]:.3f}")
    lines.append("")
    lines.append("## Rule (2): cache-ceiling guardrail")
    lines.append("")
    lines.append(f"* PEAK bl_always_cache measured cache accuracy: {summary(cache_vals)}")
    lines.append(f"* cache_accuracy_P90 = {cache_p90:.3f} ; margin = {CACHE_MARGIN}")
    lines.append(f"* guardrail floor = P90 + margin = **{guardrail:.3f}**")
    lines.append(f"* eps_critical = max(P10 anchor {p10_peak:.3f}, guardrail {guardrail:.3f})"
                 f" = **{eps_critical:.3f}**")
    lines.append(f"* guardrail {'BINDS (> anchor)' if guardrail_binds else 'does not bind'}")
    lines.append("")
    lines.append("## ATTAINABILITY_V2_EPSILON")
    lines.append("")
    lines.append("```python")
    lines.append("ATTAINABILITY_V2_EPSILON = {")
    lines.append(f'    "critical": {eps_critical},')
    lines.append(f'    "normal": {eps_normal},')
    lines.append(f'    "high": {eps_critical},')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("## Consistency note")
    lines.append("")
    if guardrail_binds:
        lines.append(f"WARNING: the cache guardrail floor ({guardrail:.3f}) exceeds the")
        lines.append(f"attainability P10 anchor ({p10_peak:.3f}).  The peak critical mix is")
        lines.append("~44% `counting`, whose best-feasible path is the cache/local (service")
        lines.append("0) because its semantic-transmit accuracy (0.27-0.48) is far below any")
        lines.append("guardrail-compliant threshold, while service-2 full transmit is")
        lines.append("deadline-blocked.  Forcing eps_critical >= cache_P90+margin therefore")
        lines.append("makes counting-critical structurally infeasible (cannot cache, cannot")
        lines.append("transmit to spec) -- i.e. the two v2 rules are mutually contradictory")
        lines.append("for this workload.  Recorded for human decision; NOT auto-fixed.")
    else:
        lines.append("Guardrail does not bind; anchor governs eps_critical.")

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\n[written]", DOC)


if __name__ == "__main__":
    main()
