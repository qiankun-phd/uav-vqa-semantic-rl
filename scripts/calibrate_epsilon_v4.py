#!/usr/bin/env python
"""Task #28 iteration 4: epsilon attainability recalibration v4 -- TRANSMISSION-ONLY anchor.

v3 (attainability_v3, 0.504/0.166, cache-compliance FORBIDDEN) proved that the
0.504 anchor was still derived from a distribution that CONTAINED the s0 cache in
the feasible service set (the v2 P10 of the oracle's realised best-*feasible*-
service LCB, oracle free to pick cache).  Once cache is banned from COMPLIANCE
(v3), ~44% of the peak all-critical mix (counting-heavy) has NO transmission path
reaching 0.504 -> the peak oracle collapses to semSucc 0.556 (111/250 critical
tasks fall back to the non-compliant cache) and every learning arm collapses
(semSucc 0.008, lambda pinned).

v4 re-anchors on the PURE-TRANSMISSION feasible set.  For each PEAK all-critical
task we take the best TRANSMISSION-service quality lower bound:

    best_tx_lcb(task) = max over transmission service levels L in {1 (token),
                        2 (image)} of the candidate semantic_accuracy_lcb at the
                        task's realised sensed-SNR obs.

The transmission LCB is a LUT/SNR quantity and is deadline-INDEPENDENT (the
deadline only sets the per-step feasibility flag, not the accuracy value); s0
cache is excluded from the max because it is compliance-banned.  The anchor is

    eps_critical(v4) = P10 of {best_tx_lcb(task) : peak critical tasks}
                       (rounded DOWN to 3 dp so the P10 cluster stays compliant).

This is anchor-independent (max over tx LCB ignores the epsilon bar) -> no
circularity.  eps_normal is held at the v3/v1 attainability anchor 0.166 (nominal
normal tasks are unaffected by the cache ban).

The doc also reports:
  * PEAK semSucc-by-construction proxy = frac peak critical with
    best_tx_lcb >= eps_critical (LCB side; the DEADLINE "seam" -- whether that tx
    is deadline-feasible in the realised run -- is adjudicated by the matrix
    oracle, criterion 1);
  * NOMINAL critical transmission reachability = frac nominal critical with
    best_tx_lcb >= eps_critical (collateral check; < 0.90 => recorded diagnostic,
    NOT auto-fixed);
  * invariant eps_critical(v4) <= 0.504 (pure-tx anchor <= cache-inclusive
    anchor).

Pair the resulting attainability_v4 constants with --critical-cache-compliance
forbidden in the run matrix.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vqa_semcom.config import load_config, resolve_path  # noqa: E402
from vqa_semcom.sim.resource_env import (  # noqa: E402
    filter_tasks_supported_by_lut,
    load_lut,
    read_csv,
)
import run_v1_9_resource_alloc as R  # noqa: E402

CFG = str(ROOT / "configs/v1_9_bubbles.yaml")
LUT_CSV = ROOT / "outputs/lut/v1_9_snr_semantic_quality_lut.csv"
DOC = ROOT / "docs/EPSILON_RECAL_V4.md"
V3_ANCHOR = 0.504
EPS_NORMAL = 0.166
EVAL_EP = 50
TPE = 20
TX_LEVELS = (1, 2)  # 1=token, 2=image; s0 cache (level 0) excluded (compliance-banned)


def quantile(vals: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0,100]) -- identical to calibrate v2."""
    v = sorted(vals)
    n = len(v)
    idx = min(n - 1, max(0, int(math.ceil(q / 100.0 * n)) - 1))
    return v[idx]


def round_down3(x: float) -> float:
    return math.floor(x * 1000.0) / 1000.0


def summary(vals: list[float]) -> str:
    return (f"n={len(vals)} min={min(vals):.3f} P05={quantile(vals,5):.3f} "
            f"P10={quantile(vals,10):.3f} P25={quantile(vals,25):.3f} "
            f"med={quantile(vals,50):.3f} P90={quantile(vals,90):.3f} "
            f"max={max(vals):.3f} mean={statistics.mean(vals):.3f}")


def _args(scenario: str) -> SimpleNamespace:
    return SimpleNamespace(
        scenario=scenario, seed=0, snr_bins=None, tasks_per_episode=TPE,
        formal_scenario=None, state_version="v2", num_uavs=None,
        quality_backend=None, disable_semantic_token=False,
    )


def collect_best_tx_lcb(cfg: dict, tasks: list, lut: dict, scenario: str,
                        *, risk: str) -> tuple[list[float], dict[str, list[float]]]:
    """Drive the standard oracle_best_feasible_evidence trajectory (matching the
    v2/v3 obs/SNR stream) and, at every task step of the requested risk, record
    the best TRANSMISSION-service LCB max(token, image).  Deadline-independent."""
    env = R.make_env(_args(scenario), cfg, tasks, lut, "oracle_best_feasible_evidence")
    out: list[float] = []
    by_qt: dict[str, list[float]] = {}
    for ep in range(EVAL_EP):
        obs = env.reset(seed=ep, options={"policy_name": "oracle_best_feasible_evidence"})
        done = False
        while not done:
            cur = obs
            if str(cur.get("risk_level", "")) == risk and str(cur.get("question_type", "")) != "":
                best = max(float(env.candidate_metrics(int(lvl), cur)["accuracy"]) for lvl in TX_LEVELS)
                out.append(best)
                by_qt.setdefault(str(cur["question_type"]), []).append(best)
            obs, _r, done, _info = env.step(
                R.choose_baseline_action("oracle_best_feasible_evidence", env, cur))
    return out, by_qt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-doc", action="store_true")
    a = ap.parse_args()

    cfg = load_config(CFG)
    tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut = load_lut(LUT_CSV)
    tasks = filter_tasks_supported_by_lut(tasks, lut)
    if not tasks:
        raise RuntimeError("No tasks supported by the V1.9 LUT.")

    print("[v4] PEAK (utm_conflict) critical best-tx LCB ...")
    peak_crit, peak_qt = collect_best_tx_lcb(cfg, tasks, lut, "utm_conflict", risk="critical")
    print("[v4] NOMINAL critical best-tx LCB ...")
    nom_crit, nom_crit_qt = collect_best_tx_lcb(cfg, tasks, lut, "nominal", risk="critical")
    print("[v4] NOMINAL normal best-tx LCB (record) ...")
    nom_norm, _ = collect_best_tx_lcb(cfg, tasks, lut, "nominal", risk="normal")

    p10_peak = quantile(peak_crit, 10)
    eps_critical = round_down3(p10_peak)   # round DOWN so the P10 cluster stays compliant
    eps_normal = EPS_NORMAL

    peak_reach = sum(v >= eps_critical for v in peak_crit) / len(peak_crit)
    nom_reach = (sum(v >= eps_critical for v in nom_crit) / len(nom_crit)) if nom_crit else float("nan")
    invariant_ok = eps_critical <= V3_ANCHOR

    def qtmean(byqt):
        return {k: round(statistics.mean(v), 3) for k, v in sorted(byqt.items())}

    def qtcount(byqt):
        return {k: len(v) for k, v in sorted(byqt.items())}

    L = []
    L.append("# EPSILON_RECAL_V4 -- attainability recalibration, iteration 4 (transmission-only anchor)")
    L.append("")
    L.append("Task #28 iteration 4.  v3 (`attainability_v3`, 0.504/0.166, cache-compliance")
    L.append("FORBIDDEN) proved 0.504 was derived from a distribution that STILL contained")
    L.append("the s0 cache in the feasible service set.  Once cache is banned from")
    L.append("COMPLIANCE, ~44% of the peak all-critical mix (counting-heavy) has NO")
    L.append("transmission path reaching 0.504 -> peak oracle collapses to semSucc 0.556")
    L.append("(cache fallback non-compliant) and every learning arm collapses (semSucc")
    L.append("0.008, lambda pinned).")
    L.append("")
    L.append("## v4 rule: pure-transmission feasible-set anchor")
    L.append("")
    L.append("best_tx_lcb(task) = max over tx levels {1 token, 2 image} of the candidate")
    L.append("semantic_accuracy_lcb at the task's realised sensed-SNR obs (s0 cache excluded;")
    L.append("the LCB is a LUT/SNR quantity, deadline-independent).")
    L.append("")
    L.append(f"    eps_critical(v4) = floor_3dp( P10( best_tx_lcb : peak critical ) )")
    L.append("")
    L.append(f"* PEAK critical best-tx LCB:   {summary(peak_crit)}")
    L.append(f"    by qtype mean={qtmean(peak_qt)} counts={qtcount(peak_qt)}")
    L.append(f"    -> P10 = {p10_peak:.6f} -> floor_3dp -> eps_critical = **{eps_critical:.3f}**")
    L.append(f"* NOMINAL critical best-tx LCB: {summary(nom_crit)}")
    L.append(f"    by qtype mean={qtmean(nom_crit_qt)} counts={qtcount(nom_crit_qt)}")
    L.append(f"* NOMINAL normal best-tx LCB (record): {summary(nom_norm)}")
    L.append("")
    L.append("## ATTAINABILITY_V4_EPSILON")
    L.append("")
    L.append("```python")
    L.append("ATTAINABILITY_V4_EPSILON = {")
    L.append(f'    "critical": {eps_critical:.3f},')
    L.append(f'    "normal": {eps_normal:.3f},')
    L.append(f'    "high": {eps_critical:.3f},')
    L.append("}")
    L.append("```")
    L.append("")
    L.append("## Construction / collateral checks")
    L.append("")
    L.append(f"* invariant  eps_critical(v4) {eps_critical:.3f} <= cache-inclusive v3 anchor "
             f"{V3_ANCHOR}  ->  {'OK' if invariant_ok else 'VIOLATED'}")
    L.append(f"* PEAK semSucc-by-construction proxy (frac peak critical best_tx_lcb >= "
             f"{eps_critical:.3f}, LCB side): **{peak_reach:.3f}**")
    L.append(f"    NOTE: the DEADLINE seam (is that tx deadline-feasible in the realised")
    L.append(f"    run?) is adjudicated by the matrix oracle semSucc (criterion 1).")
    L.append(f"* NOMINAL critical transmission reachability (frac nominal critical best_tx_lcb "
             f">= {eps_critical:.3f}): **{nom_reach:.3f}**"
             + ("  [< 0.90 DIAGNOSTIC, not auto-fixed]" if (nom_reach == nom_reach and nom_reach < 0.90) else ""))
    L.append("")

    text = "\n".join(L) + "\n"
    print("\n" + text)
    print(f"[v4] eps_critical={eps_critical:.3f} eps_normal={eps_normal:.3f} "
          f"invariant_ok={invariant_ok} peak_reach={peak_reach:.3f} nom_reach={nom_reach:.3f}")

    if a.write_doc:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(text)
        print(f"[written] {DOC}")


if __name__ == "__main__":
    main()
