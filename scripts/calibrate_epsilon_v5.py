#!/usr/bin/env python
"""Task #28 iteration 5: epsilon recalibration v5 -- attainable epsilon + escalation.

v4 (attainability_v4, 0.355/0.166) anchored eps_critical on a SINGLE pooled P10 of
the peak all-critical best-transmission LCB.  That flattened the two structurally
different critical clusters -- counting (hard, wide-tolerance, GT-dependent) and
presence/yes-no (easy) -- into one bar, and it still could not separate the
*quality* axis (some tasks whose best tx LCB never clears any sane bar) from the
*deadline* axis (tasks whose quality IS attainable but whose only compliant
transmission is deadline-blocked -- v4 diagnosed ~44% of the peak counting cluster
as deadline-killed, which no epsilon move can rescue).

v5 does three things:

  1.  Anchors epsilon on the v5 UNIFIED LUT (lut_v5 backend: qtype x svc x channel
      x snr x view x count-bucket, GT>=10 counting re-judged at +-20%), and splits
      the critical bar by qtype into a two-key table
      {(critical, counting), (critical, presence)} plus a single normal key.  The
      anchor rule is

          eps(subset) = floor_3dp( P25( best_tx_lcb : subset ) - 0.05 )

      best_tx_lcb(task) = max over tx levels {1 token, 2 image} of the candidate
      semantic_accuracy_lcb at the task's realised SNR obs under the lut_v5
      backend (s0 cache excluded; deadline-independent).

  2.  Separates the unreachable mass into a QUALITY axis (fraction of peak critical
      with no tx service clearing its eps) and a DEADLINE axis (quality attainable
      but no compliant tx is deadline-feasible), driving the environment to measure
      the deadline axis empirically (the LUT alone only bounds the quality axis).

  3.  Sizes the escalation dual budget

          delta_esc(condition) = measured spec-UNattainable fraction + 0.05

      where a critical task is spec-attainable iff SOME tx service is both
      quality-attainable (lcb >= eps_qtype) AND deadline-feasible in the realised
      run.  Peak and nominal are sized separately.

Pair the resulting attainability_v5 constants with --critical-cache-compliance
forbidden and --quality-backend lut_v5.
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
LUT_V5_CSV = ROOT / "outputs/lut/v5_unified_lut.csv"
DOC = ROOT / "docs/EPSILON_RECAL_V5.md"
V4_ANCHOR = 0.355
EVAL_EP = 50
TPE = 20
TX_LEVELS = (1, 2)          # 1=token, 2=image; s0 cache excluded (compliance-banned)
MARGIN = 0.05               # P25 - 0.05 attainable-epsilon margin
CRIT_COUNT_GT = 10          # counting-critical bucket = GT >= 10 (v5 +-20% tolerance)
GE10_BUCKETS = ("10-19", "20-49", "50+")


def quantile(vals: list[float], q: float) -> float:
    v = sorted(vals)
    n = len(v)
    idx = min(n - 1, max(0, int(math.ceil(q / 100.0 * n)) - 1))
    return v[idx]


def round_down3(x: float) -> float:
    return math.floor(x * 1000.0) / 1000.0


def summary(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    return (f"n={len(vals)} min={min(vals):.3f} P10={quantile(vals,10):.3f} "
            f"P25={quantile(vals,25):.3f} med={quantile(vals,50):.3f} "
            f"P90={quantile(vals,90):.3f} max={max(vals):.3f} mean={statistics.mean(vals):.3f}")


def lut_v5_anchors() -> dict:
    """Stage-1 anchor method: DIRECT v5-LUT quantile.  Per context cell
    (qtype, channel, snr, view, count-bucket) take best_tx_lcb = max over service
    {1 token, 2 image} of wilson_low; then eps(subset) = floor_3dp(P25 - 0.05).

    Subsets: (critical,counting) = counting cells with GT>=10 bucket; (critical,
    presence) = presence cells; normal (single key) = ALL qtype cells pooled
    (normal-risk tasks span every qtype and risk is not a LUT dimension).
    """
    import csv as _csv
    from collections import defaultdict as _dd
    byctx: dict[tuple, dict[int, float]] = _dd(dict)
    for r in _csv.DictReader(open(LUT_V5_CSV, encoding="utf-8")):
        try:
            svc = int(r["service_level"])
        except (KeyError, ValueError):
            continue
        if svc not in TX_LEVELS:
            continue
        k = (r["question_type"], r["channel_bin"], r["snr_bin"], r["view_quality_bin"], r["count_bucket"])
        byctx[k][svc] = float(r.get("wilson_low", 0.0) or 0.0)

    def best_over(pred) -> list[float]:
        return [max(d.values()) for k, d in byctx.items() if d and pred(k)]

    cc = best_over(lambda k: k[0] == "counting" and k[4] in GE10_BUCKETS)
    cp = best_over(lambda k: k[0] == "presence")
    allc = best_over(lambda k: True)
    return {
        "eps_critical_counting": round_down3(quantile(cc, 25) - MARGIN),
        "eps_critical_presence": round_down3(quantile(cp, 25) - MARGIN),
        "eps_normal": round_down3(quantile(allc, 25) - MARGIN),
        "n_counting_ge10": len(cc), "n_presence": len(cp), "n_all": len(allc),
        "p25_counting": quantile(cc, 25), "p25_presence": quantile(cp, 25), "p25_all": quantile(allc, 25),
        "quality_unreach_ceiling": None,
    }


def _args(scenario: str) -> SimpleNamespace:
    return SimpleNamespace(
        scenario=scenario, seed=0, snr_bins=None, tasks_per_episode=TPE,
        formal_scenario=None, state_version="v2", num_uavs=None,
        quality_backend="lut_v5", disable_semantic_token=False,
    )


def _is_counting_critical(obs: dict) -> bool:
    return str(obs.get("question_type", "")) == "counting" and int(obs.get("object_count", -1)) >= CRIT_COUNT_GT


def collect(cfg, tasks, lut, scenario, *, risk):
    """Drive the oracle trajectory (lut_v5 backend) and record, per critical/normal
    task step: best_tx_lcb by qtype-cluster, and the spec-attainability decomposition."""
    env = R.make_env(_args(scenario), cfg, tasks, lut, "oracle_best_feasible_evidence")
    tx_by_cluster: dict[str, list[float]] = {"counting_ge10": [], "presence_like": [], "all": []}
    # spec-attainability tallies (need eps, filled by caller in a second pass -> we
    # store per-task the per-service (lcb, deadline_ok) so eps can be applied later).
    per_task: list[dict] = []
    for ep in range(EVAL_EP):
        obs = env.reset(seed=ep, options={"policy_name": "oracle_best_feasible_evidence"})
        done = False
        while not done:
            cur = obs
            if str(cur.get("risk_level", "")) == risk and str(cur.get("question_type", "")) != "":
                svc = {}
                tau = float(cur.get("deadline_s", 0.0))          # full tau_k (change 5)
                for lvl in TX_LEVELS:
                    info = env.evaluate_action(env.candidate_action(lvl, cur), cur)
                    delay = float(info.get("delay_s", info.get("total_delay_s", 0.0)))
                    pipe = (float(info.get("tx_delay_s", 0.0)) + float(info.get("infer_delay_s", 0.0))
                            + float(info.get("queue_delay_s", 0.0)) + float(info.get("load_delay_s", 0.0))
                            + float(info.get("sense_delay_s", 0.0)))
                    svc[lvl] = (
                        float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0))),
                        bool(delay <= tau),   # full-flight deadline feasibility (spec certificate)
                        bool(pipe <= tau),    # pipeline-only deadline feasibility (informational)
                    )
                best = max(v[0] for v in svc.values())
                cluster = "counting_ge10" if _is_counting_critical(cur) else "presence_like"
                tx_by_cluster[cluster].append(best)
                tx_by_cluster["all"].append(best)
                per_task.append({
                    "qtype": str(cur.get("question_type", "")),
                    "counting_ge10": _is_counting_critical(cur),
                    "svc": svc,
                })
            obs, _r, done, _info = env.step(
                R.choose_baseline_action("oracle_best_feasible_evidence", env, cur))
    return tx_by_cluster, per_task


def anchor(vals: list[float]) -> float:
    return round_down3(quantile(vals, 25) - MARGIN) if vals else float("nan")


def eps_for(qtype: str, counting_ge10: bool, eps_cc: float, eps_cp: float) -> float:
    # counting-critical (GT>=10) uses the counting key; every other critical
    # qtype (presence / co_presence / comparison / threshold, and low-count
    # counting) uses the presence key.
    return eps_cc if (qtype == "counting" and counting_ge10) else eps_cp


def spec_decompose(per_task: list[dict], eps_cc: float, eps_cp: float) -> dict:
    """Decompose critical-task attainability.

    spec_attainable(task) := SOME tx service clears eps_qtype (quality axis) AND
    is deadline-feasible against the full tau_k WITH the realised nearest-UAV
    flight (change 5 certificate).  We also report the pipeline-only (comm+compute,
    flight-excluded) deadline axis to separate the semantic-service latency floor
    from the UAV-positioning geometry.
    """
    n = len(per_task)
    if n == 0:
        return {"n": 0, "quality_unreachable": float("nan"), "deadline_blocked": float("nan"),
                "pipeline_blocked": float("nan"), "spec_unattainable": float("nan")}
    q_unreach = d_block = pipe_block = unattain = 0
    for t in per_task:
        eps = eps_for(t["qtype"], t["counting_ge10"], eps_cc, eps_cp)
        q_services = [(dl_ok, pipe_ok) for (lcb, dl_ok, pipe_ok) in t["svc"].values() if lcb >= eps]
        quality_attainable = len(q_services) > 0
        spec = any(dl_ok for dl_ok, _ in q_services)         # quality AND full-flight deadline
        pipe_spec = any(pipe_ok for _, pipe_ok in q_services)  # quality AND pipeline deadline
        if not spec:
            unattain += 1
            if not quality_attainable:
                q_unreach += 1
            else:
                d_block += 1
        if quality_attainable and not pipe_spec:
            pipe_block += 1
    return {
        "n": n,
        "quality_unreachable": q_unreach / n,
        "deadline_blocked": d_block / n,       # quality-ok but flight-deadline-blocked
        "pipeline_blocked": pipe_block / n,    # quality-ok but comm-pipeline-deadline-blocked
        "spec_unattainable": unattain / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-doc", action="store_true")
    ap.add_argument("--emit", default=None, help="write the calibrated constants JSON here")
    a = ap.parse_args()

    cfg = load_config(CFG)
    tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut = load_lut(LUT_CSV)
    tasks = filter_tasks_supported_by_lut(tasks, lut)
    if not tasks:
        raise RuntimeError("No tasks supported by the V1.9 LUT.")

    print("[v5] PEAK (utm_conflict) critical best-tx LCB via lut_v5 ...")
    peak_tx, peak_tasks = collect(cfg, tasks, lut, "utm_conflict", risk="critical")
    print("[v5] NOMINAL critical best-tx LCB ...")
    nom_tx, nom_crit_tasks = collect(cfg, tasks, lut, "nominal", risk="critical")
    print("[v5] NOMINAL normal best-tx LCB ...")
    nomn_tx, nomn_tasks = collect(cfg, tasks, lut, "nominal", risk="normal")

    # Anchors come from the DIRECT v5-LUT quantile (stage-1 method), not the
    # env trajectory -- the env stream is used only to size the escalation budget.
    la = lut_v5_anchors()
    eps_cc = la["eps_critical_counting"]
    eps_cp = la["eps_critical_presence"]
    eps_normal = la["eps_normal"]

    peak_dec = spec_decompose(peak_tasks, eps_cc, eps_cp)
    nom_dec = spec_decompose(nom_crit_tasks, eps_cc, eps_cp)
    # escalation budget = measured spec-unattainable fraction + 0.05, clamped to a
    # valid rate (<1) so the dual channel has headroom and the admitted set is not
    # forced empty.
    delta_esc_peak = round(min(0.99, peak_dec["spec_unattainable"] + 0.05), 3)
    delta_esc_nom = round(min(0.99, (nom_dec["spec_unattainable"] if nom_dec["n"] else 0.0) + 0.05), 3)

    print("\n==== epsilon recalibration v5 ====")
    print("-- direct v5-LUT anchors (stage-1 method) --")
    print(f"counting_ge10: n_cells={la['n_counting_ge10']} P25={la['p25_counting']:.4f} -{MARGIN} -> eps_cc = {eps_cc:.3f}")
    print(f"presence:      n_cells={la['n_presence']} P25={la['p25_presence']:.4f} -{MARGIN} -> eps_cp = {eps_cp:.3f}")
    print(f"normal(all):   n_cells={la['n_all']} P25={la['p25_all']:.4f} -{MARGIN} -> eps_normal = {eps_normal:.3f}")
    print("-- env-realised best-tx distributions (context, informational) --")
    print(f"peak counting_ge10 best-tx: {summary(peak_tx['counting_ge10'])}")
    print(f"peak presence_like best-tx: {summary(peak_tx['presence_like'])}")
    print(f"nominal normal    best-tx: {summary(nomn_tx['all'])}")
    print(f"\nPEAK critical spec decomposition: {peak_dec}")
    print(f"NOMINAL critical spec decomposition: {nom_dec}")
    print(f"delta_esc peak    = spec_unattainable {peak_dec['spec_unattainable']:.3f} + 0.05 = {delta_esc_peak:.3f}")
    print(f"delta_esc nominal = {delta_esc_nom:.3f}")
    print(f"\nATTAINABILITY_V5_EPSILON = {{")
    print(f'    ("critical","counting"): {eps_cc:.3f},')
    print(f'    ("critical","presence"): {eps_cp:.3f},')
    print(f'    ("normal",):            {eps_normal:.3f},')
    print("}")

    if a.emit:
        import json
        Path(a.emit).write_text(json.dumps({
            "eps_critical_counting": eps_cc,
            "eps_critical_presence": eps_cp,
            "eps_normal": eps_normal,
            "delta_esc_peak": delta_esc_peak,
            "delta_esc_nominal": delta_esc_nom,
            "peak_decomposition": peak_dec,
            "nominal_decomposition": nom_dec,
            "peak_counting_ge10_summary": summary(peak_tx["counting_ge10"]),
            "peak_presence_summary": summary(peak_tx["presence_like"]),
            "nominal_normal_summary": summary(nomn_tx["all"]),
        }, indent=2))
        print(f"[emit] {a.emit}")


if __name__ == "__main__":
    main()
