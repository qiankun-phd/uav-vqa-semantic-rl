#!/usr/bin/env python
"""Task #33/#34 v6: escalation-budget re-calibration under COMM-WINDOW deadlines.

v5 sized the escalation dual budget delta_esc under the LEGACY full-flight
deadline, where a critical tau_k of 2.55 s (3.0 x tau_scale 0.85) could never fit
the ~13 s flight to the task area -> peak spec-unattainable ~= 1.0 (a scale
CLIFF, not physics) -> delta_esc_peak 0.90 pinned lambda_quality_critical.

v6 re-measures the SAME spec-attainability decomposition, but under
deadline_semantics="comm_window": the deadline clock charges only the tactical
comm-decision window (sense+tx+queue+infer+load) and excludes flight, and tau is
re-anchored to 2.8 s (critical) / 3.8 s (normal) on BUBBLES D2.1 Table G-2.  The
epsilon ANCHORS are unchanged (they come from the deadline-independent v5 LUT
quantile); only the escalation budget moves.

Outputs (JSON via --emit) become the SINGLE estimator for delta_esc (task
#34-(i)): the env's ESCALATION_DELTA_V5 dict is demoted to a legacy fallback and
the v6 matrix passes these calibrated values.  The decomposition separates:
  * quality_unreachable  -- no tx service clears eps_qtype (LUT quality floor);
  * pipeline_blocked     -- quality ok but the comm pipeline alone exceeds tau
                            (the meaningful deadline axis under comm_window);
  * spec_unattainable    -- neither axis leaves a feasible service (== the
                            escalation mass); delta_esc = this + 0.05.

Design knob (task #33-C): if the measured peak spec-unattainable falls outside
[0.15, 0.5] the ONLY sanctioned adjustment is the tau confidence band (0/1/2
sigma) or the peak arrival load, declared via --tau-conf / --peak-load and
recorded in the JSON.  No silent per-run tuning.

Pair with --deadline-semantics comm_window --epsilon-calibration attainability_v5
--critical-cache-compliance forbidden --quality-backend lut_v5.
"""
from __future__ import annotations

import argparse
import json
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

# Reuse the v5 pure helpers verbatim (anchors, quantiles, decomposition math) so
# the epsilon table and the spec-attainability logic stay identical to v5.
import calibrate_epsilon_v5 as V5  # noqa: E402

CFG = str(ROOT / "configs/v1_9_bubbles.yaml")
LUT_CSV = ROOT / "outputs/lut/v1_9_snr_semantic_quality_lut.csv"
EVAL_EP = V5.EVAL_EP
TPE = V5.TPE
TX_LEVELS = V5.TX_LEVELS
COMM_TAU = {"critical": 2.8, "high": 2.8, "normal": 3.8}


def _args(scenario: str) -> SimpleNamespace:
    return SimpleNamespace(
        scenario=scenario, seed=0, snr_bins=None, tasks_per_episode=TPE,
        formal_scenario=None, state_version="v2", num_uavs=None,
        quality_backend="lut_v5", disable_semantic_token=False,
    )


def _comm_cfg(tau_conf: int, peak_load: float | None):
    """Config with the escalation gate set AND deadline_semantics=comm_window.

    tau_conf selects the deadline confidence band in sigma units (task #33-C
    sanctioned knob): 1-sigma -> critical 2.8 / normal 3.8 (default); 0-sigma ->
    1.8 / 1.8 (tighter); 2-sigma -> 3.8 / 5.8 (looser).  peak_load, if given,
    scales tasks_per_episode on the peak arm only (the other sanctioned knob).
    """
    cfg = load_config(CFG)
    eo = dict(cfg.get("multi_uav_env", {}))
    eo.update({
        "epsilon_calibration": "attainability_v5",
        "critical_cache_compliance": "forbidden",
        "cache_quality": "entry_v2",
        "escalation_mode": "spec_attainable",
        "deadline_semantics": "comm_window",
    })
    # tau_conf band: mean 1.8 + tau_conf*sigma(1.0); normal = +1 extra sigma.
    thr = dict(cfg.get("thresholds", {}) or {})
    thr["tau_critical_comm"] = round(1.8 + tau_conf * 1.0, 3)
    thr["tau_normal_comm"] = round(1.8 + (tau_conf + 1) * 1.0, 3)
    cfg["thresholds"] = thr
    cfg["multi_uav_env"] = eo
    return cfg, thr, peak_load


def collect(cfg, tasks, lut, scenario, *, risk):
    """Comm-window variant of V5.collect: the certificate deadline axis subtracts
    flight (delay - fly), matching the env's _deadline_delay_s chokepoint, while
    the pipeline axis is the flight-excluded pipeline (identical here, reported
    for continuity)."""
    env = R.make_env(_args(scenario), cfg, tasks, lut, "oracle_best_feasible_evidence")
    tx_by_cluster: dict[str, list[float]] = {"counting_ge10": [], "presence_like": [], "all": []}
    per_task: list[dict] = []
    for ep in range(EVAL_EP):
        obs = env.reset(seed=ep, options={"policy_name": "oracle_best_feasible_evidence"})
        done = False
        while not done:
            cur = obs
            if str(cur.get("risk_level", "")) == risk and str(cur.get("question_type", "")) != "":
                svc = {}
                tau = float(cur.get("deadline_s", 0.0))  # comm-window tau_k (2.8 / 3.8)
                for lvl in TX_LEVELS:
                    info = env.evaluate_action(env.candidate_action(lvl, cur), cur)
                    delay = float(info.get("delay_s", info.get("total_delay_s", 0.0)))
                    fly = float(info.get("fly_delay_s", info.get("arrival_delay_s", 0.0)))
                    comm = max(0.0, delay - fly)  # deadline-charged delay under comm_window
                    lcb = float(info.get("semantic_accuracy_lcb", info.get("answer_accuracy_est", 0.0)))
                    svc[lvl] = (
                        lcb,
                        bool(comm <= tau),   # comm-window certificate deadline feasibility
                        bool(comm <= tau),   # pipeline == comm-charged under comm_window
                    )
                best = max(v[0] for v in svc.values())
                cluster = "counting_ge10" if V5._is_counting_critical(cur) else "presence_like"
                tx_by_cluster[cluster].append(best)
                tx_by_cluster["all"].append(best)
                per_task.append({
                    "qtype": str(cur.get("question_type", "")),
                    "counting_ge10": V5._is_counting_critical(cur),
                    "svc": svc,
                })
            obs, _r, done, _info = env.step(
                R.choose_baseline_action("oracle_best_feasible_evidence", env, cur))
    return tx_by_cluster, per_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default=None, help="write the calibrated constants JSON here")
    ap.add_argument("--tau-conf", type=int, default=1, choices=[0, 1, 2],
                    help="deadline confidence band in sigma units (task #33-C knob). default 1.")
    ap.add_argument("--peak-load", type=float, default=None,
                    help="override tasks-per-episode on the peak arm (task #33-C knob).")
    a = ap.parse_args()

    cfg, thr, peak_load = _comm_cfg(a.tau_conf, a.peak_load)
    tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut = load_lut(LUT_CSV)
    tasks = filter_tasks_supported_by_lut(tasks, lut)
    if not tasks:
        raise RuntimeError("No tasks supported by the V1.9 LUT.")

    print(f"[v6] comm-window escalation-budget calibration | tau_conf={a.tau_conf}sigma "
          f"tau_crit={thr['tau_critical_comm']} tau_norm={thr['tau_normal_comm']} peak_load={peak_load}")
    print("[v6] PEAK (utm_conflict) critical spec decomposition (comm-window) ...")
    peak_tx, peak_tasks = collect(cfg, tasks, lut, "utm_conflict", risk="critical")
    print("[v6] NOMINAL critical spec decomposition ...")
    nom_tx, nom_crit_tasks = collect(cfg, tasks, lut, "nominal", risk="critical")

    # epsilon anchors: identical to v5 (deadline-independent LUT quantile).
    la = V5.lut_v5_anchors()
    eps_cc = la["eps_critical_counting"]
    eps_cp = la["eps_critical_presence"]
    eps_normal = la["eps_normal"]

    peak_dec = V5.spec_decompose(peak_tasks, eps_cc, eps_cp)
    nom_dec = V5.spec_decompose(nom_crit_tasks, eps_cc, eps_cp)
    delta_esc_peak = round(min(0.99, peak_dec["spec_unattainable"] + 0.05), 3)
    delta_esc_nom = round(min(0.99, (nom_dec["spec_unattainable"] if nom_dec["n"] else 0.0) + 0.05), 3)

    print("\n==== escalation-budget recalibration v6 (comm-window) ====")
    print(f"epsilon anchors (unchanged from v5): cc={eps_cc:.3f} cp={eps_cp:.3f} normal={eps_normal:.3f}")
    print(f"PEAK    critical spec decomposition: {peak_dec}")
    print(f"NOMINAL critical spec decomposition: {nom_dec}")
    print(f"delta_esc peak    = spec_unattainable {peak_dec['spec_unattainable']:.3f} + 0.05 = {delta_esc_peak:.3f}")
    print(f"delta_esc nominal = {delta_esc_nom:.3f}")

    # Task #33-C design gate: peak spec-unattainable target [0.15, 0.5].
    peak_unattain = peak_dec["spec_unattainable"]
    in_band = 0.15 <= peak_unattain <= 0.5
    print(f"\n[design gate #33-C] peak spec-unattainable {peak_unattain:.3f} "
          f"in [0.15, 0.5]? {'YES' if in_band else 'NO -- adjust ONE knob (--tau-conf / --peak-load) and record'}")
    print(f"[design gate #33-C] nominal escalation {nom_dec['spec_unattainable']:.3f} < 0.15? "
          f"{'YES' if nom_dec['spec_unattainable'] < 0.15 else 'NO'}")

    if a.emit:
        Path(a.emit).write_text(json.dumps({
            "deadline_semantics": "comm_window",
            "tau_conf_sigma": a.tau_conf,
            "tau_critical_comm": thr["tau_critical_comm"],
            "tau_normal_comm": thr["tau_normal_comm"],
            "peak_load_override": peak_load,
            "eps_critical_counting": eps_cc,
            "eps_critical_presence": eps_cp,
            "eps_normal": eps_normal,
            "delta_esc_peak": delta_esc_peak,
            "delta_esc_nominal": delta_esc_nom,
            "peak_decomposition": peak_dec,
            "nominal_decomposition": nom_dec,
            "peak_design_gate_in_band": in_band,
            "peak_counting_ge10_summary": V5.summary(peak_tx["counting_ge10"]),
            "peak_presence_summary": V5.summary(peak_tx["presence_like"]),
        }, indent=2))
        print(f"[emit] {a.emit}")


if __name__ == "__main__":
    main()
