#!/usr/bin/env python
"""Task #35 v7: escalation-budget re-calibration UNDER the corrected link model.

v6 sized epsilon/delta_esc with the reference link budget anchored on the 50 kHz
s0 default (0.05 x pool) and with below-support SINRs silently snapped to the
lowest calibrated LUT bin.  Both make the quality axis optimistic:

  * reference_bandwidth=fair_share  -- the obs SNR bin / spec-attainability
    certificate now anchor on pool / N_uav (each UAV's fair spectrum share), so
    the reference noise floor is realistic (was ~13 dB too low);
  * lut_support_guard=outage        -- a service whose EFFECTIVE SINR is below
    the lowest LUT bin by > 2.5 dB is a quality OUTAGE (LCB 0), not an unearned
    extrapolation of the -5 dB cell.

Everything else is IDENTICAL to calibrate_epsilon_v6 (same comm-window deadline
clock, same v5 epsilon anchors, same spec decomposition, same design gate).  The
epsilon ANCHORS come from the deadline/link-independent v5 LUT quantile and are
therefore UNCHANGED; only the spec-attainability decomposition (and hence
delta_esc) can move once the reference optimism and the extrapolation are gone.

--link-model legacy   reproduces the v6 numbers (reference 50 kHz, guard off).
--link-model corrected applies fair_share + outage (the v7 link model).
Run both and diff the JSONs for the before/after re-calibration report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vqa_semcom.config import load_config, resolve_path  # noqa: E402
from vqa_semcom.sim.resource_env import filter_tasks_supported_by_lut, load_lut  # noqa: E402
from vqa_semcom.sim.resource_env import read_csv  # noqa: E402

import calibrate_epsilon_v6 as V6  # noqa: E402
import calibrate_epsilon_v5 as V5  # noqa: E402


def _link_overrides(link_model: str) -> dict:
    if link_model == "corrected":
        return {"reference_bandwidth": "fair_share", "lut_support_guard": "outage"}
    # legacy: explicit defaults (reproduce v6 exactly)
    return {"reference_bandwidth": "legacy", "lut_support_guard": "off"}


def _comm_cfg_v7(tau_conf: int, peak_load, link_model: str):
    """v6 comm-window config PLUS the v7 link-model flags."""
    cfg, thr, peak_load = V6._comm_cfg(tau_conf, peak_load)
    eo = dict(cfg.get("multi_uav_env", {}))
    eo.update(_link_overrides(link_model))
    cfg["multi_uav_env"] = eo
    return cfg, thr, peak_load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default=None)
    ap.add_argument("--tau-conf", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--peak-load", type=float, default=None)
    ap.add_argument("--link-model", default="corrected", choices=["legacy", "corrected"])
    a = ap.parse_args()

    cfg, thr, peak_load = _comm_cfg_v7(a.tau_conf, a.peak_load, a.link_model)
    tasks = read_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    lut = load_lut(V6.LUT_CSV)
    tasks = filter_tasks_supported_by_lut(tasks, lut)
    if not tasks:
        raise RuntimeError("No tasks supported by the V1.9 LUT.")

    link_ov = _link_overrides(a.link_model)
    print(f"[v7] link_model={a.link_model} ({link_ov}) | tau_conf={a.tau_conf}sigma "
          f"tau_crit={thr['tau_critical_comm']} tau_norm={thr['tau_normal_comm']} peak_load={peak_load}")
    print("[v7] PEAK (utm_conflict) critical spec decomposition ...")
    peak_tx, peak_tasks = V6.collect(cfg, tasks, lut, "utm_conflict", risk="critical")
    print("[v7] NOMINAL critical spec decomposition ...")
    nom_tx, nom_crit_tasks = V6.collect(cfg, tasks, lut, "nominal", risk="critical")

    la = V5.lut_v5_anchors()
    eps_cc, eps_cp, eps_normal = la["eps_critical_counting"], la["eps_critical_presence"], la["eps_normal"]
    peak_dec = V5.spec_decompose(peak_tasks, eps_cc, eps_cp)
    nom_dec = V5.spec_decompose(nom_crit_tasks, eps_cc, eps_cp)
    delta_esc_peak = round(min(0.99, peak_dec["spec_unattainable"] + 0.05), 3)
    delta_esc_nom = round(min(0.99, (nom_dec["spec_unattainable"] if nom_dec["n"] else 0.0) + 0.05), 3)

    print("\n==== escalation-budget recalibration v7 (%s link) ====" % a.link_model)
    print(f"epsilon anchors (v5, unchanged): cc={eps_cc:.3f} cp={eps_cp:.3f} normal={eps_normal:.3f}")
    print(f"PEAK    critical spec decomposition: {peak_dec}")
    print(f"NOMINAL critical spec decomposition: {nom_dec}")
    print(f"delta_esc peak    = {delta_esc_peak:.3f}")
    print(f"delta_esc nominal = {delta_esc_nom:.3f}")
    peak_unattain = peak_dec["spec_unattainable"]
    in_band = 0.15 <= peak_unattain <= 0.5
    print(f"\n[design gate #33-C] peak spec-unattainable {peak_unattain:.3f} in [0.15,0.5]? "
          f"{'YES' if in_band else 'NO'}")
    print(f"[design gate #33-C] nominal escalation {nom_dec['spec_unattainable']:.3f} < 0.15? "
          f"{'YES' if nom_dec['spec_unattainable'] < 0.15 else 'NO'}")

    if a.emit:
        Path(a.emit).write_text(json.dumps({
            "link_model": a.link_model,
            "link_overrides": link_ov,
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
