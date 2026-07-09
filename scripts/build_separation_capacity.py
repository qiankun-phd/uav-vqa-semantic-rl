#!/usr/bin/env python3
r"""E7 -- Semantic evidence selection -> comm delay (T4) -> separation minima d_TC -> airspace capacity.

This script instantiates the SESAR BUBBLES D2.1 "Block 4: Separation minima" chain
(section 3.3.4, pp.60-62) and injects our measured UAV-VQA end-to-end transmission
latency into the *separation-communication* component of the tactical delay T4.  The
point is to make "which evidence the UAV sends" (raw image vs semantic tokens vs an
adaptive policy) a *quantitative flight-safety variable*: heavier evidence -> longer
airtime -> larger T4 -> larger tactical-conflict separation distance d_TC -> lower
airspace capacity (Appendix F, pp.119-129).

------------------------------------------------------------------------------------
BLOCK 4 CHAIN (horizontal, cruise phase; PDF pp.60-61)
------------------------------------------------------------------------------------
    d_NMAC = 25 ft = 7.62 m                              (h_NMAC = 7.5 ft = 2.286 m)
    d_IC   = d_NMAC + TSE_h(95%) + Vc*(T1 + T2)          (imminent collision)
    d_SL   = d_IC   + SSE_h      + Vc*T_res              (separation loss)
    d_TC   = d_SL   + Vc*(T3 + T4)                       (tactical conflict)   <-- target
Vertical (Vv = vertical reference speed, all traffic classes identical in the example):
    h_IC   = h_NMAC + TSE_v(95%) + Vv*T2
    h_SL   = h_IC   + SSE_v      + Vv*T_res
    h_TC   = h_SL   + Vv*T4
T1 = h_avoid/ROC_eff (climb time to a safe vertical gap at IC level),
T3 = h_SL   /ROC_eff (climb time at TC level).  Only T1, T3 depend on the traffic
class through the cruise speed Vc and the rate of climb ROC (Table B-2, p.99).

T4 = tracking-delay + separation-communication-delay + pilot-delay, each a normal
r.v. (Table G-2, p.124); their sum is normal with
    T4_mean = 2.40 + 1.80 + 3.62 = 7.82 s ,   T4_sigma = sqrt(2.2^2+1.0^2+3.48^2) = 4.237 s .
A confidence band z in {0,1,2} sigma is applied (WP5 used 2 sigma = 0.977; p.62).

------------------------------------------------------------------------------------
CALIBRATION CONSTANTS RECOVERED FROM THE WORKED EXAMPLE (Appendix G)
------------------------------------------------------------------------------------
D2.1 tabulates the error terms (Table G-3) and the T4 delays (Table G-2) but NOT the
intermediate time windows T2, T_res, nor the exact climb-rate convention.  We recover
them from the published Table G-4/G-5 numbers (they are constant across the example):

    Vv        = 1.0  m/s      (vertical reference speed used in the example)
    T2        = 3.004 s       = h_IC - h_NMAC - TSE_v(2sig)  from Table G-5 (Vv=1)
    T_res     = 4.23  s       = h_SL - h_IC - SSE_v(2sig)    from Table G-5 (~= sigma(T4))
    ROC_off   = 0.5  m/s      effective climb rate = ROC - 0.5 reproduces Table G-4 to
                              <0.1%; the 0.5 m/s offset (undocumented in D2.1) is read
                              back from the example and is consistent for every class.
    h_avoid   = h_NMAC + TSE_v(2sig)  (vertical safe gap that defines T1)

--selftest reproduces Table G-4 (horizontal) and G-5 (vertical) from Table G-2/G-3
inputs and asserts every cell is within 1 %.

------------------------------------------------------------------------------------
SEMANTIC-COMMUNICATION MAPPING CALIBRE  (honest statement of assumptions)
------------------------------------------------------------------------------------
BUBBLES' separation-communication delay (mean 1.80 s, Table G-2) lumps: "communication
from the tracker if needed, time to formulate the solution, communication to the
(auto)pilot" -- i.e. the time to *get the evidence to the separation decision maker and
the instruction back out* on the tactical loop.

We substitute this mean by our measured evidence-delivery latency
        comm = upload_s + txside_s
where  upload_s  = airtime to deliver the evidence over the A2G link
                   (= mean channel-uses / bandwidth, from the comparison pipeline),
       txside_s  = transmitter-side semantic extraction (detector) latency (token path).
Everything else in T4 (tracking mean 2.40 s, pilot mean 3.62 s, and ALL sigmas,
including the separator sigma 1.0 s) is kept at the Table G-2 default -- we replace the
*mean* of the separation-communication component only.

Assumptions we own (not from D2.1):
  * We equate "sensor->decision evidence delivery" with the separation-comm delay.  We
    deliberately EXCLUDE the receiver-side VLM inference (inference_s): that is the
    downstream task, not the separation loop.  Including it would only enlarge the
    image penalty, so our choice is conservative for the semantic-token story.
  * The three traffic classes in Table B-2 are physical-airframe classes, not comm
    methods.  Our M1/M3/M4 differ only in the evidence they send, so we hold the
    airframe class fixed (default SAIL I-II: Vc=12 m/s, ROC=4 m/s) and vary ONLY the
    T4 comm mean with (service, snr, channel).
  * Capacity bridge (Appendix F): N_ref-1 = F(MAC)_max/(RF_MAC*RR*FAT_MAC).  In the
    gas-kinetic conflict model D2.1 uses (Appendix E, rate ~ N(N-1)), the per-pair
    event frequency scales with the managed separation cross-section, RF ~ d_TC.  Hence
    N_ref-1 ~ 1/d_TC, and the RELATIVE capacity vs the BUBBLES default-comm (1.8 s)
    separation is  rel_cap = d_TC_default / d_TC .  We also report an illustrative
    absolute N_ref anchored at the D2.1 example value 4.79 UAS for the default comm.

Run (server 160):
    ~/.conda/envs/uav_semcom/bin/python scripts/build_separation_capacity.py --selftest
    ~/.conda/envs/uav_semcom/bin/python scripts/build_separation_capacity.py \
        --latency outputs/reports/latency_breakdown_3ch.csv \
        --out outputs/reports/separation_capacity.csv \
        --fig outputs/figures/comparison/F9_separation_capacity
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
# E7v2 (task #23): the ONE shared M/G/1 priority queueing estimator.
from vqa_semcom.sim.bubbles_separation import mg1_priority_wait  # noqa: E402

FT2M = 0.3048

# ---- E7v2 shared-link queueing (task #23) -----------------------------------------
# The tactical loop delivers evidence over a SHARED radio link.  BUBBLES' 1.80 s
# separation-communication mean is treated as the queue-EMPTY tactical baseline
# (formulate + deliver instruction), and the SHARED-LINK contention is added as an
# M/G/1 non-preemptive priority wait W on top:  T4_comm = 1.80 + W.
#   * service time S = evidence airtime (upload_s); E[S^2] = S^2 (deterministic per
#     (service, snr) cell) unless a variance is supplied.
#   * C2 / critical evidence is the HIGH-priority class (index 0); payload/routine
#     evidence is low-priority (index 1).
#   * rho (per-class utilisation) is parameterised by the offered load; peak and
#     nominal are swept for the sensitivity table.
# NOTE (C2 spectrum debate, ED-282-class anchors): whether C2 shares the payload
# link is contested.  --c2-dedicated models a DEDICATED C2 band -> W is built from
# the SAME-priority payload stream only (C2 not in the shared pool); the default
# --c2-shared puts C2 in the shared pool as the high-priority class.  Both are
# config-gated; the writing layer aligns with the parallel spectrum audit.
T4_COMM_BASELINE_S = 1.80
# Offered-load presets: fraction of link time the payload/C2 streams request.
LOAD_PRESETS = {"nominal": {"c2_rho": 0.10, "payload_rho": 0.35},
                "peak": {"c2_rho": 0.25, "payload_rho": 0.55}}

# ---- Block-4 primitive constants (PDF pp.60-61) -----------------------------------
D_NMAC = 25.0 * FT2M          # 7.62 m
H_NMAC = 7.5 * FT2M           # 2.286 m
VV = 1.0                      # vertical reference speed (example)

# ---- Table G-3 errors: per-sigma horizontal/vertical (TSE given as sigma; SSE as 2sig)
TSE_H_S = 2.12
TSE_V_S = 1.82
SSE_H_S = 4.0 / 2.0           # 2.0  (table lists SSE_H at 2 sigma = 4)
SSE_V_S = 2.0 / 2.0           # 1.0  (table lists SSE_V at 2 sigma = 2)

# ---- Table G-2 tactical delay T4 components (mean, sigma) --------------------------
T4_TRACK = (2.40, 2.20)
T4_SEP = (1.80, 1.00)         # <-- separation-communication (mean replaced by us)
T4_PILOT = (3.62, 3.48)
T4_SIGMA = math.sqrt(T4_TRACK[1] ** 2 + T4_SEP[1] ** 2 + T4_PILOT[1] ** 2)  # 4.237

# ---- constants recovered from the worked example (Appendix G) ----------------------
T2 = 3.004                    # from Table G-5 vertical IC increment (Vv=1, 2 sigma)
T_RES = 4.23                  # from Table G-5 vertical SL increment (~ sigma(T4))
ROC_OFF = 0.5                 # effective climb rate = ROC - 0.5 (reproduces Table G-4)

# ---- Table B-2 physical traffic classes (Vc cruise, ROC climb; p.99) --------------
#   name -> (Vc m/s, ROC m/s, mac_radius m = size_h/2)
TRAFFIC = {
    "A1": (5.0, 4.0, 0.25),
    "A2": (5.0, 4.0, 0.50),
    "A3": (10.0, 4.0, 1.00),
    "SAIL I-II": (12.0, 4.0, 0.50),
    "SAIL III-IV": (14.0, 5.0, 1.00),
    "SAIL V-VI": (15.0, 5.0, 1.00),
    "No pass.": (25.0, 5.0, 2.00),
    "Passenger": (25.0, 3.0, 2.50),
}

# ---- Table G-4/G-5 published targets for the self-test -----------------------------
G4_HORIZ = {  # class -> (NMAC, CA/IC, SP, TC)
    "A1": (7.62, 35.33, 60.51, 163.64),
    "A2": (7.62, 35.33, 60.51, 163.64),
    "A3": (7.62, 58.79, 105.16, 311.42),
    "SAIL I-II": (7.62, 68.18, 123.02, 370.53),
    "SAIL III-IV": (7.62, 72.30, 135.61, 410.90),
    "SAIL V-VI": (7.62, 76.62, 144.17, 439.11),
    "No pass.": (7.62, 119.79, 229.70, 721.28),
    "Passenger": (7.62, 146.12, 256.04, 815.01),
}
G5_VERT = (2.29, 8.93, 15.16, 31.46)   # NMAC, CA/IC, SP, TC (identical for all classes)

# capacity anchor from D2.1 Appendix G example (default comm = 1.8 s): N_ref = 4.79 UAS
N_REF_ANCHOR = 4.79


def separation_minima(vc, roc, comm_mean, z):
    """Return the full Block-4 chain for one traffic class / comm mean / confidence z.

    comm_mean : mean of the separation-communication delay (s); 1.8 = BUBBLES default.
    z         : confidence level in sigma units (0, 1 or 2).
    """
    tse_h, tse_v = z * TSE_H_S, z * TSE_V_S
    sse_h, sse_v = z * SSE_H_S, z * SSE_V_S

    t4_mean = T4_TRACK[0] + comm_mean + T4_PILOT[0]     # replace separator MEAN only
    t4 = t4_mean + z * T4_SIGMA                          # sigmas kept at default

    rate = roc - ROC_OFF
    h_avoid = H_NMAC + tse_v
    t1 = h_avoid / rate

    # vertical chain (Vv) -> supplies h_SL used by T3
    h_ic = H_NMAC + tse_v + VV * T2
    h_sl = h_ic + sse_v + VV * T_RES
    h_tc = h_sl + VV * t4
    t3 = h_sl / rate

    # horizontal chain
    d_ic = D_NMAC + tse_h + vc * (t1 + T2)
    d_sl = d_ic + sse_h + vc * T_RES
    d_tc = d_sl + vc * (t3 + t4)
    return {
        "d_NMAC": D_NMAC, "d_IC": d_ic, "d_SL": d_sl, "d_TC": d_tc,
        "h_NMAC": H_NMAC, "h_IC": h_ic, "h_SL": h_sl, "h_TC": h_tc,
        "T4_mean": t4_mean, "T4": t4,
    }


def evidence_queue_wait(airtime_s, load, c2_dedicated=False, payload_airtime_s=None):
    """E7v2 shared-link M/G/1 priority wait W for delivering one evidence packet.

    airtime_s        : service time S of THIS evidence delivery (upload_s), s.
    load             : LOAD_PRESETS entry {c2_rho, payload_rho}.
    c2_dedicated     : if True, C2 runs on a dedicated band -> the shared pool is
                       the payload stream only, and the evidence (high-priority
                       tactical class) waits behind same-priority payload traffic;
                       if False (default), C2 is the high-priority class sharing
                       the link and the evidence rides the C2 class.
    payload_airtime_s: mean service of the low-priority payload stream (defaults to
                       airtime_s).  Used to size lambda from rho (lambda=rho/E[S]).

    Returns W (s).  The evidence is always the HIGH-priority class (tactical loop);
    C2 either shares that class (c2_dedicated=False) or is removed to a dedicated
    band (c2_dedicated=True).
    """
    es_hi = float(airtime_s)
    es_lo = float(payload_airtime_s if payload_airtime_s is not None else airtime_s)
    if es_hi <= 0.0:
        return 0.0
    c2_rho = float(load["c2_rho"])
    payload_rho = float(load["payload_rho"])
    # lambda_i = rho_i / E[S_i]; E[S^2] deterministic = E[S]^2.  Evidence is always
    # the HIGH-priority class (index 0); its wait depends only on same-or-higher
    # priority load, so the C2-mode changes exactly what shares the top class.
    if c2_dedicated:
        # DEDICATED C2 band (ED-282-class reading): C2 is NOT in the shared pool.
        # The tactical evidence still traverses the shared link as high-priority,
        # but its top-class load is just the evidence stream itself (small), so it
        # only queues behind the low-priority payload's residual.
        hi_rho = min(0.5 * c2_rho, 0.2)   # evidence-only high-priority utilisation
        classes = [
            {"lam": hi_rho / es_hi, "es": es_hi, "es2": es_hi * es_hi},
            {"lam": payload_rho / es_lo, "es": es_lo, "es2": es_lo * es_lo},
        ]
    else:
        # SHARED C2 (default): C2 is the high-priority class on the SAME link and
        # the evidence rides it, so the top-class load is the full c2_rho.
        classes = [
            {"lam": c2_rho / es_hi, "es": es_hi, "es2": es_hi * es_hi},
            {"lam": payload_rho / es_lo, "es": es_lo, "es2": es_lo * es_lo},
        ]
    return mg1_priority_wait(classes, target_index=0)


# ------------------------------------------------------------------------------------
def run_selftest():
    print("=" * 78)
    print("SELF-TEST: reproduce BUBBLES D2.1 Table G-4 (horizontal) & G-5 (vertical)")
    print("           default separation-comm mean = 1.80 s, confidence z = 2 sigma")
    print("=" * 78)
    worst = 0.0
    print("\nHORIZONTAL (m)   [class: NMAC / CA-IC / SP / TC]")
    print("  %-13s %8s %8s %8s   %8s %8s %8s   max%%" %
          ("class", "IC_ref", "IC_calc", "err%", "TC_ref", "TC_calc", "err%"))
    for name, (vc, roc, _mac) in TRAFFIC.items():
        r = separation_minima(vc, roc, 1.80, 2)
        ref = G4_HORIZ[name]
        calc = (r["d_NMAC"], r["d_IC"], r["d_SL"], r["d_TC"])
        errs = [abs(c - t) / t * 100 for c, t in zip(calc, ref)]
        worst = max(worst, max(errs))
        print("  %-13s %8.2f %8.2f %7.3f   %8.2f %8.2f %7.3f   %.3f" %
              (name, ref[1], calc[1], errs[1], ref[3], calc[3], errs[3], max(errs)))
    # vertical (identical for all classes; use SAIL I-II)
    r = separation_minima(12.0, 4.0, 1.80, 2)
    vcalc = (r["h_NMAC"], r["h_IC"], r["h_SL"], r["h_TC"])
    verrs = [abs(c - t) / t * 100 for c, t in zip(vcalc, G5_VERT)]
    worst = max(worst, max(verrs))
    print("\nVERTICAL (m)     NMAC / CA-IC / SP / TC")
    print("  ref :  %6.2f %6.2f %6.2f %6.2f" % G5_VERT)
    print("  calc:  %6.2f %6.2f %6.2f %6.2f" % vcalc)
    print("  err%%:  %6.3f %6.3f %6.3f %6.3f" % tuple(verrs))
    print("\nworst-cell error = %.3f %%   ->   %s (gate: < 1%%)" %
          (worst, "PASS" if worst < 1.0 else "FAIL"))
    return worst < 1.0


# ------------------------------------------------------------------------------------
def load_latency(path):
    """rows -> list of dict(channel, method, snr_db, upload, txside, comm).

    E7v2: `upload` (link airtime) is the M/G/1 SERVICE TIME S; `txside` (detector
    latency) is transmitter compute that is NOT link contention.  `comm` (legacy
    upload+txside) is retained only for the v5-compat baseline column.
    """
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            up = float(r["upload_s"])
            tx = float(r["txside_s"])
            out.append({
                "channel": r["channel"], "method": r["method"],
                "snr_db": float(r["snr_db"]),
                "upload": up, "txside": tx, "comm": up + tx,
            })
    return out


def build_table(latency_rows, traffic_class, bands=(0, 1, 2),
                load_name="peak", c2_dedicated=False):
    """E7v2: T4_comm = 1.80 s baseline + shared-link M/G/1 wait W(upload, load).

    Replaces the v5 defect (payload latency substituted directly as the T4
    separation-comm mean).  The evidence airtime `upload` sizes the M/G/1 service
    time; `load_name` selects the offered-load preset; `c2_dedicated` toggles the
    C2 spectrum reading.
    """
    vc, roc, _mac = TRAFFIC[traffic_class]
    load = LOAD_PRESETS[load_name]
    # default-comm baseline d_TC per band (queue-empty 1.80 s) -> capacity reference
    d_tc_def = {z: separation_minima(vc, roc, T4_COMM_BASELINE_S, z)["d_TC"] for z in bands}
    out = []
    for row in latency_rows:
        w = evidence_queue_wait(row["upload"], load, c2_dedicated=c2_dedicated)
        t4_comm = T4_COMM_BASELINE_S + w   # baseline tactical mean + shared-link wait
        for z in bands:
            r = separation_minima(vc, roc, t4_comm, z)
            rel_cap = d_tc_def[z] / r["d_TC"]
            out.append({
                "channel": row["channel"], "service": row["method"],
                "snr_db": row["snr_db"], "traffic_class": traffic_class,
                "sigma_band": z,
                "upload_airtime_s": round(row["upload"], 4),
                "queue_wait_W_s": round(w, 4),
                "T4_comm_mean_s": round(t4_comm, 4),
                "load": load_name, "c2_mode": "dedicated" if c2_dedicated else "shared",
                "comm_latency_s": round(row["comm"], 4),
                "T4_mean_s": round(r["T4_mean"], 3), "T4_s": round(r["T4"], 3),
                "d_TC_m": round(r["d_TC"], 2),
                "d_TC_default_m": round(d_tc_def[z], 2),
                "rel_capacity": round(rel_cap, 4),
                "N_ref_illustrative": round(1.0 + (N_REF_ANCHOR - 1.0) * rel_cap, 3),
            })
    return out


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["channel", "service", "snr_db", "traffic_class", "sigma_band",
            "upload_airtime_s", "queue_wait_W_s", "T4_comm_mean_s", "load", "c2_mode",
            "comm_latency_s", "T4_mean_s", "T4_s", "d_TC_m", "d_TC_default_m",
            "rel_capacity", "N_ref_illustrative"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote %d rows -> %s" % (len(rows), path))


def w_sensitivity_table(latency_rows):
    """E7v2: W sensitivity across load presets x C2 mode, for a representative
    token/image airtime.  Printed for the doc's W-sensitivity table."""
    # pick a representative upload airtime per method (median over rows).
    by_method = {}
    for r in latency_rows:
        by_method.setdefault(r["method"], []).append(r["upload"])
    reps = {m: sorted(v)[len(v) // 2] for m, v in by_method.items()}
    print("\n" + "=" * 70)
    print("E7v2 W-SENSITIVITY  (M/G/1 non-preemptive priority wait, s)")
    print("  %-14s %10s %10s %12s %12s" % ("method(upload_s)", "load", "c2_shared_W", "c2_dedic_W", "T4comm_shared"))
    rows_out = []
    for m in sorted(reps):
        up = reps[m]
        for load_name in ("nominal", "peak"):
            ws = evidence_queue_wait(up, LOAD_PRESETS[load_name], c2_dedicated=False)
            wd = evidence_queue_wait(up, LOAD_PRESETS[load_name], c2_dedicated=True)
            print("  %-14s %10s %10.4f %12.4f %12.4f" %
                  ("%s(%.3f)" % (m, up), load_name, ws, wd, T4_COMM_BASELINE_S + ws))
            rows_out.append({"method": m, "upload_s": up, "load": load_name,
                             "W_c2_shared_s": ws, "W_c2_dedicated_s": wd})
    return rows_out


def make_figure(rows, traffic_class, fig_base, channel="rician"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["M1_image", "M3_token", "M4_adaptive"]
    labels = {"M1_image": "M1 raw image", "M3_token": "M3 semantic tokens",
              "M4_adaptive": "M4 adaptive policy"}
    colors = {"M1_image": "#ffb454", "M3_token": "#5ad19a", "M4_adaptive": "#4b8bff"}

    def series(method, key, z):
        pts = [(r["snr_db"], r[key]) for r in rows
               if r["channel"] == channel and r["service"] == method and r["sigma_band"] == z]
        pts.sort()
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for m in methods:
        x, y2 = series(m, "d_TC_m", 2)
        _, y0 = series(m, "d_TC_m", 0)
        axL.fill_between(x, y0, y2, color=colors[m], alpha=0.18)
        axL.plot(x, y2, "-o", color=colors[m], lw=2, ms=4, label=labels[m])
    d_def = [r["d_TC_default_m"] for r in rows if r["sigma_band"] == 2]
    axL.axhline(d_def[0], ls="--", color="0.4", lw=1,
                label="BUBBLES default comm (1.8 s)")
    axL.set_xlabel("SNR (dB)")
    axL.set_ylabel(r"tactical-conflict separation $d_{TC}$ (m)")
    axL.text(0.02, 0.97, "(a)", transform=axL.transAxes, va="top", ha="left",
             fontsize=10, fontweight="bold")
    axL.text(0.98, 0.03, "%s, class %s" % (
        {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}.get(channel, channel),
        traffic_class), transform=axL.transAxes, va="bottom", ha="right", fontsize=8, color="0.3")
    axL.grid(True, alpha=0.3)
    axL.legend(fontsize=8, loc="upper right")

    for m in methods:
        x, c2 = series(m, "rel_capacity", 2)
        _, c0 = series(m, "rel_capacity", 0)
        axR.fill_between(x, c0, c2, color=colors[m], alpha=0.18)
        axR.plot(x, c2, "-o", color=colors[m], lw=2, ms=4, label=labels[m])
    axR.axhline(1.0, ls="--", color="0.4", lw=1, label="BUBBLES default = 1.0")
    axR.set_xlabel("SNR (dB)")
    axR.set_ylabel(r"relative airspace capacity  $d_{TC}^{\rm def}/d_{TC}$")
    axR.text(0.02, 0.97, "(b)", transform=axR.transAxes, va="top", ha="left",
             fontsize=10, fontweight="bold")
    axR.grid(True, alpha=0.3)
    axR.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_base), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (fig_base, ext), dpi=150)
    print("wrote %s.{png,pdf}" % fig_base)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="reproduce Table G-4/G-5 and check <1%% before anything else")
    ap.add_argument("--latency", default="outputs/reports/latency_breakdown_3ch.csv")
    ap.add_argument("--out", default="outputs/reports/separation_capacity.csv")
    ap.add_argument("--fig", default="outputs/figures/comparison/F9_separation_capacity")
    ap.add_argument("--traffic-class", default="SAIL I-II", choices=list(TRAFFIC))
    ap.add_argument("--fig-channel", default="rician")
    ap.add_argument("--load", default="peak", choices=list(LOAD_PRESETS),
                    help="E7v2 offered-load preset for the shared-link M/G/1 wait.")
    ap.add_argument("--c2-dedicated", action="store_true",
                    help="E7v2: model a DEDICATED C2 band (W from same-priority payload only). "
                         "Default: C2 shares the link as the high-priority class.")
    args = ap.parse_args()

    ok = run_selftest()
    if args.selftest:
        return 0 if ok else 1
    if not ok:
        print("ABORT: self-test failed, refusing to emit calibrated outputs.")
        return 1

    latency = load_latency(args.latency)
    rows = build_table(latency, args.traffic_class, load_name=args.load,
                       c2_dedicated=args.c2_dedicated)
    write_csv(rows, args.out)
    w_sensitivity_table(latency)
    make_figure(rows, args.traffic_class, args.fig, channel=args.fig_channel)

    # key numbers for the report
    def get(ch, m, snr, z, key):
        for r in rows:
            if (r["channel"] == ch and r["service"] == m and r["snr_db"] == snr
                    and r["sigma_band"] == z):
                return r[key]
        return None
    print("\n" + "=" * 70)
    print("KEY NUMBERS (channel=rician, class=%s, 2 sigma)" % args.traffic_class)
    for snr in (-5.0, 20.0):
        di = get("rician", "M1_image", snr, 2, "d_TC_m")
        dt = get("rician", "M3_token", snr, 2, "d_TC_m")
        ci = get("rician", "M1_image", snr, 2, "rel_capacity")
        ct = get("rician", "M3_token", snr, 2, "rel_capacity")
        print("  SNR %+5.0f dB:  d_TC image=%.1f m  token=%.1f m  (delta=%.1f m); "
              "rel-cap image=%.3f token=%.3f  (token/image=%.2fx)" %
              (snr, di, dt, di - dt, ci, ct, ct / ci))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
