#!/usr/bin/env python3
"""naturefig F5+F11: measured per-answer energy panels, TGCN Fig. 2 (a)+(b).

Publication-figure redraw of the two energy panels of make_energy_figures.py.
Plot-only: reads outputs/reports/comparison_v3_5qt.csv and
outputs/energy/gpu_power_phases.json with the identical energy model
(same constants, same route mix, same M6 grid) and writes PDF+SVG+PNG to
outputs/figures/naturefig/. It does NOT rewrite energy_summary.json or the
LaTeX table snippets -- the numbers of record stay untouched.

Content baseline = make_energy_figures.py at 84d1e11 (paper commit 6eb551a):
  (a) keeps the C1 budget-tunable per-sample routing frontier (lambda sweep,
      outputs/energy/c1_frontier_<channel>.csv, drawn point-for-point), the
      "energy price lambda sweep" tag, the fixed-rate digital-cliff note and
      the grayscale-safe per-method linestyles;
  (b) keeps the three direct plateau labels (1.46 / 0.043--0.047 /
      ~0.018 all full-image pipelines) and the same linestyles.

Design deltas vs those assets (presentation only, data unchanged):
  (a) F5_pareto_energy: SNR endpoint labels get short leaders so they cannot
      sit on the markers; the measured VLM compute floor (incremental
      J/answer) is a light vertical guide that explains why four image
      pipelines pile up in one column; hero-curve emphasis; panel letter.
      (The earlier grey empirical-Pareto staircase is dropped: the lambda
      frontier is the frontier object of record and the two would overlap.)
  (b) F11_answers_per_joule: y-headroom so the fixed-token line is not
      pinned to the frame; panel letter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
})
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

# ---- identical energy model to make_energy_figures.py ----------------------
BANDWIDTH_HZ = 1.0e6
P_TX_HEAD = 0.5
E_DET_LO_J = 7.0 * 0.015
E_DET_HI_J = 15.0 * 0.050
E_DET_MID_J = 0.5 * (E_DET_LO_J + E_DET_HI_J)
F_IMG_M4 = 410.0 / 936.0
M6_USES = 9.2215e4
M6_ACC = {-5.0: 0.564, 0.0: 0.578, 5.0: 0.592, 10.0: 0.599, 15.0: 0.600,
          20.0: 0.596}

STYLE = {  # color, marker, label, linestyle (distinct ls = grayscale-safe)
    "M0_naive":    ("#ff6b6b", "x", "Fixed-rate image", ":"),
    "M1_image":    ("#ffb454", "o", "Rate-adaptive image", "-"),
    "M2_analog":   ("#c678dd", "v", "Uncoded analog", "--"),
    "M6_djscc":    ("#8b5e3c", "P", "DJSCC (learned)", "-."),
    "M3_token":    ("#9aa7b4", "s", "Fixed token", "-"),
    "M4_adaptive": ("#5ad19a", "D", "Evidence routing (ours)", "-"),
}
ORDER = ["M4_adaptive", "M3_token", "M6_djscc", "M2_analog", "M1_image",
         "M0_naive"]
F_IMG = {"M0_naive": 1.0, "M1_image": 1.0, "M2_analog": 1.0, "M6_djscc": 1.0,
         "M3_token": 0.0, "M4_adaptive": F_IMG_M4}


def load_rows(csv_path: Path, channel: str):
    out: dict[str, dict[float, dict]] = {}
    for r in csv.DictReader(open(csv_path)):
        if r["channel"] != channel or r["qtype"] != "all" or r["split"] != "test":
            continue
        out.setdefault(r["method"], {})[float(r["snr_db"])] = {
            "acc": float(r["accuracy"]),
            "uses": float(r["mean_channel_uses"] or 0.0),
            "lcb": float(r["lcb"]), "ucb": float(r["ucb"]),
        }
    out["M6_djscc"] = {s: {"acc": a, "uses": M6_USES, "lcb": None, "ucb": None}
                       for s, a in M6_ACC.items()}
    return out


def energy_j(method, uses, p_tx, e_vlm, e_det=E_DET_MID_J):
    e_tx = uses / BANDWIDTH_HZ * p_tx
    f = F_IMG[method]
    return e_tx + f * e_vlm + (1.0 - f) * e_det


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/reports/comparison_v3_5qt.csv")
    ap.add_argument("--power-json", default="outputs/energy/gpu_power_phases.json")
    ap.add_argument("--out-dir", default="outputs/figures/naturefig")
    ap.add_argument("--channel", default="rician")
    args = ap.parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(REPO / args.csv, args.channel)
    power = json.loads((REPO / args.power_json).read_text())
    e_vlm_inc = power["phases"]["vlm"]["joule_per_item_incremental"]
    snrs = sorted(rows["M4_adaptive"].keys())

    E = {m: {s: energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc)
             for s in snrs if s in rows[m]} for m in ORDER}

    # ---- (a) F5: accuracy vs J/answer, log x -------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    # measured VLM compute floor (incremental) -- explains the image column
    ax.axvline(e_vlm_inc, color="0.6", lw=0.6, ls=":", zorder=0)
    ax.text(e_vlm_inc * 0.88, 0.435, "VLM compute floor",
            rotation=90, fontsize=5.5, color="0.4", ha="right", va="bottom")
    for m in ORDER:
        c, mk, lb, ls = STYLE[m]
        xs = [E[m][s] for s in snrs if s in E[m]]
        ys = [rows[m][s]["acc"] for s in snrs if s in E[m]]
        hero = m == "M4_adaptive"
        ax.plot(xs, ys, marker=mk, color=c, label=lb, ls=ls,
                lw=1.3 if hero else 0.9,
                ms=3.6 if hero else 3.0, zorder=6 if hero else 3)
        if hero:  # Wilson bars (as before)
            for s in snrs:
                ax.vlines(E[m][s], rows[m][s]["lcb"], rows[m][s]["ucb"],
                          color=c, alpha=0.55, lw=0.9, zorder=4)
        if m == "M3_token":  # Jetson detector-energy band
            for s in snrs:
                lo = energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc,
                              E_DET_LO_J)
                hi = energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc,
                              E_DET_HI_J)
                ax.hlines(rows[m][s]["acc"], lo, hi, color=c, alpha=0.4,
                          lw=2.2, zorder=2)
    # C1: energy-controllable per-sample frontier (lambda sweep over the
    # per-sample correctness predictors; scripts/p1_review_fix_analysis.py).
    # Content of record -- drawn point-for-point from the frontier CSV.
    frontier_csv = REPO / "outputs/energy" / f"c1_frontier_{args.channel}.csv"
    if frontier_csv.exists():
        fr = sorted((float(r["E_j"]), float(r["acc"]))
                    for r in csv.DictReader(open(frontier_csv)))
        ax.plot([p[0] for p in fr], [p[1] for p in fr], color="#2f2f2f",
                ls="--", lw=0.9, marker=".", ms=2.2, zorder=7,
                label="Budget-tunable per-sample routing")
        ax.annotate("energy price $\\lambda$ sweep", (0.55, 0.702),
                    fontsize=5.5, color="#2f2f2f", ha="left")
    # fixed-rate -5 dB outlier: name the mechanism inside the figure
    ax.annotate("fixed-rate @$-5$ dB:\ndigital cliff (FER${\\approx}$1)",
                (E["M0_naive"][snrs[0]], rows["M0_naive"][snrs[0]]["acc"]),
                textcoords="offset points", xytext=(-8, 2), fontsize=5.0,
                color=STYLE["M0_naive"][0], ha="right")
    # SNR endpoint labels with short leaders (off the markers, off the spine)
    s_lo, s_hi = snrs[0], snrs[-1]
    for m, s, fx, dy, ha, va in (
            ("M4_adaptive", s_hi, 0.80, +0.026, "right", "bottom"),
            ("M4_adaptive", s_lo, 0.78, -0.030, "right", "top"),
            ("M1_image",    s_hi, 0.88, +0.030, "right", "bottom"),
            ("M1_image",    s_lo, 0.68, -0.016, "right", "center")):
        c = STYLE[m][0]
        x, y = E[m][s], rows[m][s]["acc"]
        ax.annotate(f"{s:+.0f} dB", xy=(x, y), xytext=(x * fx, y + dy),
                    fontsize=5.5, color=c, ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=c,
                                    alpha=0.65, shrinkA=0.4, shrinkB=1.6))
    ax.set_xscale("log")
    ax.set_xlabel("joint energy per answer (J)", fontsize=8)
    ax.set_ylabel("VQA accuracy (test)", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.03, 0.97, "Rician K=6 dB", transform=ax.transAxes, fontsize=6.5,
            va="top")
    ax.text(-0.16, 1.02, "(a)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    ax.legend(fontsize=6.2, loc="lower left", handlelength=1.4,
              labelspacing=0.25, borderpad=0.3)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(out_dir / f"F5_pareto_energy.{ext}")
    fig.savefig(out_dir / "F5_pareto_energy.png", dpi=600)
    plt.close(fig)

    # ---- (b) F11: answers per joule vs SNR ---------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for m in ORDER:
        c, mk, lb, ls = STYLE[m]
        xs = [s for s in snrs if s in E[m]]
        ys = [rows[m][s]["acc"] / E[m][s] for s in xs]
        ax.plot(xs, ys, marker=mk, color=c, label=lb, ls=ls,
                lw=1.3 if m == "M4_adaptive" else 0.9, ms=3.2,
                zorder=5 if m == "M4_adaptive" else 3)
    # direct plateau labels (values readable without chasing the legend);
    # content of record: 1.46 / 0.043--0.047 / ~0.018.
    apj = lambda m, s: rows[m][s]["acc"] / E[m][s]
    ax.annotate(f"{apj('M3_token', 20.0):.2f}", (-4.7, apj("M3_token", -5.0)),
                textcoords="offset points", xytext=(0, -9), fontsize=5.5,
                color=STYLE["M3_token"][0])
    ax.annotate("0.043--0.047".replace("--", "–"),
                (-4.7, apj("M4_adaptive", -5.0)),
                textcoords="offset points", xytext=(0, 5), fontsize=5.5,
                color="#2e9e6e")
    ax.annotate("$\\approx$0.018 (all full-image pipelines)",
                (2.0, apj("M1_image", 10.0)),
                textcoords="offset points", xytext=(0, -11), fontsize=5.5,
                color=STYLE["M1_image"][0])
    ax.set_yscale("log")
    ax.set_ylim(top=4.0)  # headroom: token line off the frame
    ax.set_xlabel("SNR (dB)", fontsize=8)
    ax.set_ylabel("correct answers per joule", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.02, 0.97, f"Rician K=6 dB, $P_{{\\mathrm{{tx}}}}$={P_TX_HEAD} W",
            transform=ax.transAxes, fontsize=6.5, va="top")
    ax.text(-0.16, 1.02, "(b)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")
    ax.legend(fontsize=6.2, loc="center right", handlelength=1.4,
              labelspacing=0.25, borderpad=0.3)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(out_dir / f"F11_answers_per_joule.{ext}")
    fig.savefig(out_dir / "F11_answers_per_joule.png", dpi=600)
    plt.close(fig)
    print(f"wrote F5_pareto_energy + F11_answers_per_joule [pdf,svg,png] -> {out_dir}")


if __name__ == "__main__":
    main()
