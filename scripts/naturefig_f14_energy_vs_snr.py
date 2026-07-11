#!/usr/bin/env python3
"""naturefig F14: joint energy per answered question vs SNR (TGCN).

Table-to-figure conversion of the manuscript's Table `tab:jpa`: the
J/answer column becomes six method lines over the full six-point SNR grid
(log energy axis), and the bottom P_tx-sensitivity block becomes shaded
bands (P_tx swept 0.1--1 W) around the three methods the table swept
(fixed token, evidence routing, rate-adaptive image).

Data of record only -- no recomputation:
  outputs/energy/energy_summary.json: per-method j_per_answer (headline
      P_tx = 0.5 W, incremental compute) and mean channel uses per SNR;
      the P_tx band re-prices ONLY the transmit term with the generator's
      own linear model  E(p) = uses/B * p + e_cmp  (bandwidth B = 1 MHz),
      which reproduces the table's sensitivity block bit-for-bit
      (e.g. token 0.429/0.442 J, routing 14.6/16.2 J at -5 dB).

Style: nature-figure conventions -- Times-family serif (Nimbus Roman on
the render host), grayscale-safe linestyles (same per-method styles as
every quantitative figure), direct labels, Type-42 fonts.
Outputs PDF + SVG + PNG (300 dpi).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
    "lines.linewidth": 1.0, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/figures/naturefig"

STYLE = {  # color, marker, label, linestyle -- as in naturefig_f5_energy.py
    "M0_naive":    ("#ff6b6b", "x", "Fixed-rate image", ":"),
    "M1_image":    ("#ffb454", "o", "Rate-adaptive image", "-"),
    "M2_analog":   ("#c678dd", "v", "Uncoded analog", "--"),
    "M6_djscc":    ("#8b5e3c", "P", "DJSCC (learned)", "-."),
    "M3_token":    ("#9aa7b4", "s", "Fixed token", "-"),
    "M4_adaptive": ("#5ad19a", "D", "Evidence routing (ours)", "-"),
}
ORDER = ["M0_naive", "M1_image", "M2_analog", "M6_djscc", "M4_adaptive",
         "M3_token"]
BAND = ("M3_token", "M4_adaptive", "M1_image")  # tab:jpa sensitivity block
P_LO, P_HI = 0.1, 1.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    es = json.loads((REPO / "outputs/energy/energy_summary.json").read_text())
    pm, prm = es["per_method"], es["params"]
    bw = prm["bandwidth_hz"]
    assert [P_LO, P_HI] == [prm["p_tx_grid_w"][0], prm["p_tx_grid_w"][-1]]

    snrs = sorted(float(s) for s in pm["M4_adaptive"])

    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    for m in ORDER:
        c, mk, lb, ls = STYLE[m]
        hero = m == "M4_adaptive"
        xs = [s for s in snrs if str(s) in pm[m]]
        ys = [pm[m][str(s)]["j_per_answer"] for s in xs]
        if m in BAND:  # P_tx 0.1--1 W: re-price the transmit term only
            lo, hi = [], []
            for s in xs:
                v = pm[m][str(s)]
                lo.append(v["uses"] / bw * P_LO + v["e_cmp_j"])
                hi.append(v["uses"] / bw * P_HI + v["e_cmp_j"])
            ax.fill_between(xs, lo, hi, color=c, alpha=0.22, lw=0, zorder=2)
        ax.plot(xs, ys, marker=mk, color=c, label=lb, ls=ls,
                lw=1.4 if hero else 0.9, ms=3.6 if hero else 3.0,
                zorder=6 if hero else 3)
        print(m, [f"{y:.3f}" for y in ys])

    # the frontier's fixed 2.2x image/routing gap, named where it happens
    ax.annotate("", xy=(10, pm["M4_adaptive"]["10.0"]["j_per_answer"]),
                xytext=(10, pm["M1_image"]["10.0"]["j_per_answer"]),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.3",
                                shrinkA=1, shrinkB=1))
    ax.annotate("$2.2\\times$ at\nevery SNR", xy=(9.4, 20.0), fontsize=6.2,
                color="0.25", ha="right")
    ax.annotate("shaded: $P_{\\mathrm{tx}}$ swept 0.1–1 W",
                xy=(0.03, 0.585), xycoords="axes fraction", fontsize=6.0,
                color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("energy per answered question (J)")
    ax.set_xticks(snrs)
    ax.set_ylim(0.3, 60)
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.03, 0.97, "Rician K=6 dB, incremental compute",
            transform=ax.transAxes, fontsize=6.5, va="top")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.40), frameon=False,
              handlelength=1.6, labelspacing=0.25, borderpad=0.2)
    fig.tight_layout()
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        fig.savefig(OUT / f"F14_energy_vs_snr.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"wrote {OUT}/F14_energy_vs_snr.[pdf|svg|png]")


if __name__ == "__main__":
    main()
