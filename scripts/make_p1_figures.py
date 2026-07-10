#!/usr/bin/env python3
"""P1 paper figures (IEEE single-column style, no in-figure titles):

F9_separation.pdf  -- BUBBLES Block-4 safety bridge, M/G/1-queue caliber
                      (separation_v2 data): d_TC and relative capacity vs SNR,
                      Rayleigh, peak load, shared C2 link, 2-sigma T4 window.
F10_fer.pdf        -- residual frame-loss probability vs SNR: token
                      information-outage (measured mean payload) vs measured
                      fixed-rate LDPC FER, three channels.

Inputs: outputs/reports/separation_v2/sepcap_{peak,nominal}_shared.csv
        outputs/reports/p1_fer_payload.json
Outputs: outputs/figures/paper1_final/{F9_separation.pdf,F10_fer.pdf} (+png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/figures/paper1_final"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.3, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "figure.dpi": 150,
})

# Descriptive method names (paper-facing; internal codes stay in the CSVs).
SERVICE_STYLE = {
    "M0_naive": dict(color="#7f7f7f", ls=":", marker="v", label="Fixed-rate image"),
    "M1_image": dict(color="#d62728", ls="-", marker="o", label="Rate-adaptive image"),
    "M4_adaptive": dict(color="#1f77b4", ls="-", marker="s", label="Evidence routing (ours)"),
    "M2_analog": dict(color="#ff7f0e", ls="--", marker="^", label="Uncoded analog"),
    "M3_token": dict(color="#2ca02c", ls="-", marker="D", label="Fixed token"),
}
ORDER = ["M0_naive", "M1_image", "M4_adaptive", "M2_analog", "M3_token"]


def make_f9() -> None:
    df = pd.read_csv(REPO / "outputs/reports/separation_v2/sepcap_peak_shared.csv")
    sub = df[(df.channel == "rayleigh") & (df.sigma_band == 2)]
    default = float(sub.d_TC_default_m.iloc[0])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.35), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    for svc in ORDER:
        g = sub[sub.service == svc].sort_values("snr_db")
        st = SERVICE_STYLE[svc]
        ax1.plot(g.snr_db, g.d_TC_m, color=st["color"], ls=st["ls"], marker=st["marker"],
                 ms=3, label=st["label"])
        ax2.plot(g.snr_db, 100.0 * (1.0 - g.rel_capacity), color=st["color"], ls=st["ls"],
                 marker=st["marker"], ms=3, label=st["label"])
    ax1.axhline(default, color="k", ls="--", lw=0.8)
    ax1.annotate(f"BUBBLES baseline ({default:.1f} m)", xy=(7.5, 367.1), fontsize=6.8)
    ax1.set_ylim(366.5, 414)
    ax1.set_ylabel(r"separation minimum $d_{\mathrm{TC}}$ (m)")
    ax1.grid(alpha=0.25, lw=0.4)
    ax1.text(0.02, 0.975, r"Rayleigh, peak load, shared C2, $2\sigma$",
             transform=ax1.transAxes, va="top", fontsize=6.8)
    ax2.axhline(0.0, color="k", ls="--", lw=0.8)
    ax2.set_ylabel("airspace-capacity loss (%)")
    ax2.set_xlabel("SNR (dB)")
    ax2.grid(alpha=0.25, lw=0.4)
    ax2.legend(frameon=False, ncol=1, loc="upper right", handlelength=2.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "F9_separation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "F9_separation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("F9 done:", OUT / "F9_separation.pdf")


def make_f10() -> None:
    data = json.loads((REPO / "outputs/reports/p1_fer_payload.json").read_text())
    curves = data["curves"]
    ldpc = data["report"]["ldpc_fixed_rate_fer_measured"]
    grid = np.asarray(curves["snr_grid"])
    colors = {"awgn": "#1f77b4", "rayleigh": "#d62728", "rician": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    floor = 2e-5
    for name, c in colors.items():
        y = np.asarray(curves[f"tokenMeas_{name}"])
        if np.all(y <= floor):  # AWGN token outage is exactly zero
            ax.semilogy(grid, np.full_like(grid, floor), color=c, ls="-",
                        label=f"token outage ({name}) $=0$")
            continue
        ax.semilogy(grid, np.maximum(y, floor), color=c, ls="-",
                    label=f"token outage ({name})")
    for name, c in colors.items():
        if name not in ldpc:
            continue
        xs = sorted(float(k.replace("dB", "")) for k in ldpc[name])
        ys = np.maximum([ldpc[name][f"{int(x)}dB"] for x in xs], floor)
        ax.semilogy(xs, ys, color=c, ls="--", marker="o", ms=2.8,
                    label=f"fixed-rate LDPC FER ({name})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("residual frame-loss probability")
    ax.set_ylim(floor, 2.0)
    ax.set_xlim(-6, 21)
    ax.grid(True, which="both", alpha=0.22, lw=0.4)
    ax.legend(frameon=False, ncol=1, loc="lower left", handlelength=1.8)
    fig.savefig(OUT / "F10_fer.pdf", bbox_inches="tight")
    fig.savefig(OUT / "F10_fer.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("F10 done:", OUT / "F10_fer.pdf")


if __name__ == "__main__":
    make_f9()
    make_f10()
