#!/usr/bin/env python3
"""naturefig F1: 3-panel accuracy vs SNR (AWGN/Rayleigh/Rician), TGCN Fig. 1.

Publication-figure redraw of the F1 block of make_comparison_figures_v2.py.
Plot-only: reads outputs/reports/comparison_v3_5qt.csv, writes PDF+SVG+PNG to
outputs/figures/naturefig/. No data recomputation; every plotted value comes
from the tidy CSV (same source as the paper's Table II).

Design deltas vs the previous asset (data unchanged):
  - hero emphasis: the routing policy is drawn heavier and on top; baselines
    are thinner so the mid-band pile-up (0.58-0.64) stays readable;
  - Wilson 95% bands kept on routing / rate-adaptive image / fixed token
    (as before) but at lower alpha so overlaps do not muddy the lines;
  - the error-free reference is direct-labeled in the last panel (it is a
    flat reference, not a curve to trace through the legend);
  - the fixed-rate digital cliff is direct-labeled once (panel a);
  - panel letters (a)(b)(c) added; channel tags kept.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "lines.linewidth": 1.1, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt

STYLE = {
    "M0_errorfree": ("#444444", "*", "Error-free image (ideal)"),
    "M0_naive":    ("#ff6b6b", "x", "Fixed-rate image"),
    "M1_image":    ("#ffb454", "o", "Rate-adaptive image"),
    "M2_analog":   ("#c678dd", "v", "Uncoded analog"),
    "M3_token":    ("#9aa7b4", "s", "Fixed token"),
    "M4_adaptive": ("#5ad19a", "D", "Evidence routing (ours)"),
    "M5_oracle":   ("#4ea1ff", "^", "Oracle (upper bound)"),
}
ORDER = ["M0_errorfree", "M5_oracle", "M4_adaptive", "M2_analog",
         "M3_token", "M1_image", "M0_naive"]
DASHED = {"M5_oracle": "--", "M0_errorfree": ":"}
CH_TITLE = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}
BAND_METHODS = {"M4_adaptive", "M1_image", "M3_token"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/reports/comparison_v3_5qt.csv")
    ap.add_argument("--out-dir", default="outputs/figures/naturefig")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = list(csv.DictReader(open(args.csv)))
    for r in rows:
        r["snr_db"] = float(r["snr_db"]); r["accuracy"] = float(r["accuracy"])
        r["lcb"] = float(r["lcb"]); r["ucb"] = float(r["ucb"])
    channels = [c for c in ("awgn", "rayleigh", "rician")
                if any(r["channel"] == c for r in rows)]

    fig, axes = plt.subplots(1, len(channels), figsize=(7.16, 1.9), sharey=True)
    for k, (ax, ch) in enumerate(zip(axes, channels)):
        for m in ORDER:
            pts = sorted([(r["snr_db"], r["accuracy"], r["lcb"], r["ucb"])
                          for r in rows if r["channel"] == ch
                          and r["method"] == m and r["qtype"] == "all"])
            if not pts:
                continue
            col, mk, lab = STYLE[m]
            hero = m == "M4_adaptive"
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1] for p in pts], DASHED.get(m, "-"), color=col,
                    marker=mk, label=lab, markersize=3.6 if hero else 3.0,
                    lw=1.6 if hero else 0.95, zorder=6 if hero else 3)
            if m in BAND_METHODS:
                ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts],
                                color=col, alpha=0.16 if hero else 0.10,
                                linewidth=0, zorder=2)
        # direct labels: error-free reference (last panel), cliff (first panel)
        if k == len(channels) - 1:
            ef = [r for r in rows if r["channel"] == ch
                  and r["method"] == "M0_errorfree" and r["qtype"] == "all"]
            if ef:
                y0 = ef[0]["accuracy"]
                ax.annotate(f"error-free ref. {y0:.3f}", xy=(18.5, y0),
                            xytext=(19.6, 0.472), ha="right", fontsize=5.8,
                            color="#444444",
                            arrowprops=dict(arrowstyle="-", lw=0.5,
                                            color="#444444", alpha=0.6,
                                            shrinkA=0.5, shrinkB=1.0))
        if k == 0:
            nv = sorted([(r["snr_db"], r["accuracy"]) for r in rows
                         if r["channel"] == ch and r["method"] == "M0_naive"
                         and r["qtype"] == "all"])
            if len(nv) >= 2:
                ax.annotate("digital cliff", xy=nv[1],
                            xytext=(nv[1][0] + 2.2, nv[1][1] - 0.055),
                            fontsize=5.8, color="#ff6b6b",
                            arrowprops=dict(arrowstyle="-", lw=0.5,
                                            color="#ff6b6b", alpha=0.8,
                                            shrinkA=0.5, shrinkB=1.5))
        ax.text(0.04, 0.05, CH_TITLE.get(ch, ch), transform=ax.transAxes,
                va="bottom", ha="left", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.text(-0.02, 1.06, f"({chr(97 + k)})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="right")
        ax.set_xlabel("SNR (dB)")
        ax.set_yticks([0.4, 0.5, 0.6, 0.7])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("VQA answer accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7,
               frameon=False, borderaxespad=0.1, handlelength=1.8,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.83), pad=0.4, w_pad=0.8)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{args.out_dir}/F1_accuracy_snr.{ext}")
    fig.savefig(f"{args.out_dir}/F1_accuracy_snr.png", dpi=600)
    plt.close(fig)
    print(f"wrote F1_accuracy_snr.[pdf,svg,png] -> {args.out_dir}")


if __name__ == "__main__":
    main()
