#!/usr/bin/env python3
"""naturefig F8: top-t token-budget sweep, TGCN Fig. 3.

Publication-figure redraw of draw_f8() in build_token_budget_sweep.py using
the fast path only: reads the existing sweep CSV
(outputs/reports/token_budget_full.csv, no recompute) and writes PDF+SVG+PNG
to outputs/figures/naturefig/.

Design deltas vs the previous asset (data unchanged):
  - panel letters (a)(b)(c); (c) keeps its channel-use cost axis but is now
    explained by the caption update in the manuscript;
  - the three claims of the text are annotated where they happen:
    comparison saturates at t=3 (a), threshold peaks at t=32 (a,b ring),
    counting still climbing at t=48 (a);
  - legend label "co_presence" -> "co-presence" (display only; the CSV keys
    are unchanged);
  - hero curve = "all" kept dark; per-type colors unchanged.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

QT_SYMBOLIC = ("counting", "comparison", "co_presence", "threshold")
COLORS = {"counting": "#e06c75", "comparison": "#61afef",
          "co_presence": "#98c379", "threshold": "#c678dd", "all": "#444444"}
DISPLAY = {"co_presence": "co-presence"}
CH_TAG = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", default="outputs/reports/token_budget_full.csv")
    ap.add_argument("--fig-channel", default="rician")
    ap.add_argument("--out-dir", default="outputs/figures/naturefig")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for r in csv.DictReader(open(args.from_csv)):
        rows.append([r["channel"], r["t_budget"], float(r["snr_db"]),
                     r["qtype"], float(r["accuracy"]), int(r["n"]),
                     float(r["mean_tokens_sent"]), float(r["mean_payload_bytes"]),
                     float(r["mean_channel_uses"])])
    ch = args.fig_channel
    sub = [r for r in rows if r[0] == ch]

    f_lab, f_tick, f_leg, f_tag = 8, 7.5, 7, 7
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
    series = {}  # (panel_idx, qt) -> sorted pts, for annotations
    for pi, (ax, snr) in enumerate(zip(axes[:2], (-5.0, 20.0))):
        for qt in list(QT_SYMBOLIC) + ["all"]:
            pts = sorted([(r[6], r[4], r[1]) for r in sub
                          if r[2] == snr and r[3] == qt])
            if pts:
                series[(pi, qt)] = pts
                hero = qt == "all"
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                        color=COLORS[qt], label=DISPLAY.get(qt, qt),
                        linewidth=1.4 if hero else 1.0,
                        markersize=2.8 if hero else 2.4,
                        zorder=5 if hero else 3)
        ax.set_xscale("log")
        ax.set_xlabel("mean tokens sent (top-$t$)", fontsize=f_lab)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.05, f"SNR = {snr:g} dB", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=f_tag, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    # ---- claim annotations (values come straight from the sweep CSV) ------
    def budget_point(pi, qt, budget):
        for x, y, tb in series.get((pi, qt), []):
            if tb == budget:
                return x, y
        return None

    for pi in (0, 1):
        pk = budget_point(pi, "threshold", "32")
        if pk:  # ring on the t=32 peak (the "less is more" claim)
            axes[pi].scatter([pk[0]], [pk[1]], s=34, facecolors="none",
                             edgecolors=COLORS["threshold"], linewidths=0.9,
                             zorder=6)
            if pi == 0:
                axes[pi].annotate("peak $t{=}32$", xy=pk,
                                  xytext=(pk[0] * 1.28, pk[1] + 0.082),
                                  fontsize=5.8, color=COLORS["threshold"],
                                  ha="left",
                                  arrowprops=dict(arrowstyle="-", lw=0.5,
                                                  color=COLORS["threshold"],
                                                  alpha=0.7, shrinkA=0.5,
                                                  shrinkB=2.0))
    sat = budget_point(0, "comparison", "3")
    if sat:
        axes[0].annotate("saturates $t{=}3$", xy=sat,
                         xytext=(sat[0] * 0.36, sat[1] - 0.093), fontsize=5.8,
                         color=COLORS["comparison"], ha="left", va="top",
                         arrowprops=dict(arrowstyle="-", lw=0.5,
                                         color=COLORS["comparison"], alpha=0.7,
                                         shrinkA=0.5, shrinkB=1.5))
    cnt = series.get((0, "counting"))
    if cnt:
        axes[0].text(cnt[-1][0] * 1.15, cnt[-1][1] - 0.155,
                     "still climbing at $t{=}48$", fontsize=5.8,
                     color=COLORS["counting"], ha="right", va="top")

    # ---- (c) cost view: accuracy vs channel uses at 5 dB -------------------
    ax = axes[2]
    for qt in list(QT_SYMBOLIC) + ["all"]:
        pts = sorted([(r[8], r[4]) for r in sub if r[2] == 5.0 and r[3] == qt])
        if pts:
            hero = qt == "all"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=COLORS[qt], label=DISPLAY.get(qt, qt),
                    linewidth=1.4 if hero else 1.0,
                    markersize=2.8 if hero else 2.4, zorder=5 if hero else 3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1.2e4, 1.6e4, 2.0e4]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v/1e4:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"uses / query ($\times 10^4$)", fontsize=f_lab)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.05, "SNR = 5 dB", transform=ax.transAxes, va="bottom",
            ha="left", fontsize=f_tag, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    for k, a in enumerate(axes):
        a.tick_params(labelsize=f_tick)
        a.text(-0.02, 1.05, f"({chr(97 + k)})", transform=a.transAxes,
               fontsize=8, fontweight="bold", va="bottom", ha="right")
    axes[0].set_ylabel("accuracy", fontsize=f_lab)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=f_leg,
               frameon=False, handlelength=1.5, columnspacing=1.0,
               bbox_to_anchor=(0.48, 1.02))
    fig.text(0.995, 0.965, CH_TAG.get(ch, ch), ha="right", va="top",
             fontsize=f_tag, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.4, w_pad=0.7)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{args.out_dir}/F8_token_budget.{ext}")
    fig.savefig(f"{args.out_dir}/F8_token_budget.png", dpi=600)
    plt.close(fig)
    print(f"wrote F8_token_budget.[pdf,svg,png] -> {args.out_dir}")


if __name__ == "__main__":
    main()
