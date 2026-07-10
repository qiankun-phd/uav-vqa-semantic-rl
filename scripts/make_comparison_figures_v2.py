#!/usr/bin/env python3
"""Final comparison figures from comparison_all.csv (tidy long format).

F1  3-panel accuracy vs SNR (one panel per channel), all methods
    -- sized for an IEEE two-column figure* (7.16 in text width)
F2  cliff effect (one channel): fixed-rate digital cliffs vs graceful adaptive
F4  per-question-type grouped bars at a fixed SNR (one channel)
F5  Pareto: accuracy vs channel uses per query (bandwidth-fair across analog/digital)
    -- F2/F4/F5 sized for an IEEE single column (3.5 in)

IEEE styling: descriptive legend names (no internal M0..M6 codes), fonts
>= ~8 pt at final size for the figure* and ~7-8 pt for single-column
figures, no in-figure titles, vector PDF with embedded (Type 42) fonts.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "lines.linewidth": 1.2, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
})
import matplotlib.pyplot as plt

# Descriptive method names (paper-facing; internal codes stay in the CSVs).
STYLE = {
    "M0_errorfree": ("#444444", "*", "Error-free image (ideal)"),
    "M0_naive":    ("#ff6b6b", "x", "Fixed-rate image"),
    "M1_image":    ("#ffb454", "o", "Rate-adaptive image"),
    "M2_analog":   ("#c678dd", "v", "Uncoded analog"),
    "M3_token":    ("#9aa7b4", "s", "Fixed token"),
    "M4_adaptive": ("#5ad19a", "D", "Evidence routing (ours)"),
    "M5_oracle":   ("#4ea1ff", "^", "Oracle (upper bound)"),
}
ORDER = ["M0_errorfree", "M5_oracle", "M4_adaptive", "M2_analog", "M3_token", "M1_image", "M0_naive"]
DASHED = {"M5_oracle": "--", "M0_errorfree": ":"}
CH_TITLE = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician K=6 dB"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["snr_db"] = float(r["snr_db"]); r["accuracy"] = float(r["accuracy"])
        r["mean_payload_bytes"] = float(r["mean_payload_bytes"]); r["lcb"] = float(r["lcb"]); r["ucb"] = float(r["ucb"])
        r["mean_channel_uses"] = float(r.get("mean_channel_uses") or 0.0)
        r["cbr"] = float(r.get("cbr") or 0.0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default="outputs/figures/comparison")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--bar-snr", type=float, default=5.0)
    ap.add_argument("--pareto-channel", default="rician")
    ap.add_argument("--cliff-channel", default="rayleigh")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load(args.csv)
    channels = [c for c in ["awgn", "rayleigh", "rician"] if any(r["channel"] == c for r in rows)]

    # ---- F1: 3-panel acc vs SNR (qtype=all, split=test); figure* width ----
    fig, axes = plt.subplots(1, len(channels), figsize=(7.16, 2.15), sharey=True)
    if len(channels) == 1:
        axes = [axes]
    # Wilson 95% band drawn for the adaptive policy and the two key baselines.
    BAND_METHODS = {"M4_adaptive", "M1_image", "M3_token"}
    for ax, ch in zip(axes, channels):
        for m in ORDER:
            pts = sorted([(r["snr_db"], r["accuracy"], r["lcb"], r["ucb"]) for r in rows
                          if r["channel"] == ch and r["method"] == m and r["qtype"] == "all"])
            if not pts:
                continue
            col, mk, lab = STYLE[m]
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1] for p in pts], DASHED.get(m, "-"), color=col, marker=mk,
                    label=lab, markersize=3.2)
            if m in BAND_METHODS:
                ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts],
                                color=col, alpha=0.15, linewidth=0)
        # per-panel channel label (small annotation, IEEE: no big titles)
        ax.text(0.04, 0.05, CH_TITLE.get(ch, ch), transform=ax.transAxes,
                va="bottom", ha="left", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.set_xlabel("SNR (dB)"); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("VQA answer accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7,
               frameon=False, borderaxespad=0.1, handlelength=1.8,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.85), pad=0.4, w_pad=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F1_acc_snr_3panel_{args.tag}.{ext}",
                    dpi=300)
    plt.close(fig)

    # ---- F2: cliff effect (one channel); single-column width ----
    cliff_ch = args.cliff_channel
    if any(r["channel"] == cliff_ch for r in rows):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for m in ["M0_errorfree", "M5_oracle", "M4_adaptive", "M1_image", "M2_analog", "M0_naive"]:
            pts = sorted([(r["snr_db"], r["accuracy"]) for r in rows
                          if r["channel"] == cliff_ch and r["method"] == m and r["qtype"] == "all"])
            if not pts:
                continue
            col, mk, lab = STYLE[m]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], DASHED.get(m, "-"), color=col, marker=mk,
                    label=lab, markersize=3.2)
        ax.axhline(0.5, ls=":", color="#888", lw=0.8)
        ax.text(0.02, 0.502, "chance $\\approx$ 0.5", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", color="#666", fontsize=6.5)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("VQA answer accuracy")
        ax.text(0.03, 0.97, CH_TITLE.get(cliff_ch, cliff_ch), transform=ax.transAxes,
                va="top", ha="left", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.4, loc="lower right", handlelength=1.8,
                  labelspacing=0.3, borderpad=0.35)
        fig.tight_layout(pad=0.4)
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.out_dir}/F2_cliff_{args.tag}.{ext}", dpi=300)
        plt.close(fig)

    # ---- F4: per-question-type grouped bars at fixed SNR (rician) ----
    ch = args.pareto_channel
    qtypes = sorted({r["qtype"] for r in rows if r["qtype"] not in ("all",)})
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]
    import numpy as np
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(qtypes)); w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        vals, lo, hi = [], [], []
        for qt in qtypes:
            rr = [r for r in rows if r["channel"] == ch and r["method"] == m
                  and r["qtype"] == qt and abs(r["snr_db"] - args.bar_snr) < 1e-6]
            if rr:
                vals.append(rr[0]["accuracy"])
                lo.append(rr[0]["accuracy"] - rr[0]["lcb"])
                hi.append(rr[0]["ucb"] - rr[0]["accuracy"])
            else:
                vals.append(0); lo.append(0); hi.append(0)
        col, mk, lab = STYLE[m]
        ax.bar(x + i * w, vals, w, color=col, label=lab,
               yerr=[lo, hi], error_kw=dict(ecolor="0.3", lw=0.6, capsize=1.4))
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(qtypes, fontsize=6.8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.text(0.99, 0.98, f"{CH_TITLE.get(ch, ch)}, SNR = {args.bar_snr:g} dB",
            transform=ax.transAxes, va="top", ha="right", fontsize=6.5, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6,
               frameon=False, handlelength=1.0, columnspacing=0.8,
               labelspacing=0.3, bbox_to_anchor=(0.5, 1.02))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(rect=(0, 0, 1, 0.82), pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F4_qtype_{args.tag}.{ext}", dpi=300)
    plt.close(fig)

    # ---- F5: Pareto acc vs channel uses per query (unified analog/digital axis) ----
    use_uses = any(r["mean_channel_uses"] > 0 for r in rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    pareto_methods = (["M0_naive", "M1_image", "M2_analog", "M3_token", "M4_adaptive"]
                      if use_uses else ["M1_image", "M3_token", "M4_adaptive"])
    xkey = "mean_channel_uses" if use_uses else "mean_payload_bytes"
    for m in pareto_methods:
        pts = [(r[xkey], r["accuracy"], r["lcb"], r["ucb"]) for r in rows
               if r["channel"] == ch and r["method"] == m and r["qtype"] == "all" and r[xkey] > 0]
        if not pts:
            continue
        col, mk, lab = STYLE[m]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        yerr = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
        # Wilson 95% vertical bars (bandwidth-fair Pareto): show only for the
        # routing policy to avoid clutter; other methods drawn as bare markers.
        if m == "M4_adaptive":
            ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=col, elinewidth=0.8,
                        capsize=1.6, alpha=0.7, zorder=2)
        ax.scatter(xs, ys, color=col, marker=mk, s=22, label=lab, zorder=3)
    ax.set_xscale("log")
    if use_uses:
        # CBR normalization spelled out in the caption (keeps the label short).
        ax.set_xlabel("mean complex channel uses per query")
    else:
        ax.set_xlabel("mean payload (bytes / query)")
    ax.set_ylabel("accuracy")
    ax.text(0.03, 0.97, CH_TITLE.get(ch, ch), transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=6.4, loc="lower left", handlelength=1.4,
              labelspacing=0.3, borderpad=0.35)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_dir}/F5_pareto_{args.tag}.{ext}", dpi=300)
    plt.close(fig)

    print(f"wrote F1(3-panel)/F2/F4/F5 [{args.tag}] -> {args.out_dir}; channels={channels}")


if __name__ == "__main__":
    raise SystemExit(main())
