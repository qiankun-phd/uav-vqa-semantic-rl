#!/usr/bin/env python3
"""naturefig Fig. 1: evidence-level semantic-communication system overview (TGCN).

Pure vector schematic (no data files): UAV side (camera -> YOLOv8n -> token
packet -> LDPC r=1/2 + BPSK  ||  JPEG + rate-adaptive LDPC full-image path)
-> block-fading channel -> edge side (symbolic decoder reads tokens || frozen
VLM reads the reconstructed image) -> answer.  The question-type router sits
at the transmit decision point (symbolic -> tokens, perception -> image; the
per-sample energy price lambda is the budget knob), and each path carries its
MEASURED per-answer energy (token 0.435 J vs image ~33 J, split into answerer
compute vs radio).  Every number is lifted verbatim from the manuscript
(Secs. III, VI-E/F/H; Tables `tab:power`, `tab:jpa`); nothing is invented.

Style: nature-figure conventions -- 7.16 in final width, all text >= 8 pt,
grayscale-safe fills + hatches, no in-figure title, Type-42 fonts.
Outputs SVG + PDF + PNG (300 dpi).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "hatch.linewidth": 0.6,
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---- palette: same method families as every quantitative figure ------------
C_TOKEN = "#9aa7b4"   # fixed-token grey-blue
C_TOKEN_F = "#edf0f3"
C_IMAGE = "#ffb454"   # rate-adaptive-image orange
C_IMAGE_F = "#fff3e0"
C_ROUTE = "#5ad19a"   # evidence-routing green
C_ROUTE_F = "#e8f8f0"
C_ROUTE_T = "#1f7a54"
C_INK = "#2a3340"
C_CHAN = "#7f9ab8"
C_CHAN_F = "#eef4fb"
C_CMP = "#4a5763"     # answerer-compute energy (dark)
C_TX = "#e8c268"      # radio energy (light, hatched)
C_MUTE = "#5c6a78"

W, H = 7.16, 3.42
OUT = Path(__file__).resolve().parents[1] / "outputs/figures/naturefig"


def box(ax, x0, y0, x1, y1, text, fc, ec, fs=8, bold=False, lw=1.0,
        style="round,pad=0.02,rounding_size=0.05", tc=None, ls="-", z=3):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=style, fc=fc, ec=ec, lw=lw,
                                linestyle=ls, mutation_aspect=1.0, zorder=z))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=fs, color=tc or C_INK, zorder=4,
                fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, pts, color=C_INK, lw=1.1, ls="-", head=True, z=2):
    """Poly-line arrow through pts; head on the last segment."""
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        ax.add_patch(FancyArrowPatch(
            pts[i], pts[i + 1],
            arrowstyle=("-|>" if (head and last) else "-"),
            mutation_scale=9, lw=lw, linestyle=ls, color=color,
            shrinkA=0, shrinkB=0, zorder=z))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- region backdrops ---------------------------------------------------
    y_reg0, y_reg1 = 0.68, 3.37
    box(ax, 0.05, y_reg0, 3.55, y_reg1, "", "#f6f8fa", "#c8d0d8", lw=0.8, z=1)
    box(ax, 4.25, y_reg0, 7.11, y_reg1, "", "#f6f8fa", "#c8d0d8", lw=0.8, z=1)
    ax.text(0.16, 3.235, "UAV (transmitter)", fontsize=9, fontweight="bold",
            color=C_MUTE, va="center", zorder=4)
    ax.text(7.00, 3.235, "edge server (receiver)", fontsize=9,
            fontweight="bold", color=C_MUTE, va="center", ha="right", zorder=4)

    yT0, yT1 = 2.56, 3.04          # token lane band
    yI0, yI1 = 0.84, 1.32          # image lane band
    yTc, yIc = (yT0 + yT1) / 2, (yI0 + yI1) / 2

    # ---- UAV side: camera, detector, encoders -------------------------------
    box(ax, 0.14, 1.65, 0.63, 2.13, "aerial\nframe", "white", C_INK, lw=0.9)
    box(ax, 0.73, yT0, 1.53, yT1, "YOLOv8n\ndetector", C_TOKEN_F, C_TOKEN)
    box(ax, 1.63, yT0, 2.59, yT1, "tokens: cls, box,\nconf (0.7–1.2 KB)",
        C_TOKEN_F, C_TOKEN)
    box(ax, 2.69, yT0, 3.47, yT1, "LDPC $r{=}1/2$\n+ BPSK", C_TOKEN_F, C_TOKEN)
    arrow(ax, [(1.53, yTc), (1.63, yTc)])
    arrow(ax, [(2.59, yTc), (2.69, yTc)])

    box(ax, 1.63, yI0, 2.59, yI1, "JPEG encoder\n(rate-adaptive)",
        C_IMAGE_F, C_IMAGE)
    box(ax, 2.69, yI0, 3.47, yI1, "LDPC\n(rate-adaptive)", C_IMAGE_F, C_IMAGE)
    arrow(ax, [(2.59, yIc), (2.69, yIc)])
    # camera elbows: up to the detector, down to the JPEG encoder
    arrow(ax, [(0.385, 2.13), (0.385, yTc), (0.73, yTc)])
    arrow(ax, [(0.385, 1.65), (0.385, yIc), (1.63, yIc)])

    # ---- router (transmit decision point) -----------------------------------
    rx0, rx1, ry0, ry1 = 1.05, 2.16, 1.65, 2.19
    box(ax, rx0, ry0, rx1, ry1,
        "question-type\nrouter\n(no CSI needed)", C_ROUTE_F, C_ROUTE,
        bold=True, lw=1.3)
    # question q enters the router from below (ground-user query); it hops
    # over the frame bus (white break in the bus line at the crossing)
    qx = 1.45
    ax.add_patch(Rectangle((qx - 0.05, yIc - 0.035), 0.10, 0.07, fc="white",
                           ec="none", zorder=3.3))
    arrow(ax, [(qx, 0.98), (qx, ry0)], color=C_ROUTE, lw=1.2, z=3.5)
    ax.text(qx, 0.94, "question $q$", fontsize=8, color=C_INK,
            ha="center", va="top")
    # routing control: dashed green to each lane's coded output
    arrow(ax, [(rx1, 1.92), (3.08, 1.92), (3.08, yT0 - 0.015)],
          color=C_ROUTE, ls="--", lw=1.2, z=3.5)
    arrow(ax, [(3.08, 1.92), (3.08, yI1 + 0.015)], color=C_ROUTE, ls="--",
          lw=1.2, z=3.5)
    ax.text(2.05, 2.375, "symbolic $q$ $\\to$ tokens:  counting ·\n"
            "comparison · co-presence · threshold", fontsize=8,
            color=C_ROUTE_T, va="center", ha="center")
    ax.text(2.42, 1.485, "perception $q$ $\\to$ image:\npresence",
            fontsize=8, color=C_ROUTE_T, va="center", ha="center")

    # lambda budget knob (per-sample energy price)
    kx, ky, kr = 0.55, 1.30, 0.13
    th = np.linspace(np.pi * 0.15, np.pi * 0.85, 40)
    ax.plot(kx + kr * np.cos(th), ky + kr * np.sin(th), color=C_ROUTE, lw=1.6,
            zorder=4, solid_capstyle="round")
    ang = np.pi * 0.38
    ax.plot([kx, kx + 0.95 * kr * np.cos(ang)],
            [ky, ky + 0.95 * kr * np.sin(ang)], color=C_INK, lw=1.4, zorder=4)
    ax.plot([kx], [ky], marker="o", ms=2.5, color=C_INK, zorder=5)
    ax.text(kx - 0.28, 1.02, "energy price $\\lambda$\n(budget knob)",
            fontsize=8, ha="left", va="top", color=C_INK)
    arrow(ax, [(kx + 0.10, ky + 0.10), (rx0 + 0.10, ry0)], color=C_ROUTE,
          ls="--", lw=1.0, z=3.5)

    # ---- channel band --------------------------------------------------------
    cx0, cx1 = 3.62, 4.18
    box(ax, cx0, y_reg0, cx1, y_reg1, "", C_CHAN_F, C_CHAN, lw=0.9)
    ax.text((cx0 + cx1) / 2, (yI1 + yT0) / 2, "block-fading channel\n"
            "$\\gamma = -5\\ldots20$ dB",
            fontsize=8, rotation=90, ha="center", va="center", color=C_INK,
            zorder=4)
    for yc in (yTc, yIc):  # fading squiggle where each lane crosses
        xs = np.linspace(cx0 + 0.06, cx1 - 0.06, 60)
        ax.plot(xs, yc - 0.15 + 0.028 * np.sin(2 * np.pi * (xs - cx0) / 0.19),
                color=C_CHAN, lw=0.8, zorder=3)
    # lane crossings + per-decision payload tags
    arrow(ax, [(3.47, yTc), (4.36, yTc)], color=C_TOKEN, lw=1.3, z=3.5)
    arrow(ax, [(3.47, yIc), (4.36, yIc)], color=C_IMAGE, lw=1.3, z=3.5)
    ax.text((cx0 + cx1) / 2, yT1 + 0.055, "0.9 KB", fontsize=8, ha="center",
            va="bottom", color=C_INK, fontweight="bold", zorder=4)
    ax.text((cx0 + cx1) / 2, yI0 - 0.055, "14–215 KB", fontsize=8,
            ha="center", va="top", color=C_INK, fontweight="bold", zorder=4)

    # ---- edge side: two answerers -> answer ----------------------------------
    box(ax, 4.36, yT0, 5.66, yT1, "symbolic decoder\n(rule-based, 0 J)",
        C_TOKEN_F, C_TOKEN)
    box(ax, 4.36, yI0, 5.66, yI1, "frozen VLM\n(Qwen2-VL-2B)",
        C_IMAGE_F, C_IMAGE)
    ax.text(5.01, yI1 + 0.055, "reads the reconstructed image", fontsize=8,
            style="italic", color=C_MUTE, ha="center", va="bottom")
    ax.text(5.01, yT0 - 0.055, "reads the decoded tokens", fontsize=8,
            style="italic", color=C_MUTE, ha="center", va="top")
    bx0, bx1, by0, by1 = 6.24, 7.00, 1.66, 2.14
    box(ax, bx0, by0, bx1, by1, "answer\n$\\hat A_q$", "white", C_INK,
        bold=True, lw=1.2)
    arrow(ax, [(5.66, yTc), (5.97, yTc), (5.97, 2.02), (bx0, 2.02)],
          color=C_TOKEN, lw=1.3)
    arrow(ax, [(5.66, yIc), (5.97, yIc), (5.97, 1.78), (bx0, 1.78)],
          color=C_IMAGE, lw=1.3)

    # ---- measured per-answer energy chips (compute vs radio) -----------------
    def chip(x0, x1, title, tag, segs, ec, bar_label):
        y0c, y1c = 0.08, 0.56
        box(ax, x0, y0c, x1, y1c, "", "white", ec, lw=1.2)
        ax.text(x0 + 0.08, y1c - 0.115, title, fontsize=8, fontweight="bold",
                va="center", color=C_INK, zorder=5)
        ax.text(x1 - 0.08, y1c - 0.115, tag, fontsize=8, va="center",
                ha="right", color=C_INK, zorder=5)
        bar_x0, bar_x1 = x0 + 0.08, x1 - 0.08
        bar_y0, bar_h = y0c + 0.06, 0.145
        tot = sum(v for v, _, _ in segs)
        xcur = bar_x0
        for val, fc, hatch in segs:
            wseg = (bar_x1 - bar_x0) * val / tot
            ax.add_patch(Rectangle((xcur, bar_y0), wseg, bar_h, fc=fc,
                                   ec=C_INK, lw=0.5, hatch=hatch, zorder=4))
            xcur += wseg
        ax.text(bar_x0 + 0.04, bar_y0 + bar_h / 2, bar_label, fontsize=8,
                color="white", va="center", zorder=5)

    # token chip: 0.435 J = detector 0.43 J + radio <~0.01 J (Sec. VI-E)
    chip(0.28, 2.88, "token path: 0.435 J / answer",
         "radio $\\lesssim$0.01 J", [(0.43, C_CMP, None),
                                     (0.0075, C_TX, "////")],
         C_TOKEN, "detector compute 0.43 J")
    # image chip: ~33 J = VLM 32.3 J + radio <= 2.7 J (Tables IV/V)
    chip(4.28, 6.88, "image path: $\\approx$33 J / answer",
         "radio $\\leq$2.7 J", [(32.3, C_CMP, None), (2.7, C_TX, "////")],
         C_IMAGE, "VLM compute 32.3 J")

    # 75x per-answer energy gap bracket between the chips
    ax.add_patch(FancyArrowPatch((2.94, 0.26), (4.22, 0.26),
                                 arrowstyle="<|-|>", mutation_scale=9,
                                 lw=1.1, color=C_INK, zorder=4))
    ax.text(3.58, 0.315, "$\\approx$75$\\times$", fontsize=8.5,
            fontweight="bold", ha="center", va="bottom", color=C_INK,
            zorder=5)

    for stem, dpi in (("F12_architecture.svg", None),
                      ("F12_architecture.pdf", None),
                      ("F12_architecture.png", 300)):
        fig.savefig(OUT / stem, dpi=dpi)
    print(f"wrote {OUT}/F12_architecture.[svg|pdf|png]")


if __name__ == "__main__":
    main()
