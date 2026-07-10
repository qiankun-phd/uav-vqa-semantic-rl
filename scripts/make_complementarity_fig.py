#!/usr/bin/env python3
"""F6 -- evidence<->question complementarity bar chart.

token-gain  Delta = acc(token, s1) - acc(image, s2)  per question type, pooled over
the three channels (AWGN/Rayleigh/Rician), TEST split only.

The deltas are COMPUTED HERE directly from the canonical 5-question-type merged
prediction logs (outputs/vlm/v3_0_{ch}_predictions.csv), NOT hardcoded, so the
figure can never drift from the data.  This mirrors scripts/build_evidence_
complementarity.py exactly (same test split, same accuracy definition).

IMPORTANT: image_id is read as a STRING via csv.DictReader; the crc32 test split
must see the raw zero-padded id, never an int-coerced one.
"""
from __future__ import annotations

import csv
import math
import zlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
})
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CH = ["awgn", "rayleigh", "rician"]

# a-priori reasoning taxonomy (from question semantics, NOT from results)
REASONING = {
    "presence":    "perceptual",
    "counting":    "symbolic",
    "comparison":  "symbolic",
    "co_presence": "symbolic",
    "threshold":   "symbolic",
}
# bar order: symbolic (reasoning) first, then perceptual; within symbolic keep the
# token-gain magnitude order counting > comparison > co_presence > threshold.
ORDER = ["counting", "comparison", "co_presence", "threshold", "presence"]


def ok(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def is_test(i, frac=0.2):
    return (zlib.crc32(str(i).encode()) % 100) < int(frac * 100)


def se_prop(k, n):
    """Standard error of a single proportion; 0 if n==0."""
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def compute_deltas():
    """(image,q,snr) decision -> {service: correct}; return per-qtype token/image
    accuracy, Delta, and a 95% CI half-width on Delta (independent-proportions SE)."""
    dec = defaultdict(dict)
    qt = {}
    for c in CH:
        p = f"outputs/vlm/v3_0_{c}_predictions.csv"
        try:
            rows = list(csv.DictReader(open(p)))
        except FileNotFoundError:
            print(f"WARN missing {p}")
            continue
        for r in rows:
            if r["service_level"] not in ("1", "2"):
                continue
            if not is_test(r["image_id"]):  # image_id kept as str -> safe crc32 split
                continue
            key = (r["image_id"], r["question"], r["snr_bin"])
            dec[key].setdefault(r["service_level"], ok(r["correct"]))
            qt[key] = r["question_type"]

    agg = defaultdict(lambda: {"1": [0, 0], "2": [0, 0]})  # qt -> service -> [k,n]
    for key, sv in dec.items():
        q = qt[key]
        for s in ("1", "2"):
            if s in sv:
                agg[q][s][1] += 1
                agg[q][s][0] += int(sv[s])

    out = {}
    for q in ORDER:
        if q not in agg:
            continue
        kt, nt = agg[q]["1"]
        ki, ni = agg[q]["2"]
        at = kt / nt if nt else 0.0
        ai = ki / ni if ni else 0.0
        d = at - ai
        se = math.sqrt(se_prop(kt, nt) ** 2 + se_prop(ki, ni) ** 2)
        out[q] = {"token": at, "image": ai, "delta": d, "ci": 1.96 * se}
    return out


def main():
    stats = compute_deltas()
    qt = [q for q in ORDER if q in stats]
    delta = [stats[q]["delta"] for q in qt]
    ci = [stats[q]["ci"] for q in qt]
    colors = ["#5ad19a" if REASONING[q] == "symbolic" else "#ffb454" for q in qt]

    print("=== F6 token-gain (pooled 3ch, test) [computed, not hardcoded] ===")
    for q in qt:
        s = stats[q]
        print(f"  {q:12s} {REASONING[q]:11s} token={s['token']:.3f} "
              f"image={s['image']:.3f}  Delta={s['delta']:+.3f} +/-{s['ci']:.3f}")

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    bars = ax.bar(qt, delta, color=colors, edgecolor="#2a3340",
                  yerr=ci, error_kw=dict(ecolor="0.25", lw=0.8, capsize=2))
    ax.axhline(0, color="#2a3340", lw=0.8)
    for b, d, e in zip(bars, delta, ci):
        off = (e + 0.010) if d >= 0 else -(e + 0.020)
        ax.text(b.get_x() + b.get_width() / 2, d + off,
                f"{d:+.3f}", ha="center", fontsize=6.5)
    ax.set_ylabel("token-gain  Δ = acc(token) − acc(image)")
    ax.text(0.02, 0.95, "Δ>0 → detector token wins\nΔ<0 → image wins",
            transform=ax.transAxes, va="top", fontsize=6.5,
            bbox=dict(boxstyle="round", fc="#f2f6ff", ec="#4ea1ff"))
    ax.legend(handles=[Patch(color="#5ad19a", label="symbolic (count-based)"),
                       Patch(color="#ffb454", label="perceptual (existence)")],
              loc="upper right", fontsize=6.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(f"outputs/figures/comparison/F6_complementarity.{ext}", dpi=300)
    print("wrote F6_complementarity")


if __name__ == "__main__":
    raise SystemExit(main())
