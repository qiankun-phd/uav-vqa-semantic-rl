#!/usr/bin/env python
"""M2: v8 convergence triple figure + lambda_QC golden contrast (paper II).

Zero new compute: reads the v8 training traces
(outputs/rl/eps_recal_v8/proposed_{0,1,2}{,_nomtrain}/ppo_training_trace.csv)
and renders

  * fig_m2_convergence_triple.pdf  (a) episode return, (b) quality-critical
    constraint cost vs its 0.02 limit, (c) dual trajectories lambda_QC and
    lambda_conflict -- peak-trained vs nominal-trained, 3-seed mean +- std.
  * fig_m2_lambda_qc_golden.pdf    standalone lambda_QC contrast: peak ratchets
    to the cap 8.0 (infeasibility signature) vs nominal interior ~3.3 (shadow
    price), with the dual warm-up window (episodes 0-149) shaded.
  * m2_convergence_series.csv      aggregated series for local re-plotting.

Outputs land in outputs/rl/m2_convergence_v8/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home() / "phd_research/vqa_main"
V8 = ROOT / "outputs/rl/eps_recal_v8"
OUT = ROOT / "outputs/rl/m2_convergence_v8"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (0, 1, 2)
QC_LIMIT = 0.02
CONFLICT_LIMIT = 0.08
LAMBDA_CAP_Q = 8.0
DUAL_WARMUP = 150

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COL_PEAK = "#c0392b"
COL_NOM = "#2471a3"
COL_AUX = "#7d6608"


def read_trace(d: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader((d / "ppo_training_trace.csv").open()))
    out: dict[str, np.ndarray] = {}
    for key in ("episode", "raw_return", "quality_cost_critical", "conflict_cost",
                "escalation_cost", "lambda_quality_critical", "lambda_conflict"):
        out[key] = np.array([float(r[key]) for r in rows])
    return out


def stack(dirs: list[Path], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traces = [read_trace(d) for d in dirs]
    n = min(len(t[key]) for t in traces)
    mat = np.stack([t[key][:n] for t in traces])
    ep = traces[0]["episode"][:n]
    return ep, mat.mean(axis=0), mat.std(axis=0)


def smooth(x: np.ndarray, w: int = 9) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def band(ax, ep, mean, std, color, label, do_smooth=True):
    m = smooth(mean) if do_smooth else mean
    s = smooth(std) if do_smooth else std
    ax.plot(ep, m, color=color, lw=1.2, label=label)
    ax.fill_between(ep, m - s, m + s, color=color, alpha=0.18, lw=0)


peak_dirs = [V8 / f"proposed_{s}" for s in SEEDS]
nom_dirs = [V8 / f"proposed_{s}_nomtrain" for s in SEEDS]

series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
for name, dirs in (("peak", peak_dirs), ("nomtrain", nom_dirs)):
    for key in ("raw_return", "quality_cost_critical", "escalation_cost",
                "lambda_quality_critical", "lambda_conflict"):
        series[f"{name}:{key}"] = stack(dirs, key)

# ------------------------------------------------------------------ CSV dump
with (OUT / "m2_convergence_series.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["series", "episode", "mean", "std"])
    for name, (ep, mean, std) in series.items():
        for e, m, s in zip(ep, mean, std):
            w.writerow([name, int(e), f"{m:.6f}", f"{s:.6f}"])

# ------------------------------------------------------------- triple figure
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1))

ax = axes[0]
for name, col, lab in (("peak", COL_PEAK, "peak-trained"), ("nomtrain", COL_NOM, "nominal-trained")):
    ep, mean, std = series[f"{name}:raw_return"]
    band(ax, ep, mean, std, col, lab)
ax.set_xlabel("training episode")
ax.set_ylabel("episode return")
ax.set_title("(a) return", loc="left")
ax.legend(frameon=False, loc="lower right")

ax = axes[1]
for name, col, lab in (("peak", COL_PEAK, "peak-trained"), ("nomtrain", COL_NOM, "nominal-trained")):
    ep, mean, std = series[f"{name}:quality_cost_critical"]
    band(ax, ep, mean, std, col, lab)
ax.axhline(QC_LIMIT, color="k", ls="--", lw=0.8)
ax.annotate(f"limit $d_{{QC}}={QC_LIMIT}$", xy=(0.02, QC_LIMIT), xycoords=("axes fraction", "data"),
            xytext=(0, 3), textcoords="offset points", fontsize=6.5)
ax.set_xlabel("training episode")
ax.set_ylabel("quality-critical cost $\\bar c_{QC}$")
ax.set_title("(b) constraint cost vs limit", loc="left")
ax.set_ylim(bottom=-0.02)

ax = axes[2]
ep, mean, std = series["peak:lambda_quality_critical"]
band(ax, ep, mean, std, COL_PEAK, "$\\lambda_{QC}$ peak", do_smooth=False)
ep, mean, std = series["nomtrain:lambda_quality_critical"]
band(ax, ep, mean, std, COL_NOM, "$\\lambda_{QC}$ nominal", do_smooth=False)
ep, mean, std = series["peak:lambda_conflict"]
band(ax, ep, mean, std, COL_AUX, "$\\lambda_{conf}$ peak", do_smooth=False)
ax.axhline(LAMBDA_CAP_Q, color=COL_PEAK, ls=":", lw=0.8)
ax.annotate("cap $\\lambda^{max}_{Q}=8$", xy=(0.02, LAMBDA_CAP_Q), xycoords=("axes fraction", "data"),
            xytext=(0, -8), textcoords="offset points", fontsize=6.5, color=COL_PEAK)
ax.axvspan(0, DUAL_WARMUP, color="gray", alpha=0.12, lw=0)
ax.annotate("dual warm-up", xy=(DUAL_WARMUP / 2, 0.97), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=6.5, color="dimgray")
ax.set_xlabel("training episode")
ax.set_ylabel("dual variable $\\lambda$")
ax.set_title("(c) dual trajectories", loc="left")
ax.legend(frameon=False, loc="center right", fontsize=6.5)

fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_m2_convergence_triple.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- golden fig
fig, ax = plt.subplots(figsize=(3.5, 2.3))
ep, mean, std = series["peak:lambda_quality_critical"]
band(ax, ep, mean, std, COL_PEAK, "peak-trained (infeasible domain)", do_smooth=False)
ep, mean, std = series["nomtrain:lambda_quality_critical"]
band(ax, ep, mean, std, COL_NOM, "nominal-trained (feasible domain)", do_smooth=False)
ax.axhline(LAMBDA_CAP_Q, color=COL_PEAK, ls=":", lw=0.9)
ax.annotate("per-channel cap $\\lambda^{max}_{Q}=8$", xy=(0.03, LAMBDA_CAP_Q),
            xycoords=("axes fraction", "data"), xytext=(0, -9), textcoords="offset points",
            fontsize=7, color=COL_PEAK)
ax.axvspan(0, DUAL_WARMUP, color="gray", alpha=0.12, lw=0)
ax.annotate("dual warm-up\n(ascent frozen)", xy=(DUAL_WARMUP / 2, 0.98),
            xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7, color="dimgray")
ax.set_xlabel("training episode")
ax.set_ylabel("$\\lambda_{QC}$ (quality-critical dual)")
ax.legend(frameon=False, loc="center right", fontsize=7)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_m2_lambda_qc_golden.pdf", bbox_inches="tight")
plt.close(fig)

print("M2 figures written to", OUT)
for f in sorted(OUT.iterdir()):
    print("  ", f.name)
