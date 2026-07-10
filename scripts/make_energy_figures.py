#!/usr/bin/env python3
"""Joint communication+computation energy accounting for paper-1 TGCN (A1).

Reads the five-question-type tidy report (outputs/reports/comparison_v3_5qt.csv,
the same source as figures F1/F2/F4/F5) and the measured GPU power phases
(outputs/energy/gpu_power_phases.json, from scripts/measure_gpu_energy.py),
and emits under --out-dir:

  F5_pareto_energy.{pdf,png}      accuracy vs measured J/answer (log x, Rician)
  F11_answers_per_joule.{pdf,png} correct answers per joule vs SNR (Rician)
  jpa_main_table.tex              J/answer main-table rows (Rician, -5/5/20 dB)
  jpa_sensitivity.tex             P_tx sweep 0.1--1 W at -5 dB (M3/M4/M1)
  energy_summary.json             every computed number used in the paper text

Energy model, per answered question (Section III energy accounting):
  E(method, snr) = E_tx + E_cmp
  E_tx  = mean_channel_uses / B * P_tx     (B = 1 MHz; P_tx parameterized,
                                            3GPP TR 36.777 aerial-UE anchor
                                            23 dBm ~ 0.2 W, grid 0.1--1 W)
  E_cmp = f_img * E_vlm + (1 - f_img) * E_det   per the method's route mix
where E_vlm is the measured RTX 4060 (edge-server proxy) energy per VLM
answer (incremental over idle = headline; total also reported) and E_det is
the onboard detector energy, parameterized as a Jetson Orin Nano band
(7--15 W module power x a 15--50 ms/frame latency band; the 4060-measured
detector energy is kept as a cross-check). Symbolic decode is CPU-only and
billed as 0 J (declared).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
})

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

BANDWIDTH_HZ = 1.0e6
P_TX_GRID = [0.1, 0.25, 0.5, 1.0]
P_TX_HEAD = 0.5  # headline transmit power (within the 0.1--1 W A2G band)

# Onboard detector (token path) energy band: Jetson Orin Nano class.
# 7--15 W configurable module power (NVIDIA datasheet) x 15--50 ms/frame
# (TensorRT FP16 .. eager upper bound for a nano-scale detector).
E_DET_LO_J = 7.0 * 0.015
E_DET_HI_J = 15.0 * 0.050
E_DET_MID_J = 0.5 * (E_DET_LO_J + E_DET_HI_J)

# M4 zero-parameter rule route mix: presence -> image; test-set presence
# share 410/936 unique questions (constant across SNR replays).
F_IMG_M4 = 410.0 / 936.0

# M6 (compact DeepJSCC) is not in the tidy CSV; Rician grid from the paper
# campaign (outputs/djscc; Table "Learned baseline M6", all five types).
M6_USES = 9.2215e4
M6_ACC = {-5.0: 0.564, 0.0: 0.578, 5.0: 0.592, 10.0: 0.599, 15.0: 0.600, 20.0: 0.596}

# Descriptive method names (paper-facing; internal codes stay in the CSVs).
STYLE = {  # color, marker, label, linestyle (distinct ls = grayscale-safe)
    "M0_naive":    ("#ff6b6b", "x", "Fixed-rate image", ":"),
    "M1_image":    ("#ffb454", "o", "Rate-adaptive image", "-"),
    "M2_analog":   ("#c678dd", "v", "Uncoded analog", "--"),
    "M6_djscc":    ("#8b5e3c", "P", "DJSCC (learned)", "-."),
    "M3_token":    ("#9aa7b4", "s", "Fixed token", "-"),
    "M4_adaptive": ("#5ad19a", "D", "Evidence routing (ours)", "-"),
}
ORDER = ["M4_adaptive", "M3_token", "M6_djscc", "M2_analog", "M1_image", "M0_naive"]
F_IMG = {  # image-route (VLM-invoking) fraction per method
    "M0_naive": 1.0, "M1_image": 1.0, "M2_analog": 1.0, "M6_djscc": 1.0,
    "M3_token": 0.0, "M4_adaptive": F_IMG_M4,
}


def load_rows(csv_path: Path, channel: str) -> dict[str, dict[float, dict]]:
    out: dict[str, dict[float, dict]] = {}
    for r in csv.DictReader(open(csv_path)):
        if r["channel"] != channel or r["qtype"] != "all" or r["split"] != "test":
            continue
        m = r["method"]
        out.setdefault(m, {})[float(r["snr_db"])] = {
            "acc": float(r["accuracy"]),
            "uses": float(r["mean_channel_uses"] or 0.0),
            "lcb": float(r["lcb"]), "ucb": float(r["ucb"]),
        }
    # Inject M6 (constant uses; no lcb/ucb tracked here).
    out["M6_djscc"] = {s: {"acc": a, "uses": M6_USES, "lcb": None, "ucb": None}
                       for s, a in M6_ACC.items()}
    return out


def energy_j(method: str, uses: float, p_tx: float, e_vlm: float,
             e_det: float = E_DET_MID_J) -> tuple[float, float, float]:
    """Returns (E_total, E_tx, E_cmp) per answered question in joules."""
    e_tx = uses / BANDWIDTH_HZ * p_tx
    f = F_IMG[method]
    e_cmp = f * e_vlm + (1.0 - f) * e_det
    return e_tx + e_cmp, e_tx, e_cmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/reports/comparison_v3_5qt.csv")
    ap.add_argument("--power-json", default="outputs/energy/gpu_power_phases.json")
    ap.add_argument("--out-dir", default="outputs/energy")
    ap.add_argument("--channel", default="rician")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(REPO / args.csv, args.channel)
    power = json.loads((REPO / args.power_json).read_text())
    ph = power["phases"]
    p_idle = power["idle_baseline_w"]
    e_vlm_inc = ph["vlm"]["joule_per_item_incremental"]
    e_vlm_tot = ph["vlm"]["joule_per_item_total"]
    e_det_4060_inc = ph["yolo"]["joule_per_item_incremental"]

    snrs = sorted(rows["M4_adaptive"].keys())

    # ---- per-method energy tables (headline: incremental VLM, P_tx=0.5 W) --
    summary: dict = {
        "params": {
            "bandwidth_hz": BANDWIDTH_HZ, "p_tx_grid_w": P_TX_GRID,
            "p_tx_headline_w": P_TX_HEAD,
            "e_vlm_incremental_j": e_vlm_inc, "e_vlm_total_j": e_vlm_tot,
            "e_det_jetson_band_j": [E_DET_LO_J, E_DET_HI_J],
            "e_det_mid_j": E_DET_MID_J,
            "e_det_4060_incremental_j": e_det_4060_inc,
            "f_img_m4": F_IMG_M4, "idle_w": p_idle,
            "vlm_power_w": ph["vlm"]["power_avg_w"],
            "vlm_sec_per_item": ph["vlm"]["sec_per_item_mean"],
            "yolo_power_w": ph["yolo"]["power_avg_w"],
            "yolo_sec_per_item": ph["yolo"]["sec_per_item_mean"],
        },
        "per_method": {},
    }
    for m in ORDER:
        summary["per_method"][m] = {}
        for s in snrs:
            if s not in rows[m]:
                continue
            e, e_tx, e_cmp = energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc)
            e_tot, _, _ = energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_tot)
            summary["per_method"][m][s] = {
                "acc": rows[m][s]["acc"], "uses": rows[m][s]["uses"],
                "e_tx_j": e_tx, "e_cmp_j": e_cmp, "j_per_answer": e,
                "j_per_answer_totalpower": e_tot,
                "answers_per_joule": rows[m][s]["acc"] / e,
            }

    # ---- F5 (energy axis): accuracy vs J/answer, log x ---------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for m in ORDER:
        pts = [(summary["per_method"][m][s]["j_per_answer"],
                rows[m][s]["acc"], s) for s in snrs if s in rows[m]]
        c, mk, lb, ls = STYLE[m]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, marker=mk, color=c, label=lb, lw=1.0, ms=3.2, ls=ls,
                zorder=5 if m == "M4_adaptive" else 3)
        if m == "M4_adaptive":  # Wilson bars
            for s, x, y in zip([p[2] for p in pts], xs, ys):
                lc, uc = rows[m][s]["lcb"], rows[m][s]["ucb"]
                ax.vlines(x, lc, uc, color=c, alpha=0.55, lw=0.9, zorder=4)
        if m == "M3_token":  # Jetson detector-energy band (dominates M3 E)
            e_lo = [energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc, E_DET_LO_J)[0] for s in snrs]
            e_hi = [energy_j(m, rows[m][s]["uses"], P_TX_HEAD, e_vlm_inc, E_DET_HI_J)[0] for s in snrs]
            for ylo_x, yhi_x, y in zip(e_lo, e_hi, ys):
                ax.hlines(y, ylo_x, yhi_x, color=c, alpha=0.4, lw=2.2, zorder=2)
    # C1: energy-controllable per-sample frontier (lambda sweep over the
    # per-sample correctness predictors; scripts/p1_review_fix_analysis.py).
    frontier_csv = out_dir / f"c1_frontier_{args.channel}.csv"
    if frontier_csv.exists():
        fr = sorted((float(r["E_j"]), float(r["acc"]))
                    for r in csv.DictReader(open(frontier_csv)))
        ax.plot([p[0] for p in fr], [p[1] for p in fr], color="#2f2f2f",
                ls="--", lw=0.9, marker=".", ms=2.2, zorder=6,
                label="Budget-tunable per-sample routing")
        ax.annotate("energy price $\\lambda$ sweep", (0.55, 0.702),
                    fontsize=5.5, color="#2f2f2f", ha="left")
    # annotate SNR direction on M4 and M1 with offset-point labels placed in
    # the empty regions (the -5 dB label previously collided with markers).
    ann = {("M4_adaptive", snrs[-1]): ((0, 5), "center"),
           ("M4_adaptive", snrs[0]): ((6, -7), "left"),
           ("M1_image", snrs[-1]): ((-7, -1), "right"),
           ("M1_image", snrs[0]): ((0, -12), "center")}
    for (m, s), (off, ha) in ann.items():
        e = summary["per_method"][m][s]["j_per_answer"]
        ax.annotate(f"{s:+.0f} dB", (e, rows[m][s]["acc"]),
                    textcoords="offset points", xytext=off,
                    fontsize=5.5, color=STYLE[m][0], ha=ha,
                    bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.15))
    # fixed-rate -5 dB outlier: name the mechanism inside the figure
    e_n5 = summary["per_method"]["M0_naive"][snrs[0]]["j_per_answer"]
    ax.annotate("fixed-rate @$-5$ dB:\ndigital cliff (FER${\\approx}$1)",
                (e_n5, rows["M0_naive"][snrs[0]]["acc"]),
                textcoords="offset points", xytext=(-8, 2), fontsize=5.0,
                color=STYLE["M0_naive"][0], ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("joint energy per answer (J)", fontsize=8)
    ax.set_ylabel("VQA accuracy (test)", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.02, 0.97, "Rician K=6 dB", transform=ax.transAxes, fontsize=6.5,
            va="top")
    ax.legend(fontsize=6.2, loc="lower left", handlelength=1.4, labelspacing=0.25, borderpad=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"F5_pareto_energy.{ext}", dpi=300)
    plt.close(fig)

    # ---- F11: answers per joule vs SNR -------------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for m in ORDER:
        xs = [s for s in snrs if s in rows[m]]
        ys = [summary["per_method"][m][s]["answers_per_joule"] for s in xs]
        c, mk, lb, ls = STYLE[m]
        ax.plot(xs, ys, marker=mk, color=c, label=lb, lw=1.0, ms=3.2, ls=ls)
    # direct plateau labels (values readable without chasing the legend)
    apj = lambda m, s: summary["per_method"][m][s]["answers_per_joule"]
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
    ax.set_xlabel("SNR (dB)", fontsize=8)
    ax.set_ylabel("correct answers per joule", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.02, 0.97, f"Rician K=6 dB, $P_{{\\mathrm{{tx}}}}$={P_TX_HEAD} W",
            transform=ax.transAxes, fontsize=6.5, va="top")
    ax.legend(fontsize=6.2, loc="center right", handlelength=1.4, labelspacing=0.25, borderpad=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"F11_answers_per_joule.{ext}", dpi=300)
    plt.close(fig)

    # ---- LaTeX snippets ----------------------------------------------------
    def fmt(x: float) -> str:
        if x >= 100:
            return f"{x:.0f}"
        if x >= 10:
            return f"{x:.1f}"
        if x >= 1:
            return f"{x:.2f}"
        return f"{x:.3f}"

    key_snrs = [-5.0, 5.0, 20.0]
    lines = []
    for m in ORDER:
        cells = [STYLE[m][2].split(" (")[0]]
        for s in key_snrs:
            if s in summary["per_method"][m]:
                d = summary["per_method"][m][s]
                cells += [f"${d['acc']:.3f}$", f"${fmt(d['j_per_answer'])}$"]
            else:
                cells += ["--", "--"]
        lines.append(" & ".join(cells) + r" \\")
    (out_dir / "jpa_main_table.tex").write_text("\n".join(lines) + "\n")

    sens = []
    for p_tx in P_TX_GRID:
        row = [f"${p_tx:.2f}$"]
        for m in ("M3_token", "M4_adaptive", "M1_image"):
            e, _, _ = energy_j(m, rows[m][-5.0]["uses"], p_tx, e_vlm_inc)
            row.append(f"${fmt(e)}$")
        sens.append(" & ".join(row) + r" \\")
    (out_dir / "jpa_sensitivity.tex").write_text("\n".join(sens) + "\n")

    (out_dir / "energy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["params"], indent=2))
    for m in ORDER:
        s = -5.0
        d = summary["per_method"][m][s]
        print(f"{m:12s} @-5dB acc={d['acc']:.3f} Etx={d['e_tx_j']:.3f} "
              f"Ecmp={d['e_cmp_j']:.2f} J/ans={d['j_per_answer']:.2f} "
              f"ans/J={d['answers_per_joule']:.3f}")
    print(f"[done] wrote figures+tables under {out_dir}")


if __name__ == "__main__":
    main()
