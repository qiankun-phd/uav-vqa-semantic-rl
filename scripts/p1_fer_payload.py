#!/usr/bin/env python3
"""P1-5 (log/model-level, zero GPU): (a) residual frame-loss curves that explain
the token path's flatness vs the fixed-rate cliff; (b) the M1 rate-adaptive
payload-vs-SNR table that de-black-boxes the mean payload numbers.

(a) FER(gamma):
    - token path (rate-adaptive): information-outage probability of the small
      token payload (design unit 256 B and the measured mean s1 payload),
      computed with the exact code-path functions (outage_probability);
    - fixed-rate path: the *measured* post-decode LDPC frame-error rate from the
      campaign's calibration caches (outputs/lut/link_calibration_v2_0_*_naive.json).
(b) payload(SNR): per-SNR byte budget (B*tau*SE) and the actually fitted JPEG
    bytes + outage rate over the test images, same _fit_jpeg_to_budget code path.

Outputs: outputs/reports/p1_fer_payload.json / .md and
         outputs/figures/paper1_final/F10_fer_payload.pdf
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.degradation.digital_link import (
    FadingConfig, build_link_config, outage_probability, rate_budget_bytes,
    _fit_jpeg_to_budget,
)
from vqa_semcom.evidence.builder import image_path_for_task, read_tasks_csv, select_vlm_tasks

SNR_BINS = [-5, 0, 5, 10, 15, 20]
SNR_GRID = np.arange(-10.0, 22.5, 0.5)
CHANNELS = {"awgn": FadingConfig("awgn", 6.0), "rayleigh": FadingConfig("rayleigh", 6.0),
            "rician": FadingConfig("rician", 6.0)}


def is_test(iid: str) -> bool:
    return (zlib.crc32(str(iid).encode()) % 100) < 20


def token_outage_curve(payload_bytes: float, link_cfg, fading: FadingConfig) -> list[float]:
    r_min = (payload_bytes * 8.0 / link_cfg.tx_time_budget_s) / link_cfg.bandwidth_hz
    return [outage_probability(float(s), fading, r_min) for s in SNR_GRID]


def main() -> int:
    cfg = load_config("configs/v2_0_ldpc_channel.yaml")
    link_cfg = build_link_config(cfg)

    # measured mean s1 payload from the campaign log (test split)
    df = pd.read_csv(resolve_path("outputs/vlm/v3_0_rician_predictions.csv"),
                     dtype={"image_id": str},
                     usecols=["image_id", "service_level", "payload_bytes", "snr_bin", "question"])
    s1 = df[(df.service_level == 1) & df.image_id.map(is_test)]
    mean_s1_payload = float(s1.payload_bytes.mean())

    report: dict = {"token_payload_design_B": 256, "token_payload_measured_mean_B": round(mean_s1_payload, 1),
                    "bandwidth_hz": link_cfg.bandwidth_hz, "tx_time_budget_s": link_cfg.tx_time_budget_s,
                    "min_image_payload_B": link_cfg.min_payload_bytes}

    # (a) token outage curves + fixed-rate LDPC FER from calibration caches
    curves: dict = {"snr_grid": [float(s) for s in SNR_GRID]}
    for name, fad in CHANNELS.items():
        curves[f"token256_{name}"] = token_outage_curve(256.0, link_cfg, fad)
        curves[f"tokenMeas_{name}"] = token_outage_curve(mean_s1_payload, link_cfg, fad)
        curves[f"imageMin_{name}"] = token_outage_curve(float(link_cfg.min_payload_bytes), link_cfg, fad)
    ldpc_fer: dict = {}
    for name in CHANNELS:
        p = Path(resolve_path(f"outputs/lut/link_calibration_v2_0_{name}_naive.json"))
        if p.exists():
            calib = json.loads(p.read_text())["calib"]
            ldpc_fer[name] = {k: v["fer"] for k, v in sorted(calib.items(), key=lambda kv: kv[1]["snr_db"])}
    report["ldpc_fixed_rate_fer_measured"] = ldpc_fer
    report["token_outage_at_bins"] = {
        name: {f"{s}dB": float(token_outage_curve(mean_s1_payload, link_cfg, fad)[list(SNR_GRID).index(float(s))])
               for s in SNR_BINS} for name, fad in CHANNELS.items()}

    # (b) M1 payload vs SNR: budget + actual fitted bytes over test images
    import cv2

    vlm_cfg = cfg.get("vlm", {})
    tasks = read_tasks_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    sel = select_vlm_tasks(tasks, question_types=set(vlm_cfg.get("question_types", ["presence", "counting"])),
                           max_tasks_per_image=int(vlm_cfg.get("max_tasks_per_image", 6)))
    ids = sorted({t["image_id"] for t in sel if is_test(t["image_id"])})
    root = resolve_path(cfg["paths"]["visdrone_val"])
    payload: dict = {}
    for name, fad in CHANNELS.items():
        lc = link_cfg.__class__(fading=fad, ldpc=link_cfg.ldpc, modulation=link_cfg.modulation,
                                packet_payload_bits=link_cfg.packet_payload_bits,
                                calib_blocks=link_cfg.calib_blocks, image_grid=link_cfg.image_grid,
                                channel_mode=link_cfg.channel_mode, bandwidth_hz=link_cfg.bandwidth_hz,
                                tx_time_budget_s=link_cfg.tx_time_budget_s,
                                min_payload_bytes=link_cfg.min_payload_bytes)
        per_snr = {}
        for s in SNR_BINS:
            budget = rate_budget_bytes(float(s), lc)
            fitted = []
            r_min = (lc.min_payload_bytes * 8.0 / lc.tx_time_budget_s) / lc.bandwidth_hz
            p_out = outage_probability(float(s), fad, r_min)
            for iid in ids:
                img = cv2.imread(str(image_path_for_task(Path(root), iid)))
                if img is None:
                    continue
                _dec, meta = _fit_jpeg_to_budget(img, budget)
                fitted.append(meta["bytes"])
            per_snr[f"{s}dB"] = {
                "budget_B": round(budget, 1),
                "fitted_mean_B": round(float(np.mean(fitted)), 1),
                "fitted_p10_B": round(float(np.percentile(fitted, 10)), 1),
                "fitted_p90_B": round(float(np.percentile(fitted, 90)), 1),
                "outage_prob": round(p_out, 5),
                "n_images": len(fitted),
            }
            print(f"[{name} {s}dB] budget={budget/1e3:.1f}KB fitted={np.mean(fitted)/1e3:.1f}KB p_out={p_out:.4f}",
                  flush=True)
        payload[name] = per_snr
    report["m1_payload_vs_snr"] = payload

    out_json = Path(resolve_path("outputs/reports/p1_fer_payload.json"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"report": report, "curves": curves}, indent=2))

    # markdown table (rician, the paper's default channel)
    lines = ["# P1-5: residual frame loss + M1 payload vs SNR", "",
             f"token payload: design 256 B, measured mean {mean_s1_payload:.0f} B", "",
             "## M1 rate-adaptive payload (Rician)", "",
             "| SNR | budget (KB) | fitted mean (KB) | outage |", "|---|---|---|---|"]
    for s in SNR_BINS:
        r = payload["rician"][f"{s}dB"]
        lines.append(f"| {s} dB | {r['budget_B']/1e3:.1f} | {r['fitted_mean_B']/1e3:.1f} | {r['outage_prob']:.4f} |")
    lines += ["", "## token outage at bins (measured payload)", ""]
    for name in CHANNELS:
        lines.append(f"- {name}: " + ", ".join(f"{k}={v:.2e}" for k, v in report["token_outage_at_bins"][name].items()))
    lines += ["", "## fixed-rate LDPC FER (measured calibration)", ""]
    for name, d in ldpc_fer.items():
        lines.append(f"- {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in d.items()))
    out_md = Path(resolve_path("outputs/reports/p1_fer_payload.md"))
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure: two panels (FER curves; payload bars)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                         "lines.linewidth": 1.2, "legend.fontsize": 6.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax = axes[0]
    colors = {"awgn": "#1f77b4", "rayleigh": "#d62728", "rician": "#2ca02c"}
    for name in CHANNELS:
        ax.semilogy(SNR_GRID, np.maximum(curves[f"tokenMeas_{name}"], 1e-7), color=colors[name],
                    label=f"token ({name})")
        if name in ldpc_fer:
            xs = [float(k.replace("dB", "")) for k in ldpc_fer[name]]
            ys = np.maximum([ldpc_fer[name][k] for k in ldpc_fer[name]], 1e-7)
            ax.semilogy(xs, ys, color=colors[name], linestyle="--", marker="o", markersize=2.5,
                        label=f"fixed-rate LDPC ({name})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("residual frame-loss prob.")
    ax.set_ylim(1e-7, 1.5)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(ncol=1, frameon=False, loc="lower left")
    ax2 = axes[1]
    xs = np.arange(len(SNR_BINS))
    for i, name in enumerate(CHANNELS):
        vals = [payload[name][f"{s}dB"]["fitted_mean_B"] / 1e3 for s in SNR_BINS]
        ax2.bar(xs + (i - 1) * 0.27, vals, width=0.25, color=colors[name], label=name)
    ax2.set_xticks(xs, [f"{s}" for s in SNR_BINS])
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("M1 delivered payload (KB)")
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    ax2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_fig = Path(resolve_path("outputs/figures/paper1_final/F10_fer_payload.pdf"))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    fig.savefig(out_fig.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"figure -> {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
