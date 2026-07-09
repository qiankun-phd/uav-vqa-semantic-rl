#!/usr/bin/env python3
"""P1-2 analysis: M6 (compact DeepJSCC) results in the paper's caliber.

Reads outputs/vlm/m6_djscc_rician_predictions.csv (test-only run), computes
per-SNR accuracy (all five question types + the presence/counting subset that
M2/M0_naive cover), Wilson bounds, mean PSNR and complex channel uses per SNR,
and places them next to the M1/M2/M3/M4 Rician rows of
outputs/reports/comparison_v3_5qt.csv.

Outputs: outputs/reports/p1_m6_results.json / .md
         outputs/reports/p1_m6_tidy_rows.csv (same schema as comparison_v3_5qt)
"""
from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SNRS = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def is_test(iid: str) -> bool:
    return (zlib.crc32(str(iid).encode()) % 100) < 20


def main() -> int:
    df = pd.read_csv(REPO / "outputs/vlm/m6_djscc_rician_predictions.csv",
                     dtype={"image_id": str})
    df = df[df.image_id.map(is_test)].drop_duplicates(["image_id", "question", "snr_db"])
    df["ok"] = df["correct"].map(lambda v: 1 if str(v).strip().lower() == "true" else 0)
    df["snr_db"] = df["snr_db"].astype(float)
    print(f"M6 cells: {len(df)}; qtypes {df.question_type.value_counts().to_dict()}")

    report = {"per_snr": {}, "per_snr_2type": {}, "pooled": {}, "by_qtype": {}}
    tidy_rows = []
    for s in SNRS:
        sub = df[df.snr_db == s]
        p, lo, hi = wilson(int(sub.ok.sum()), len(sub))
        img = sub.drop_duplicates(["image_id"])
        psnr = float(img.psnr_db.mean())
        uses = float(img.channel_uses.mean())
        report["per_snr"][f"{s:g}"] = {"acc": round(p, 4), "lo": round(lo, 4),
                                       "hi": round(hi, 4), "n": len(sub),
                                       "mean_psnr_db": round(psnr, 2),
                                       "mean_channel_uses": round(uses, 0)}
        tidy_rows.append({"channel": "rician", "method": "M6_djscc", "snr_db": s,
                          "qtype": "all", "split": "test", "accuracy": round(p, 4),
                          "n": len(sub), "lcb": round(lo, 4), "ucb": round(hi, 4),
                          "mean_payload_bytes": round(uses, 1),
                          "mean_channel_uses": round(uses, 1), "cbr": ""})
        sub2 = sub[sub.question_type.isin(["presence", "counting"])]
        p2, lo2, hi2 = wilson(int(sub2.ok.sum()), len(sub2))
        report["per_snr_2type"][f"{s:g}"] = {"acc": round(p2, 4), "n": len(sub2)}
    p, lo, hi = wilson(int(df.ok.sum()), len(df))
    report["pooled"] = {"acc": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(df)}
    for qt, g in df.groupby("question_type"):
        pq, loq, hiq = wilson(int(g.ok.sum()), len(g))
        report["by_qtype"][qt] = {"acc": round(pq, 4), "n": len(g)}

    tidy = pd.read_csv(REPO / "outputs/reports/comparison_v3_5qt.csv")
    ref = tidy[(tidy.channel == "rician") & (tidy.qtype == "all") & (tidy.split == "test")
               & tidy.method.isin(["M1_image", "M2_analog", "M3_token", "M4_adaptive",
                                    "M0_naive", "M5_oracle"])]
    report["reference_rician"] = {
        m: {f"{r.snr_db:g}": round(r.accuracy, 4) for r in g.itertuples()}
        for m, g in ref.groupby("method")
    }

    out = REPO / "outputs/reports/p1_m6_results.json"
    out.write_text(json.dumps(report, indent=2))
    pd.DataFrame(tidy_rows).to_csv(REPO / "outputs/reports/p1_m6_tidy_rows.csv", index=False)

    lines = ["# P1-2: M6 compact DeepJSCC results (Rician, test)",
             "", "| SNR (dB) | " + " | ".join(f"{s:g}" for s in SNRS) + " |",
             "|---|" + "---|" * len(SNRS)]
    lines.append("| M6 acc (5-type) | " + " | ".join(
        f"{report['per_snr'][f'{s:g}']['acc']:.3f}" for s in SNRS) + " |")
    lines.append("| M6 acc (pres+count) | " + " | ".join(
        f"{report['per_snr_2type'][f'{s:g}']['acc']:.3f}" for s in SNRS) + " |")
    lines.append("| M6 PSNR (dB) | " + " | ".join(
        f"{report['per_snr'][f'{s:g}']['mean_psnr_db']:.1f}" for s in SNRS) + " |")
    for m in ("M1_image", "M2_analog", "M3_token", "M4_adaptive"):
        if m in report["reference_rician"]:
            lines.append(f"| {m} | " + " | ".join(
                f"{report['reference_rician'][m].get(f'{s:g}', float('nan')):.3f}"
                for s in SNRS) + " |")
    lines.append("")
    lines.append(f"pooled M6: {report['pooled']}")
    lines.append(f"by qtype: {report['by_qtype']}")
    lines.append(f"mean channel uses: "
                 + ", ".join(f"{s:g}dB={report['per_snr'][f'{s:g}']['mean_channel_uses']:.0f}"
                             for s in SNRS))
    md = "\n".join(lines) + "\n"
    (REPO / "outputs/reports/p1_m6_results.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
