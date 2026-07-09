#!/usr/bin/env python3
"""P1-3 analysis: VLM-reads-tokens vs calibrated symbolic decoder vs
VLM-reads-image, per question type and SNR, on identical (image, question,
snr) cells (Rician test split, SNR in {-5,5,20} dB).

Inputs:
  outputs/vlm/p1_vlm_reads_tokens_rician_predictions.csv  (from p1_vlm_reads_tokens.py)
  outputs/vlm/v3_0_rician_predictions.csv                 (campaign log: s1 symbolic + s2 image)
Outputs:
  outputs/reports/p1_vlm_reads_tokens.json / .md
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "outputs/reports/p1_vlm_reads_tokens.json"
OUT_MD = REPO / "outputs/reports/p1_vlm_reads_tokens.md"
QTYPES = ["presence", "counting", "comparison", "co_presence", "threshold"]
SNRS = ["-5dB", "5dB", "20dB"]


def is_test(iid: str) -> bool:
    return (zlib.crc32(str(iid).encode()) % 100) < 20


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (b, c)."""
    from scipy.stats import binom

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * binom.cdf(k, n, 0.5)
    if b == c:
        p -= binom.pmf(k, n, 0.5)
    return min(1.0, float(p))


def tf(v) -> int:
    return 1 if str(v).strip().lower() == "true" else 0


def main() -> int:
    vt = pd.read_csv(REPO / "outputs/vlm/p1_vlm_reads_tokens_rician_predictions.csv",
                     dtype={"image_id": str})
    vt["vlm_token_ok"] = vt["correct"].map(tf)
    vt["symbolic_ok"] = vt["symbolic_correct"].map(tf)
    vt["raw_symbolic_ok"] = vt["raw_decoder_correct"].map(
        lambda v: 1 if str(v).strip().lower() == "true" else (0 if str(v).strip().lower() == "false" else np.nan))
    # counting rows carry a real raw-decoder verdict; other types equal the rule verdict
    vt.loc[vt.raw_symbolic_ok.isna(), "raw_symbolic_ok"] = vt.loc[vt.raw_symbolic_ok.isna(), "symbolic_ok"]

    camp = pd.read_csv(REPO / "outputs/vlm/v3_0_rician_predictions.csv", dtype={"image_id": str},
                       usecols=["image_id", "question", "question_type", "snr_bin",
                                "service_level", "correct"])
    s2 = camp[(camp.service_level == 2) & camp.snr_bin.isin(SNRS)]
    s2 = s2[s2.image_id.map(is_test)].drop_duplicates(["image_id", "question", "snr_bin"])
    s2 = s2.rename(columns={"correct": "s2_correct"})[
        ["image_id", "question", "snr_bin", "s2_correct"]]
    s2["vlm_image_ok"] = s2["s2_correct"].map(lambda v: tf(v) if isinstance(v, str) else int(bool(v)))

    df = vt.merge(s2[["image_id", "question", "snr_bin", "vlm_image_ok"]],
                  on=["image_id", "question", "snr_bin"], how="left", validate="1:1")
    print(f"cells: {len(df)}; missing image pair: {int(df.vlm_image_ok.isna().sum())}")

    def block(sub: pd.DataFrame) -> dict:
        n = len(sub)
        out = {"n": int(n)}
        for col, name in (("vlm_token_ok", "vlm_reads_tokens"), ("symbolic_ok", "symbolic_decoder"),
                          ("raw_symbolic_ok", "raw_symbolic"), ("vlm_image_ok", "vlm_reads_image")):
            v = sub[col].dropna()
            p, lo, hi = wilson(int(v.sum()), len(v))
            out[name] = {"acc": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        b = int(((sub.vlm_token_ok == 1) & (sub.symbolic_ok == 0)).sum())
        c = int(((sub.vlm_token_ok == 0) & (sub.symbolic_ok == 1)).sum())
        out["mcnemar_vlmtok_vs_symbolic"] = {"b_vlm_only": b, "c_sym_only": c,
                                             "p": round(mcnemar_exact(b, c), 6)}
        return out

    report: dict = {"scope": "Rician test split, SNR {-5,5,20} dB, per unique (image,question,snr)"}
    report["pooled"] = block(df)
    report["by_qtype"] = {qt: block(df[df.question_type == qt]) for qt in QTYPES}
    report["by_snr"] = {s: block(df[df.snr_bin == s]) for s in SNRS}
    report["by_qtype_snr"] = {qt: {s: block(df[(df.question_type == qt) & (df.snr_bin == s)])
                                   for s in SNRS} for qt in QTYPES}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))

    lines = ["# P1-3: VLM-reads-tokens vs symbolic decoder vs VLM-reads-image",
             "", f"Scope: {report['scope']}", "",
             "| slice | n | VLM reads tokens | symbolic decoder | raw symbolic | VLM reads image | McNemar p (vlm-tok vs sym) |",
             "|---|---|---|---|---|---|---|"]

    def row(label: str, blk: dict) -> str:
        f = lambda k: f"{blk[k]['acc']:.3f} [{blk[k]['lo']:.3f},{blk[k]['hi']:.3f}]"
        mc = blk["mcnemar_vlmtok_vs_symbolic"]
        return (f"| {label} | {blk['n']} | {f('vlm_reads_tokens')} | {f('symbolic_decoder')} | "
                f"{f('raw_symbolic')} | {f('vlm_reads_image')} | {mc['p']:g} (b={mc['b_vlm_only']},c={mc['c_sym_only']}) |")

    lines.append(row("pooled", report["pooled"]))
    for qt in QTYPES:
        lines.append(row(qt, report["by_qtype"][qt]))
    for s in SNRS:
        lines.append(row(f"SNR {s}", report["by_snr"][s]))
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
