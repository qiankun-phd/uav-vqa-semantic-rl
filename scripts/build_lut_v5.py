#!/usr/bin/env python3
"""Task #28 v5: rebuild the ONE unified semantic-quality LUT (single source of truth).

Reads the v3_0 three-channel campaign per-sample prediction logs and rebuilds a
single quality table keyed by

    question_type x service_level x channel x snr_bin x view_quality_bin x count_bucket

with risk_level DROPPED from the quality axis (risk never belonged in a quality
table -- it is a task attribute, not a channel/decoder property).  Counting rows
are RE-JUDGED offline from the stored (normalized_prediction, object_count) with
the v5 tolerance rule:

    counting correct  <=>  |pred - gt| <= max(1, round(r * gt))
    r = 0.20  if gt >= 10   (critical counting bucket -- v5 widens +-10% -> +-20%)
    r = 0.10  otherwise     (normal counting unchanged)

Presence / other qtypes keep the stored `correct` column (yes/no judging is
tolerance-independent).  Counting count_bucket in {1-4, 5-9, 10-19, 20-49, 50+};
presence and other qtypes have no count dimension (count_bucket = "na").

Sparse cells (n < MIN_SAMPLES=30) INHERIT the pooled parent-cell Wilson interval
(parent = same qtype x service x channel x snr, pooled over view x count_bucket).
s0 (service_level 0 = cache) rows are emitted but flagged
`simulator_derived=1` -- they are NOT used at runtime (runtime s0 quality is
entry-driven under cache_quality=entry_v2; see EPSILON_RECAL_V5.md change 3).

Output: outputs/lut/v5_unified_lut.csv  (read identically by the calibrator and
the RL record layer -- the single source of truth).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

CHANNELS = ["awgn", "rayleigh", "rician"]
MIN_SAMPLES = 30
Z = 1.96
COUNT_BUCKETS = [(1, 4, "1-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, 49, "20-49"), (50, 10**9, "50+")]


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def count_bucket(gt: int) -> str:
    for lo, hi, label in COUNT_BUCKETS:
        if lo <= gt <= hi:
            return label
    if gt <= 0:
        return "0"
    return "50+"


def _int_or_none(text: str) -> int | None:
    s = str(text).strip().lower()
    if not s or s in ("unknown", "none", "nan", ""):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        # normalized_prediction for counting is already a stringified int, but be safe.
        import re
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else None


def rejudge_counting(pred_text: str, gt: int) -> bool | None:
    """v5 offline re-judge; returns None if the prediction is unparseable."""
    pred = _int_or_none(pred_text)
    if pred is None:
        return None
    r = 0.20 if gt >= 10 else 0.10
    tol = max(1, round(r * gt))
    return abs(pred - gt) <= tol


def _correct_flag(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v3_0")
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--out", default="outputs/lut/v5_unified_lut.csv")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    args = ap.parse_args()

    # cell -> [k, n, payload_sum]
    cells: dict[tuple, list[float]] = defaultdict(lambda: [0, 0, 0.0])
    # parent (qtype, service, channel, snr) -> [k, n]  (pooled over view x bucket)
    parents: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    total = 0
    unparseable_counting = 0
    flips = 0  # counting rows whose judgement changed under the v5 tolerance
    ge10 = 0
    ge10_correct_v5 = 0
    ge10_correct_legacy = 0

    for ch in [c.strip() for c in args.channels.split(",") if c.strip()]:
        path = Path(args.pred_dir) / f"{args.prefix}_{ch}_predictions.csv"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    level = int(r["service_level"])
                except (KeyError, ValueError):
                    continue
                qtype = r.get("question_type", "")
                channel = r.get("channel_bin", ch) or ch
                snr = r.get("snr_bin", "")
                view = r.get("view_quality_bin", "")
                legacy_correct = _correct_flag(r.get("correct", ""))
                if qtype == "counting":
                    gt = _int_or_none(r.get("object_count", "")) or 0
                    bucket = count_bucket(gt)
                    verdict = rejudge_counting(r.get("normalized_prediction", ""), gt)
                    if verdict is None:
                        unparseable_counting += 1
                        correct = False
                    else:
                        correct = verdict
                    if correct != legacy_correct:
                        flips += 1
                    if gt >= 10:
                        ge10 += 1
                        ge10_correct_v5 += int(correct)
                        ge10_correct_legacy += int(legacy_correct)
                else:
                    bucket = "na"
                    correct = legacy_correct
                key = (qtype, level, channel, snr, view, bucket)
                cell = cells[key]
                cell[0] += int(correct)
                cell[1] += 1
                cell[2] += float(r.get("payload_bytes") or 0)
                pkey = (qtype, level, channel, snr)
                parents[pkey][0] += int(correct)
                parents[pkey][1] += 1
                total += 1
        print(f"[{ch}] merged (running total {total} rows, {len(cells)} cells)")

    parent_wilson = {pk: wilson(k, n) for pk, (k, n) in parents.items()}

    rows = []
    n_inherited = 0
    for key in sorted(cells.keys()):
        qtype, level, channel, snr, view, bucket = key
        k, n, byte_sum = cells[key]
        n = int(n)
        k = int(k)
        p, lo, hi = wilson(k, n)
        inherited = 0
        if n < args.min_samples:
            pk = (qtype, level, channel, snr)
            pp, plo, phi = parent_wilson.get(pk, (p, lo, hi))
            # inherit the pooled parent interval (keeps the honest sparse-cell tax)
            p, lo, hi = pp, plo, phi
            inherited = 1
            n_inherited += 1
        rows.append({
            "question_type": qtype,
            "service_level": level,
            "channel_bin": channel,
            "snr_bin": snr,
            "view_quality_bin": view,
            "count_bucket": bucket,
            "sample_count": n,
            "expected_accuracy": round(p, 6),
            "wilson_low": round(lo, 6),
            "wilson_high": round(hi, 6),
            "avg_payload_bytes": round(byte_sum / max(n, 1), 1),
            "inherited": inherited,
            "simulator_derived": int(level == 0),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    counts = [r["sample_count"] for r in rows]
    print(f"wrote {len(rows)} cells -> {args.out}")
    print(f"  total rows merged           : {total}")
    print(f"  cells                       : {len(rows)}")
    print(f"  mean samples/cell           : {sum(counts)/max(len(counts),1):.1f}  min={min(counts)} max={max(counts)}")
    print(f"  cells n<{args.min_samples} inheriting parent : {n_inherited} ({100.0*n_inherited/max(len(rows),1):.1f}%)")
    print(f"  s0 (cache) simulator-derived: {sum(r['simulator_derived'] for r in rows)} cells")
    print(f"  counting unparseable pred   : {unparseable_counting}")
    print(f"  counting judgement flips    : {flips} (legacy +-10% -> v5 tolerance)")
    if ge10:
        print(f"  GT>=10 counting rows        : {ge10}")
        print(f"    accuracy legacy(+-10%)    : {ge10_correct_legacy/ge10:.4f}")
        print(f"    accuracy v5(+-20%)        : {ge10_correct_v5/ge10:.4f}  (delta +{(ge10_correct_v5-ge10_correct_legacy)/ge10:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
