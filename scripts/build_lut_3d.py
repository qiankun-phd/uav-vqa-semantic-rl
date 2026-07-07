#!/usr/bin/env python3
"""W3b: collapse the semantic-quality LUT to three dimensions.

Groups the v3_0 per-sample prediction logs by (question_type, service_level,
snr_bin) only -- dropping the view/freshness/risk axes whose 648-cell grid
left most cells with single-digit sample counts -- and writes a dense 3-D LUT
with honest Wilson confidence intervals:

    outputs/lut/v2_0_lut_3d.csv

Also prints the cell-count / mean-samples-per-cell comparison against the
legacy 6-D LUT (outputs/lut/v1_9_snr_semantic_quality_lut.csv).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

CHANNELS = ["awgn", "rayleigh", "rician"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def _correct(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _snr_db(label: str) -> float:
    return float(str(label).lower().replace("db", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v3_0")
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--legacy-lut", default="outputs/lut/v1_9_snr_semantic_quality_lut.csv")
    ap.add_argument("--out", default="outputs/lut/v2_0_lut_3d.csv")
    args = ap.parse_args()

    cells: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0, 0, 0.0])
    total_rows = 0
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
                key = (r["question_type"], level, r["snr_bin"])
                cell = cells[key]
                cell[0] += int(_correct(r["correct"]))
                cell[1] += 1
                cell[2] += float(r.get("payload_bytes") or 0)
                total_rows += 1
        print(f"[{ch}] merged (running total {total_rows} rows, {len(cells)} cells)")

    rows = []
    for (qtype, level, snr_bin), (k, n, byte_sum) in sorted(
            cells.items(), key=lambda kv: (kv[0][0], kv[0][1], _snr_db(kv[0][2]))):
        p, lo, hi = wilson(int(k), int(n))
        rows.append({
            "question_type": qtype,
            "service_level": level,
            "snr_bin": snr_bin,
            "sample_count": int(n),
            "expected_accuracy": round(p, 6),
            "wilson_low": round(lo, 6),
            "wilson_high": round(hi, 6),
            "avg_payload_bytes": round(byte_sum / max(n, 1), 1),
            "avg_payload_kb": round(byte_sum / max(n, 1) / 1024.0, 3),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} cells -> {args.out}")

    mean_n_3d = sum(r["sample_count"] for r in rows) / max(len(rows), 1)
    print(f"3-D grid: {len(rows)} cells, mean samples/cell = {mean_n_3d:.1f}, "
          f"min = {min(r['sample_count'] for r in rows)}")
    legacy = Path(args.legacy_lut)
    if legacy.exists():
        with legacy.open(newline="", encoding="utf-8") as f:
            legacy_rows = list(csv.DictReader(f))
        counts = [int(float(r.get("sample_count") or 0)) for r in legacy_rows]
        mean_n_6d = sum(counts) / max(len(counts), 1)
        print(f"legacy 6-D grid ({legacy.name}): {len(legacy_rows)} cells, "
              f"mean samples/cell = {mean_n_6d:.1f}, min = {min(counts) if counts else 0}")
        print(f"cell reduction: {len(legacy_rows)} -> {len(rows)} "
              f"({len(legacy_rows) / max(len(rows), 1):.1f}x fewer), "
              f"samples/cell gain: {mean_n_6d:.1f} -> {mean_n_3d:.1f} "
              f"({mean_n_3d / max(mean_n_6d, 1e-9):.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
