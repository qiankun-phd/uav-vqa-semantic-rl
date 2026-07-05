#!/usr/bin/env python3
"""Merge the DroneVehicle main/cmp/extra rician prediction CSVs into a single
outputs/vlm/dv_rician_predictions.csv, which is the {prefix}_{channel} name that
build_comparison_v2.py --prefix dv expects. CSV-aware concat (prediction rows
contain multi-line quoted prompt fields, so a naive line concat would corrupt
them)."""
import csv
import os
import sys

SRCS = [
    "outputs/vlm/dv_rician_main_predictions.csv",
    "outputs/vlm/dv_rician_cmp_predictions.csv",
    "outputs/vlm/dv_rician_extra_predictions.csv",
]
OUT = "outputs/vlm/dv_rician_predictions.csv"

fieldnames = None
rows = []
n_by_src = {}
for src in SRCS:
    if not os.path.exists(src):
        print(f"skip missing {src}", file=sys.stderr)
        continue
    with open(src, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = r.fieldnames
        cnt = 0
        for row in r:
            rows.append(row)
            cnt += 1
        n_by_src[src] = cnt

if fieldnames is None:
    raise SystemExit("no source prediction files found")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

for s, c in n_by_src.items():
    print(f"  {s}: {c} rows")
print(f"merged {len(rows)} rows -> {OUT}")
