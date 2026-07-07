#!/usr/bin/env python3
"""W3a: train the deployable per-sample quality predictor from v3_0 logs.

Pools the three fading-channel prediction CSVs (awgn/rayleigh/rician), splits
by image hash into fit / calibration / test folds, trains the calibrated
:class:`PersamplePredictor`, and reports:

  * per-channel + pooled test accuracy of the per-sample policy vs
    always-token / always-image / oracle,
  * ECE before/after temperature scaling (test fold),
  * decision agreement with the uncalibrated prototype recipe
    (scripts/build_persample_policy.py) trained per channel,
  * a reliability table CSV (calib + test folds).

Outputs:
  outputs/models/persample_predictor_v1.json
  outputs/models/persample_predictor_v1_reliability.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from vqa_semcom.quality.persample_predictor import (  # noqa: E402
    PersamplePredictor,
    expected_calibration_error,
    featurize,
    fit_logistic,
)

CHANNELS = ["awgn", "rayleigh", "rician"]
FEATURE_KEYS = ["question_type", "target_class", "view_quality_bin", "risk_level",
                "snr_bin", "raw_detector_count", "density_score", "presence_polarity"]


def _hash(image_id: str) -> int:
    return zlib.crc32(str(image_id).encode()) % 100


def fold_of(image_id: str) -> str:
    h = _hash(image_id)
    if h < 20:
        return "test"      # same 20% test protocol as build_comparison_v2.is_test
    if h < 36:
        return "calib"     # ~20% of the remaining train mass
    return "fit"


def _correct(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def load_groups(path: Path, channel: str) -> list[dict]:
    """One record per (image, question, snr) with s1/s2 labels + v1 features."""
    groups: dict[tuple, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("service_level") not in ("1", "2"):
                continue
            key = (r["image_id"], r["question"], r["snr_bin"])
            g = groups.setdefault(key, {"channel": channel, "image_id": r["image_id"],
                                        "question_type": r["question_type"],
                                        "snr_bin": r["snr_bin"], "bytes": {}})
            g[f's{r["service_level"]}'] = _correct(r["correct"])
            g["bytes"][r["service_level"]] = int(r.get("payload_bytes") or 0)
            if r["service_level"] == "1":  # s1 row carries transmitter-side features
                g["record"] = {k: r.get(k) for k in FEATURE_KEYS}
    return [g for g in groups.values() if "record" in g and "s1" in g and "s2" in g]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v3_0")
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--out", default="outputs/models/persample_predictor_v1.json")
    ap.add_argument("--reliability-out",
                    default="outputs/models/persample_predictor_v1_reliability.csv")
    args = ap.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    all_groups: list[dict] = []
    for ch in channels:
        path = Path(args.pred_dir) / f"{args.prefix}_{ch}_predictions.csv"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        groups = load_groups(path, ch)
        print(f"[{ch}] {len(groups)} grouped samples")
        all_groups.extend(groups)
    if not all_groups:
        raise SystemExit("no training data found")

    folds = [fold_of(g["image_id"]) for g in all_groups]
    records = [g["record"] for g in all_groups]
    labels = {"1": [g["s1"] for g in all_groups], "2": [g["s2"] for g in all_groups]}
    calib_mask = [f == "calib" for f in folds]
    train_mask = [f != "test" for f in folds]
    n_fit = sum(f == "fit" for f in folds)
    n_cal = sum(calib_mask)
    n_te = sum(f == "test" for f in folds)
    print(f"folds: fit={n_fit} calib={n_cal} test={n_te}")

    # Train the deployable pooled model (weights on fit fold, T on calib fold).
    fit_records = [r for r, f in zip(records, folds) if f != "test"]
    fit_labels = {s: [v for v, f in zip(vals, folds) if f != "test"]
                  for s, vals in labels.items()}
    fit_calib = [c for c, f in zip(calib_mask, folds) if f != "test"]
    model = PersamplePredictor.fit(
        fit_records, fit_labels, calib_mask=fit_calib,
        meta={"trained_on": channels, "prefix": args.prefix, "date": str(date.today()),
              "split": "crc32(image_id)%100: <20 test, 20-35 calib, >=36 fit",
              "n_test_holdout": n_te})
    for s in model.services:
        print(f"service {s}: temperature T={model.heads[s].temperature:.3f}")

    # ---- test-fold evaluation --------------------------------------------
    test_idx = [i for i, f in enumerate(folds) if f == "test"]
    test_records = [records[i] for i in test_idx]
    test_groups = [all_groups[i] for i in test_idx]
    sel = model.select(test_records, margin=args.margin)
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))  # method -> ch -> [k, n, bytes]
    for g, s in zip(test_groups, sel):
        for ch in ("pooled", g["channel"]):
            for m, hit, b in (
                ("always_token", g["s1"], g["bytes"].get("1", 0)),
                ("always_image", g["s2"], g["bytes"].get("2", 0)),
                ("ml_persample", g[f"s{s}"], g["bytes"].get(s, 0)),
                ("oracle", g["s1"] or g["s2"], 0.0),
            ):
                a = stats[m][ch]
                a[0] += int(hit); a[1] += 1; a[2] += b
    print("\ntest accuracy:")
    for m in ("always_token", "always_image", "ml_persample", "oracle"):
        row = "  ".join(f"{ch}={stats[m][ch][0]/max(stats[m][ch][1],1):.4f}"
                        for ch in ["pooled", *channels])
        k, n, b = stats[m]["pooled"]
        print(f"  {m:13s} {row}  mean_bytes={b/max(n,1):.0f}")

    # ECE before/after temperature scaling on the test fold.
    for s in model.services:
        y = [labels[s][i] for i in test_idx]
        p_raw = model.predict_proba(test_records, s, calibrated=False)
        p_cal = model.predict_proba(test_records, s, calibrated=True)
        print(f"service {s}: test ECE raw={expected_calibration_error(p_raw, y):.4f} "
              f"calibrated={expected_calibration_error(p_cal, y):.4f}")

    # ---- agreement with the per-channel prototype recipe -----------------
    agree_k = agree_n = 0
    for ch in channels:
        idx = [i for i, g in enumerate(all_groups) if g["channel"] == ch]
        tr = [i for i in idx if folds[i] != "test"]
        te = [i for i in idx if folds[i] == "test"]
        if not tr or not te:
            continue
        Xtr = [featurize(records[i]) for i in tr]
        w1 = fit_logistic(Xtr, [float(labels["1"][i]) for i in tr])
        w2 = fit_logistic(Xtr, [float(labels["2"][i]) for i in tr])
        Xte = np.hstack([np.asarray([featurize(records[i]) for i in te]),
                         np.ones((len(te), 1))])
        p1 = 1 / (1 + np.exp(-Xte @ w1))
        p2 = 1 / (1 + np.exp(-Xte @ w2))
        proto_sel = ["1" if (b - a) <= args.margin else "2" for a, b in zip(p1, p2)]
        pooled_sel = model.select([records[i] for i in te], margin=args.margin)
        agree = sum(int(a == b) for a, b in zip(proto_sel, pooled_sel))
        agree_k += agree; agree_n += len(te)
        print(f"[{ch}] pooled-model vs per-channel-prototype decision agreement: "
              f"{agree/len(te):.4f} (n={len(te)})")
    if agree_n:
        print(f"overall agreement: {agree_k/agree_n:.4f} (n={agree_n})")

    # ---- artefacts --------------------------------------------------------
    model.save(args.out)
    print(f"wrote model -> {args.out}")
    rel_rows = []
    for s in model.services:
        for lo, hi, mp, emp, n in model.heads[s].reliability:
            rel_rows.append([s, "calib", round(lo, 2), round(hi, 2),
                             round(mp, 4), round(emp, 4), int(n)])
        y = np.asarray([1.0 if labels[s][i] else 0.0 for i in test_idx])
        p = model.predict_proba(test_records, s, calibrated=True)
        from vqa_semcom.quality.persample_predictor import _reliability_table
        for lo, hi, mp, emp, n in _reliability_table(p, y):
            rel_rows.append([s, "test", round(lo, 2), round(hi, 2),
                             round(mp, 4), round(emp, 4), int(n)])
    os.makedirs(os.path.dirname(args.reliability_out), exist_ok=True)
    with open(args.reliability_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["service", "fold", "bin_lo", "bin_hi", "mean_predicted",
                    "empirical_accuracy", "n"])
        w.writerows(rel_rows)
    print(f"wrote reliability table -> {args.reliability_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
