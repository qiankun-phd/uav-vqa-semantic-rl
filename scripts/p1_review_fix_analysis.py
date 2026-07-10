#!/usr/bin/env python3
"""Review-fix analysis for paper-1 TGCN (CPU/log-level only; no GPU).

Covers four reviewer items, all recomputed from existing prediction logs:

  M11  Disjoint-fit sensitivity of the counting calibration scale:
       refit the per-(class, SNR-bin) clipped ratio on the 80% train split
       only, re-decode counting answers on the 20% crc32 hold-out, and
       report the counting Delta = acc(token) - acc(image) under the
       pooled fit (as published) vs the disjoint fit.
  M6   The rate-adaptive-image 5 dB Rician (0.623) > error-free (0.611)
       inversion: paired per-type deltas + McNemar + JPEG working points.
  M7   Image/routing J-per-answer ratio at every SNR on all three channels
       (energy model identical to make_energy_figures.py).
  C1   Energy-controllable routing: per-sample logistic predictors
       P(correct | x, level) (same recipe as build_persample_policy.py),
       swept over a Lagrangian energy price to trace the energy-accuracy
       frontier between 0.435 J (all-token) and ~32 J (all-image).

Outputs: outputs/energy/review_fix_analysis.json (all numbers) and
         outputs/energy/c1_frontier_<channel>.csv (frontier points).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CHANNELS = ["awgn", "rayleigh", "rician"]
PRED = {ch: REPO / f"outputs/vlm/v3_0_{ch}_predictions.csv" for ch in CHANNELS}
CLEAN = REPO / "outputs/vlm/v3_0_clean_predictions.csv"
POWER_JSON = REPO / "outputs/energy/gpu_power_phases.json"
TIDY_CSV = REPO / "outputs/reports/comparison_v3_5qt.csv"
OUT_DIR = REPO / "outputs/energy"

BANDWIDTH_HZ = 1.0e6
P_TX_HEAD = 0.5
E_DET_MID_J = 0.5 * (7.0 * 0.015 + 15.0 * 0.050)  # Jetson band midpoint
F_IMG_M4 = 410.0 / 936.0


def is_test(image_id: str, frac: float = 0.2) -> bool:
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def snr_val(snr_bin: str) -> float:
    return float(str(snr_bin).replace("dB", ""))


def ok(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def count_correct(pred_count: int, gt_count: int, tol_ratio: float = 0.10) -> bool:
    tol = max(1, round(tol_ratio * gt_count))
    return abs(pred_count - gt_count) <= tol


def calibrated_count(raw: int, ratio: float, min_raw: int = 3) -> int:
    if raw <= 0:
        return 0
    if raw < min_raw:
        return raw
    return max(0, int(round(raw * ratio)))


def fit_ratios(rows, use_row) -> dict:
    """Replicates _counting_class_channel_calibration on log rows.

    rows: counting service-1 rows; use_row: predicate selecting the fit set.
    Returns {(target_class, snr_bin): accepted_ratio}."""
    sums = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        if not use_row(r):
            continue
        key = (r["target_class"], r["snr_bin"])
        sums[key][0] += float(r["gt"])
        sums[key][1] += float(r["det"])
    ratios = {}
    for key, (gs, ds) in sums.items():
        ratios[key] = 1.0 if ds <= 0 else max(0.5, min(4.0, gs / ds))
    # acceptance step: keep the ratio only if it does not lower the summed
    # correctness on the fit set (same tie-break as the eval script).
    accepted = {}
    by_key = defaultdict(list)
    for r in rows:
        if use_row(r):
            by_key[(r["target_class"], r["snr_bin"])].append(r)
    for key, ratio in ratios.items():
        raw_s = sum(count_correct(r["det"], r["gt"]) for r in by_key[key])
        cal_s = sum(count_correct(calibrated_count(r["det"], ratio), r["gt"])
                    for r in by_key[key])
        accepted[key] = 1.0 if cal_s + 1e-9 < raw_s else ratio
    return accepted


# --------------------------------------------------------------------------
# Load logs once per channel (big files; keep only what we need)
# --------------------------------------------------------------------------
def load_channel(path: Path):
    """Returns (counting_s1, counting_s2, groups) where groups is the C1
    per-decision structure {(img,q,snr): {...}} and counting_* are slim rows."""
    csv.field_size_limit(sys.maxsize)
    c1_rows, c2_rows = [], []
    groups = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            svc = r.get("service_level")
            if svc not in ("1", "2"):
                continue
            key = (r["image_id"], r["question"], r["snr_bin"])
            g = groups.setdefault(key, {
                "qt": r["question_type"], "img": r["image_id"],
                "snr": r["snr_bin"], "feat": None})
            g[f"s{svc}"] = ok(r["correct"])
            if svc == "1":
                g["feat"] = featurize(r)
            if r["question_type"] == "counting":
                slim = {
                    "image_id": r["image_id"],
                    "target_class": r["target_class"],
                    "snr_bin": r["snr_bin"],
                    "gt": int(float(r["object_count"] or 0)),
                    "det": int(float(r["transmitted_detector_count"] or 0)),
                    "cal_logged": int(float(r["calibrated_detector_count"] or 0)),
                    "correct": ok(r["correct"]),
                    "test": is_test(r["image_id"]),
                }
                (c1_rows if svc == "1" else c2_rows).append(slim)
    groups = {k: g for k, g in groups.items()
              if g["feat"] is not None and "s1" in g and "s2" in g}
    return c1_rows, c2_rows, groups


QTYPES = ["presence", "counting", "comparison", "co_presence", "threshold"]
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
           "awning-tricycle", "bus", "motor"]
VIEWS = ["poor", "fair", "good"]
RISKS = ["normal", "elevated", "critical"]


def featurize(r):
    f = []
    f += [1.0 if r["question_type"] == q else 0.0 for q in QTYPES]
    f += [1.0 if r.get("target_class") == c else 0.0 for c in CLASSES]
    f += [1.0 if r.get("view_quality_bin") == v else 0.0 for v in VIEWS]
    f += [1.0 if r.get("risk_level") == k else 0.0 for k in RISKS]
    f.append(snr_val(r["snr_bin"]) / 20.0)
    raw = float(r.get("raw_detector_count") or 0)
    f.append(min(raw, 60.0) / 60.0)
    f.append(1.0 if raw > 0 else 0.0)
    f.append(0.0)
    f.append(1.0 if r.get("presence_polarity") == "negative" else 0.0)
    return f


def fit_logistic(X, y, l2=1e-3, iters=400, lr=0.5):
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w


def predict(w, X):
    Xb = np.hstack([np.asarray(X), np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-Xb @ w))


# --------------------------------------------------------------------------
def mcnemar(b: int, c: int) -> float:
    """Two-sided exact-ish binomial McNemar p (normal approx for large n)."""
    n = b + c
    if n == 0:
        return 1.0
    z = (abs(b - c) - 1) / math.sqrt(n)
    return math.erfc(z / math.sqrt(2))


def main() -> None:
    out = {}
    power = json.loads(POWER_JSON.read_text())
    e_vlm_inc = power["phases"]["vlm"]["joule_per_item_incremental"]

    # tidy rows for M7 energy ratios (qtype=all, test)
    tidy = defaultdict(dict)
    for r in csv.DictReader(open(TIDY_CSV)):
        if r["qtype"] != "all" or r["split"] != "test":
            continue
        tidy[(r["channel"], r["method"])][float(r["snr_db"])] = {
            "acc": float(r["accuracy"]),
            "uses": float(r["mean_channel_uses"] or 0.0),
        }

    # ---- M7 / O1: image-vs-routing energy ratio, all channels, all SNRs ----
    m7 = {}
    for ch in CHANNELS:
        m7[ch] = {}
        for s in sorted(tidy[(ch, "M4_adaptive")]):
            def energy(m, f_img):
                uses = tidy[(ch, m)][s]["uses"]
                e_tx = uses / BANDWIDTH_HZ * P_TX_HEAD
                return e_tx + f_img * e_vlm_inc + (1 - f_img) * E_DET_MID_J
            e_img = energy("M1_image", 1.0)
            e_rt = energy("M4_adaptive", F_IMG_M4)
            m7[ch][s] = {"E_image": round(e_img, 3), "E_routing": round(e_rt, 3),
                         "ratio": round(e_img / e_rt, 3)}
    out["M7_energy_ratio"] = m7

    # ---- per-channel log loading ------------------------------------------
    chan_data = {}
    for ch in CHANNELS:
        print(f"[load] {ch} ...", flush=True)
        chan_data[ch] = load_channel(PRED[ch])

    # ---- M11: disjoint-fit counting calibration ----------------------------
    m11 = {"per_channel": {}}
    pooled = {"tok_pooled": [0, 0], "tok_disjoint": [0, 0], "img": [0, 0]}
    for ch in CHANNELS:
        c1_rows, c2_rows, _ = chan_data[ch]
        # validation: pooled fit must reproduce the logged calibrated counts
        pooled_ratios = fit_ratios(c1_rows, lambda r: True)
        n_match = sum(
            calibrated_count(r["det"], pooled_ratios.get((r["target_class"], r["snr_bin"]), 1.0)) == r["cal_logged"]
            for r in c1_rows)
        # disjoint fit: train split only
        train_ratios = fit_ratios(c1_rows, lambda r: not r["test"])
        test_rows = [r for r in c1_rows if r["test"]]
        tok_pooled = [sum(count_correct(
            calibrated_count(r["det"], pooled_ratios.get((r["target_class"], r["snr_bin"]), 1.0)), r["gt"])
            for r in test_rows), len(test_rows)]
        tok_disj = [sum(count_correct(
            calibrated_count(r["det"], train_ratios.get((r["target_class"], r["snr_bin"]), 1.0)), r["gt"])
            for r in test_rows), len(test_rows)]
        tok_logged = [sum(r["correct"] for r in test_rows), len(test_rows)]
        img_rows = [r for r in c2_rows if r["test"]]
        img = [sum(r["correct"] for r in img_rows), len(img_rows)]
        m11["per_channel"][ch] = {
            "replication_match": f"{n_match}/{len(c1_rows)}",
            "token_acc_logged": round(tok_logged[0] / tok_logged[1], 4),
            "token_acc_pooled_refit": round(tok_pooled[0] / tok_pooled[1], 4),
            "token_acc_disjoint_fit": round(tok_disj[0] / tok_disj[1], 4),
            "image_acc": round(img[0] / img[1], 4),
            "delta_pooled": round(tok_pooled[0] / tok_pooled[1] - img[0] / img[1], 4),
            "delta_disjoint": round(tok_disj[0] / tok_disj[1] - img[0] / img[1], 4),
            "n_test": len(test_rows),
        }
        for k, v in (("tok_pooled", tok_pooled), ("tok_disjoint", tok_disj), ("img", img)):
            pooled[k][0] += v[0]
            pooled[k][1] += v[1]
    m11["pooled_3ch"] = {
        "token_acc_pooled": round(pooled["tok_pooled"][0] / pooled["tok_pooled"][1], 4),
        "token_acc_disjoint": round(pooled["tok_disjoint"][0] / pooled["tok_disjoint"][1], 4),
        "image_acc": round(pooled["img"][0] / pooled["img"][1], 4),
        "delta_pooled": round(pooled["tok_pooled"][0] / pooled["tok_pooled"][1] - pooled["img"][0] / pooled["img"][1], 4),
        "delta_disjoint": round(pooled["tok_disjoint"][0] / pooled["tok_disjoint"][1] - pooled["img"][0] / pooled["img"][1], 4),
    }
    out["M11_disjoint_calibration"] = m11

    # ---- M6: 5 dB Rician image vs error-free image, paired ----------------
    csv.field_size_limit(sys.maxsize)
    clean = {}
    payload_clean = []
    for r in csv.DictReader(open(CLEAN, newline="")):
        if r.get("service_level") != "2" or not is_test(r["image_id"]):
            continue
        clean[(r["image_id"], r["question"])] = (ok(r["correct"]), r["question_type"])
        payload_clean.append(float(r.get("payload_bytes") or 0))
    ric5, payload_5db = {}, []
    with open(PRED["rician"], newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("service_level") == "2" and r.get("snr_bin") == "5dB"
                    and is_test(r["image_id"])):
                ric5[(r["image_id"], r["question"])] = (ok(r["correct"]), r["question_type"])
                payload_5db.append(float(r.get("payload_bytes") or 0))
    keys = sorted(set(clean) & set(ric5))
    per_type = defaultdict(lambda: [0, 0, 0, 0])  # [n, clean_k, ric5_k, _]
    b = c = 0  # b: clean right / 5dB wrong ; c: clean wrong / 5dB right
    for k in keys:
        ck, qt = clean[k]
        rk, _ = ric5[k]
        per_type[qt][0] += 1
        per_type[qt][1] += int(ck)
        per_type[qt][2] += int(rk)
        if ck and not rk:
            b += 1
        if rk and not ck:
            c += 1
    m6 = {"n_paired": len(keys), "discordant_clean_only": b,
          "discordant_5db_only": c, "mcnemar_p": round(mcnemar(b, c), 4),
          "acc_clean": round(sum(clean[k][0] for k in keys) / len(keys), 4),
          "acc_rician5dB": round(sum(ric5[k][0] for k in keys) / len(keys), 4),
          "mean_payload_clean_B": round(float(np.mean(payload_clean)), 1),
          "mean_payload_5dB_B": round(float(np.mean(payload_5db)), 1),
          "per_type": {}}
    for qt, (n, ck, rk, _) in sorted(per_type.items()):
        m6["per_type"][qt] = {"n": n, "clean": round(ck / n, 4),
                              "rician5dB": round(rk / n, 4),
                              "delta": round((rk - ck) / n, 4)}
    out["M6_inversion"] = m6

    # ---- C1: energy-controllable routing frontier --------------------------
    c1_out = {}
    for ch in CHANNELS:
        _, _, groups = chan_data[ch]
        train = [g for g in groups.values() if not is_test(g["img"])]
        test = [g for g in groups.values() if is_test(g["img"])]
        w1 = fit_logistic([g["feat"] for g in train], [g["s1"] for g in train])
        w2 = fit_logistic([g["feat"] for g in train], [g["s2"] for g in train])
        p1 = predict(w1, [g["feat"] for g in test])
        p2 = predict(w2, [g["feat"] for g in test])
        # per-decision energies (same accounting as make_energy_figures.py)
        e_tok, e_img = {}, {}
        for s, d in tidy[(ch, "M3_token")].items():
            e_tok[s] = d["uses"] / BANDWIDTH_HZ * P_TX_HEAD + E_DET_MID_J
        for s, d in tidy[(ch, "M1_image")].items():
            e_img[s] = d["uses"] / BANDWIDTH_HZ * P_TX_HEAD + e_vlm_inc
        E1 = np.array([e_tok[snr_val(g["snr"])] for g in test])
        E2 = np.array([e_img[snr_val(g["snr"])] for g in test])
        y1 = np.array([g["s1"] for g in test], dtype=float)
        y2 = np.array([g["s2"] for g in test], dtype=float)

        def realized(pick_img: np.ndarray):
            acc = float(np.where(pick_img, y2, y1).mean())
            en = float(np.where(pick_img, E2, E1).mean())
            return acc, en

        lambdas = [0.0] + list(np.logspace(-5, -0.5, 46)) + [1e9]
        pts = []
        for lam in lambdas:
            pick = (p2 - lam * E2) > (p1 - lam * E1)
            acc, en = realized(pick)
            pts.append({"lambda": lam, "acc": round(acc, 4),
                        "E_j": round(en, 3),
                        "frac_image": round(float(pick.mean()), 4)})
        # boundary references
        refs = {
            "all_token": dict(zip(("acc", "E_j"), map(lambda v: round(v, 4), realized(np.zeros(len(test), bool))))),
            "all_image": dict(zip(("acc", "E_j"), map(lambda v: round(v, 4), realized(np.ones(len(test), bool))))),
            "accuracy_first_persample": dict(zip(("acc", "E_j"), map(lambda v: round(v, 4), realized(p2 > p1)))),
            "type_rule_M4": dict(zip(("acc", "E_j"), map(lambda v: round(v, 4), realized(np.array([g["qt"] == "presence" for g in test]))))),
        }
        # monotone (Pareto) filter for reporting
        pts_sorted = sorted(pts, key=lambda d: d["E_j"])
        pareto, best = [], -1.0
        for d in pts_sorted:
            if d["acc"] > best + 1e-9:
                pareto.append(d)
                best = d["acc"]
        c1_out[ch] = {"n_test": len(test), "refs": refs,
                      "n_swept": len(pts), "pareto_points": pareto}
        with open(OUT_DIR / f"c1_frontier_{ch}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lambda", "acc", "E_j", "frac_image"])
            for d in pts:
                w.writerow([d["lambda"], d["acc"], d["E_j"], d["frac_image"]])
    out["C1_frontier"] = c1_out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "review_fix_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
