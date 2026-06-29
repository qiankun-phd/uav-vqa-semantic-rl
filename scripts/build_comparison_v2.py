#!/usr/bin/env python3
"""Multi-channel comparison with train/test split + per-question-type breakdown.

For each channel:
  * group predictions by task (image_id, question, snr) -> {service: (correct, bytes)}
  * split tasks by image_id into train (policy learning) / test (reporting), 20% test
  * learn M4 policy on TRAIN: per (question_type, snr) the service with best Wilson-LCB
  * report on TEST: M1 (s2) / M3 (s1) / M4 (adaptive) / M5 (oracle) accuracy + payload
  * merge M2 analog (if its csv exists) -> accuracy per snr (+qtype)
Outputs one tidy long CSV consumed by the figure scripts.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import zlib
from collections import defaultdict

CHANNELS = ["awgn", "rayleigh", "rician"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def correct(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def snr_val(s):
    return float(str(s).replace("dB", "").replace("db", ""))


def is_test(image_id, frac=0.2):
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def load_channel(path):
    """task key -> {service: (correct, bytes)}, plus qtype map."""
    tasks = defaultdict(dict)
    qtype = {}
    for r in csv.DictReader(open(path)):
        key = (r["image_id"], r["question"], r["snr_bin"])
        tasks[key][r["service_level"]] = (correct(r["correct"]), int(r.get("payload_bytes") or 0))
        qtype[key] = r.get("question_type", "")
    return tasks, qtype


def learn_policy(tasks, qtype, split_test):
    """per (qtype, snr) -> best-LCB service among {1,2}, learned on TRAIN tasks."""
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, svcs in tasks.items():
        if is_test(key[0]) == split_test:  # learn on train only -> split_test=False
            continue
        cs = (qtype[key], key[2])
        for s in ("1", "2"):
            if s in svcs:
                cell[cs][s][1] += 1
                cell[cs][s][0] += int(svcs[s][0])
    pol = {}
    for cs, sv in cell.items():
        best, blcb = None, -1
        for s, (k, n) in sv.items():
            _, lcb, _ = wilson(k, n)
            if lcb > blcb:
                blcb, best = lcb, s
        pol[cs] = best
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v2_0")
    ap.add_argument("--out", default="outputs/reports/comparison_all.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    out_rows = []  # channel, method, snr_db, qtype, split, accuracy, n, lcb, ucb, mean_bytes

    def emit(ch, method, snr, qt, split, k, n, b):
        if n == 0:
            return
        p, lcb, ucb = wilson(k, n)
        out_rows.append([ch, method, snr_val(snr), qt, split, round(p, 4), n,
                         round(lcb, 4), round(ucb, 4), round(b / n, 1)])

    for ch in CHANNELS:
        pred = f"{args.pred_dir}/{args.prefix}_{ch}_predictions.csv"
        if not os.path.exists(pred):
            print(f"skip {ch}: no {pred}")
            continue
        tasks, qtype = load_channel(pred)
        pol = learn_policy(tasks, qtype, split_test=False)
        snrs = sorted({k[2] for k in tasks}, key=snr_val)
        # accumulators: method -> snr -> qt -> [k, n, bytes]
        acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0])))
        for key, svcs in tasks.items():
            if not is_test(key[0], args.test_frac):
                continue  # report on TEST only
            qt, snr = qtype[key], key[2]
            for qbucket in ("all", qt):
                if "2" in svcs:
                    c, b = svcs["2"]; a = acc["M1_image"][snr][qbucket]; a[0] += int(c); a[1] += 1; a[2] += b
                if "1" in svcs:
                    c, b = svcs["1"]; a = acc["M3_token"][snr][qbucket]; a[0] += int(c); a[1] += 1; a[2] += b
                ch_sel = pol.get((qt, snr))
                if ch_sel in svcs:
                    c, b = svcs[ch_sel]; a = acc["M4_adaptive"][snr][qbucket]; a[0] += int(c); a[1] += 1; a[2] += b
                cand = [svcs[s] for s in ("1", "2") if s in svcs]
                if cand:
                    best = max(cand, key=lambda t: int(t[0]))
                    a = acc["M5_oracle"][snr][qbucket]; a[0] += int(best[0]); a[1] += 1; a[2] += best[1]
        for method, sd in acc.items():
            for snr, qd in sd.items():
                for qt, (k, n, b) in qd.items():
                    emit(ch, method, snr, qt, "test", k, n, b)

        # ---- merge M2 analog if present ----
        m2 = f"{args.pred_dir}/m2_analog_{ch}_predictions.csv"
        if os.path.exists(m2):
            macc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
            for r in csv.DictReader(open(m2)):
                if not is_test(r["image_id"], args.test_frac):
                    continue
                snr, qt = r["snr_bin"], r.get("question_type", "")
                cu = int(r.get("channel_uses") or 0)
                for qb in ("all", qt):
                    a = macc[snr][qb]; a[0] += int(correct(r["correct"])); a[1] += 1; a[2] += cu
            for snr, qd in macc.items():
                for qt, (k, n, b) in qd.items():
                    emit(ch, "M2_analog", snr, qt, "test", k, n, b)
        else:
            print(f"note: M2 analog not yet available for {ch}")

        # ---- merge NAIVE fixed-rate LDPC digital (cliffs), service s2 only ----
        nv = f"{args.pred_dir}/{args.prefix}_{ch}_naive_predictions.csv"
        if os.path.exists(nv):
            nacc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
            for r in csv.DictReader(open(nv)):
                if r.get("service_level") != "2":
                    continue
                if not is_test(r["image_id"], args.test_frac):
                    continue
                snr, qt = r["snr_bin"], r.get("question_type", "")
                b = int(r.get("payload_bytes") or 0)
                for qb in ("all", qt):
                    a = nacc[snr][qb]; a[0] += int(correct(r["correct"])); a[1] += 1; a[2] += b
            for snr, qd in nacc.items():
                for qt, (k, n, b) in qd.items():
                    emit(ch, "M0_naive", snr, qt, "test", k, n, b)
        else:
            print(f"note: naive digital not yet available for {ch}")
        print(f"{ch}: policy={ {k: pol[k] for k in sorted(pol)} }")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "method", "snr_db", "qtype", "split", "accuracy", "n", "lcb", "ucb", "mean_payload_bytes"])
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows -> {args.out}")

    # quick console summary (qtype=all, test)
    for ch in CHANNELS:
        sub = [r for r in out_rows if r[0] == ch and r[3] == "all"]
        if not sub:
            continue
        snrs = sorted({r[2] for r in sub})
        methods = sorted({r[1] for r in sub})
        print(f"\n[{ch}] accuracy (test, all qtypes):")
        print("  %-12s" % "SNR" + "".join("%8.0f" % s for s in snrs))
        for m in methods:
            line = "  %-12s" % m
            for s in snrs:
                v = [r[5] for r in sub if r[1] == m and r[2] == s]
                line += "%8s" % (f"{v[0]:.3f}" if v else "-")
            print(line)


if __name__ == "__main__":
    raise SystemExit(main())
