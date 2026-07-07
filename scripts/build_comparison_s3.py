#!/usr/bin/env python3
"""Q1/W1.4: {s1,s2,s3} comparison on the rician channel + presence ROI-mode breakdown.

Merges the logged v26 SmolVLM s1/s2 predictions with the new s3 ROI predictions
(main/cmp/extra task sets), extends the build_comparison_v2 protocol to the
candidate set {1,2,3} and emits:

  * outputs/reports/comparison_s3.csv          -- M1_image / M3_token / M6_roi /
      M4_adaptive12 / M4_adaptive123 / M5_oracle12 / M5_oracle123 (test split)
  * outputs/reports/s3_presence_breakdown.csv  -- presence accuracy by
      polarity x service arm x roi_mode (all tasks, per SNR)

Bandwidth accounting mirrors build_comparison_v2: s1 tokens at 0.5 bit/use,
s2 and s3 (both JPEG evidence) at the ergodic spectral efficiency of the
rate-adaptive link.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import build_comparison_v2 as bc  # noqa: E402

CH = "rician"
TAGS = ("main", "cmp", "extra")
PRED_DIR = "outputs/vlm"
OUT_CMP = "outputs/reports/comparison_s3.csv"
OUT_BRK = "outputs/reports/s3_presence_breakdown.csv"


def uses_for(service: str, snr, payload_bytes: float) -> float:
    if service == "1":
        return payload_bytes * 8.0 / bc.SE_TOKEN
    return payload_bytes * 8.0 / bc.se_s2(CH, snr)  # s2 and s3: JPEG over rate-adaptive link


def load_all():
    """(image_id, question, snr) -> {service: (correct, bytes)}, plus qtype/meta maps."""
    tasks = defaultdict(dict)
    qtype, roi_mode, polarity = {}, {}, {}
    seen_files = []
    for tag in TAGS:
        for path, svc_keep in ((f"{PRED_DIR}/v26_rician_{tag}_predictions.csv", ("1", "2")),
                               (f"{PRED_DIR}/s3_rician_{tag}_predictions.csv", ("3",))):
            if not os.path.exists(path):
                print(f"skip missing {path}")
                continue
            seen_files.append(path)
            for r in csv.DictReader(open(path)):
                if r["service_level"] not in svc_keep:
                    continue
                key = (r["image_id"], r["question"], r["snr_bin"])
                tasks[key][r["service_level"]] = (bc.correct(r["correct"]), int(r.get("payload_bytes") or 0))
                qtype[key] = r.get("question_type", "")
                if r["service_level"] == "3":
                    roi_mode[key] = r.get("roi_mode", "")
                if r.get("presence_polarity"):
                    polarity[key] = r["presence_polarity"]
    print("loaded:", *seen_files, sep="\n  ")
    return tasks, qtype, roi_mode, polarity


def learn_policy(tasks, qtype, cands):
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, svcs in tasks.items():
        if bc.is_test(key[0]):
            continue
        cs = (qtype[key], key[2])
        for s in cands:
            if s in svcs:
                cell[cs][s][1] += 1
                cell[cs][s][0] += int(svcs[s][0])
    pol = {}
    for cs, sv in cell.items():
        best, blcb = None, -1.0
        for s, (k, n) in sv.items():
            _, lcb, _ = bc.wilson(k, n)
            if lcb > blcb:
                blcb, best = lcb, s
        pol[cs] = best
    return pol


def main() -> int:
    tasks, qtype, roi_mode, polarity = load_all()
    pol12 = learn_policy(tasks, qtype, ("1", "2"))
    pol123 = learn_policy(tasks, qtype, ("1", "2", "3"))

    # method -> snr -> qbucket -> [k, n, bytes, uses]
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0.0])))
    fixed = {"M1_image": "2", "M3_token": "1", "M6_roi": "3"}
    for key, svcs in tasks.items():
        if not bc.is_test(key[0]):
            continue
        qt, snr = qtype[key], key[2]
        sv = bc.snr_val(snr)
        for qb in ("all", qt):
            for method, s in fixed.items():
                if s in svcs:
                    c, b = svcs[s]
                    a = acc[method][snr][qb]
                    a[0] += int(c); a[1] += 1; a[2] += b; a[3] += uses_for(s, sv, b)
            for method, pol in (("M4_adaptive12", pol12), ("M4_adaptive123", pol123)):
                s = pol.get((qt, snr))
                if s in svcs:
                    c, b = svcs[s]
                    a = acc[method][snr][qb]
                    a[0] += int(c); a[1] += 1; a[2] += b; a[3] += uses_for(s, sv, b)
            for method, cands in (("M5_oracle12", ("1", "2")), ("M5_oracle123", ("1", "2", "3"))):
                cand = [(s, svcs[s]) for s in cands if s in svcs]
                if cand:
                    s, best = max(cand, key=lambda t: int(t[1][0]))
                    a = acc[method][snr][qb]
                    a[0] += int(best[0]); a[1] += 1; a[2] += best[1]; a[3] += uses_for(s, sv, best[1])

    out_rows = []
    for method, sd in sorted(acc.items()):
        for snr, qd in sd.items():
            for qt, (k, n, b, u) in qd.items():
                if n == 0:
                    continue
                p, lcb, ucb = bc.wilson(k, n)
                mu = u / n
                out_rows.append([CH, method, bc.snr_val(snr), qt, "test", round(p, 4), n,
                                 round(lcb, 4), round(ucb, 4), round(b / n, 1),
                                 round(mu, 1), round(mu / bc.MEAN_SOURCE_REALS, 6)])
    os.makedirs(os.path.dirname(OUT_CMP), exist_ok=True)
    with open(OUT_CMP, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "method", "snr_db", "qtype", "split", "accuracy", "n",
                    "lcb", "ucb", "mean_payload_bytes", "mean_channel_uses", "cbr"])
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {OUT_CMP}")

    # ---- presence breakdown: polarity x arm x roi_mode (all tasks) ----
    brk = defaultdict(lambda: [0, 0])  # (snr, polarity, arm, mode) -> [k, n]
    for key, svcs in tasks.items():
        if qtype[key] != "presence":
            continue
        pol = polarity.get(key, "")
        snr = key[2]
        for s, arm in (("1", "s1_token"), ("2", "s2_image"), ("3", "s3_roi")):
            if s not in svcs:
                continue
            k = int(svcs[s][0])
            for mode in (["all"] + ([roi_mode.get(key, "")] if s == "3" and roi_mode.get(key) else [])):
                a = brk[(snr, pol, arm, mode)]
                a[0] += k; a[1] += 1
    with open(OUT_BRK, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "polarity", "arm", "roi_mode", "accuracy", "n", "lcb", "ucb"])
        for (snr, pol, arm, mode), (k, n) in sorted(brk.items(), key=lambda t: (bc.snr_val(t[0][0]), t[0][1], t[0][2], t[0][3])):
            p, lcb, ucb = bc.wilson(k, n)
            w.writerow([bc.snr_val(snr), pol, arm, mode, round(p, 4), n, round(lcb, 4), round(ucb, 4)])
    print(f"wrote presence breakdown -> {OUT_BRK}")

    # ---- key readouts ----
    def agg(rows_filter):
        k = n = 0
        for key_, (kk, nn) in rows_filter:
            k += kk; n += nn
        return (k / n if n else 0.0), n

    print("\n[key readout 1] negative-polarity presence (all SNR):")
    for arm in ("s1_token", "s2_image", "s3_roi"):
        p, n = agg([(kk, v) for kk, v in brk.items() if kk[1] == "negative" and kk[2] == arm and kk[3] == "all"])
        print(f"  {arm:9s}: acc={p:.4f} n={n}")
    for mode in ("suspect", "thumbnail", "target_topk"):
        p, n = agg([(kk, v) for kk, v in brk.items() if kk[1] == "negative" and kk[2] == "s3_roi" and kk[3] == mode])
        if n:
            print(f"  s3[{mode:11s}]: acc={p:.4f} n={n}")

    print("\n[key readout 2] low-SNR (-5,0 dB) accuracy, all qtypes (test):")
    for method in ("M3_token", "M1_image", "M6_roi", "M4_adaptive12", "M4_adaptive123", "M5_oracle123"):
        for snr in ("-5dB", "0dB"):
            a = acc[method].get(snr, {}).get("all")
            if a and a[1]:
                print(f"  {method:14s} @{snr:>5s}: acc={a[0]/a[1]:.4f} n={a[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
