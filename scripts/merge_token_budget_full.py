#!/usr/bin/env python3
"""Q2/W2 merge: token_budget_full.csv (symbolic sweep + VLM presence sweep)
and confidence_calibration.csv (reliability of answer_confidence).

  * symbolic question types come from the existing outputs/reports/token_budget_sweep.csv
    (build_token_budget_sweep.py, CPU symbolic decoders);
  * presence rows are aggregated from outputs/reports/presence_token_budget_raw.csv
    (run_presence_token_budget.py, VLM with s1 token evidence), charged at
    0.5 bit/complex-use like every s1 token payload.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
import build_comparison_v2 as bc  # noqa: E402

SWEEP = "outputs/reports/token_budget_sweep.csv"
RAW = "outputs/reports/presence_token_budget_raw.csv"
OUT_FULL = "outputs/reports/token_budget_full.csv"
OUT_CAL = "outputs/reports/confidence_calibration.csv"

CAL_EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0000001]


def main() -> int:
    rows = []
    header = ["channel", "t_budget", "snr_db", "qtype", "accuracy", "n",
              "mean_tokens_sent", "mean_payload_bytes", "mean_channel_uses"]
    if os.path.exists(SWEEP):
        for r in csv.DictReader(open(SWEEP)):
            rows.append([r[h] for h in header])
    else:
        print(f"warning: {SWEEP} missing; token_budget_full will be presence-only")

    agg = defaultdict(lambda: [0, 0, 0, 0, 0.0])  # (ch,t,snr)->k,n,tok,bytes,uses
    cal = defaultdict(lambda: [0, 0, 0.0])        # (t,bin)->k,n,conf_sum
    n_raw = 0
    for r in csv.DictReader(open(RAW)):
        n_raw += 1
        key = (r["channel"], r["t_budget"], float(r["snr_db"]))
        pb = int(r["payload_bytes"])
        a = agg[key]
        a[0] += int(bc.correct(r["correct"])); a[1] += 1
        a[2] += int(r["tokens_sent"]); a[3] += pb; a[4] += pb * 8.0 / bc.SE_TOKEN
        try:
            conf = float(r["answer_confidence"])
        except ValueError:
            continue
        if conf != conf:  # NaN
            continue
        b = next(i for i in range(len(CAL_EDGES) - 1) if CAL_EDGES[i] <= conf < CAL_EDGES[i + 1])
        for tkey in ("all", r["t_budget"]):
            c = cal[(tkey, b)]
            c[0] += int(bc.correct(r["correct"])); c[1] += 1; c[2] += conf

    for (ch, t, snr), (k, n, tok, by, us) in sorted(agg.items(), key=lambda x: (x[0][0], str(x[0][1]), x[0][2])):
        rows.append([ch, t, snr, "presence", round(k / n, 4), n,
                     round(tok / n, 2), round(by / n, 1), round(us / n, 1)])

    os.makedirs(os.path.dirname(OUT_FULL), exist_ok=True)
    with open(OUT_FULL, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({n_raw} raw presence samples) -> {OUT_FULL}")

    with open(OUT_CAL, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_budget", "conf_bin_lo", "conf_bin_hi", "n", "mean_confidence", "accuracy", "gap"])
        for (t, b), (k, n, cs) in sorted(cal.items(), key=lambda x: (str(x[0][0]), x[0][1])):
            mc = cs / n
            acc = k / n
            w.writerow([t, CAL_EDGES[b], round(min(1.0, CAL_EDGES[b + 1]), 4), n,
                        round(mc, 4), round(acc, 4), round(acc - mc, 4)])
    print(f"wrote calibration table -> {OUT_CAL}")

    # ECE (all t)
    tot = sum(n for (t, _), (_, n, _) in cal.items() if t == "all")
    if tot:
        ece = sum(abs(k / n - cs / n) * n for (t, _), (k, n, cs) in cal.items() if t == "all") / tot
        print(f"ECE(all presence samples) = {ece:.4f} over n={tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
