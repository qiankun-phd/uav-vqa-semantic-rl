#!/usr/bin/env python3
"""Harvest analysis for s3 ROI + W2 presence t-scan chain (CPU only)."""
import csv, os
from collections import defaultdict

REP = "outputs/reports"

def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# ---------- 1. s3 service ladder (payload step: s1 token vs s3 roi vs s2 image) ----------
comp = read(f"{REP}/comparison_s3.csv")
ladder_methods = {"M3_token": "s1_token", "M6_roi": "s3_roi", "M1_image": "s2_image"}
out = []
for qt in ("all", "presence"):
    for row in comp:
        if row["method"] in ladder_methods and row["qtype"] == qt:
            out.append({
                "qtype": qt, "service": ladder_methods[row["method"]],
                "method": row["method"], "snr_db": row["snr_db"],
                "accuracy": row["accuracy"], "lcb": row["lcb"], "ucb": row["ucb"],
                "mean_payload_bytes": row["mean_payload_bytes"],
                "mean_payload_kb": f"{float(row['mean_payload_bytes'])/1024:.3f}",
                "mean_channel_uses": row["mean_channel_uses"], "cbr": row["cbr"],
            })
with open(f"{REP}/s3_service_ladder.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("=== s3 SERVICE LADDER (qtype=all), payload KB by SNR ===")
print(f"{'snr':>5} {'s1_token':>18} {'s3_roi':>18} {'s2_image':>18}")
byk = defaultdict(dict)
for r in out:
    if r["qtype"] == "all":
        byk[r["snr_db"]][r["service"]] = (r["accuracy"], r["mean_payload_kb"])
for snr in sorted(byk, key=float):
    d = byk[snr]
    def fmt(s): return f"acc{d[s][0]} {d[s][1]}KB" if s in d else "-"
    print(f"{snr:>5} {fmt('s1_token'):>18} {fmt('s3_roi'):>18} {fmt('s2_image'):>18}")

# ---------- 2. s3 vs s2 by SNR reversal check (qtype=all + presence breakdown) ----------
print("\n=== s3(M6_roi) vs s2(M1_image) accuracy by SNR, qtype=all ===")
m6 = {r["snr_db"]: r for r in comp if r["method"]=="M6_roi" and r["qtype"]=="all"}
m1 = {r["snr_db"]: r for r in comp if r["method"]=="M1_image" and r["qtype"]=="all"}
for snr in sorted(m6, key=float):
    a6, a1 = float(m6[snr]["accuracy"]), float(m1[snr]["accuracy"])
    flag = "  <-- s3>=s2" if a6>=a1 else ""
    print(f"  snr={snr:>5}: s3={a6:.4f} (payload {float(m6[snr]['mean_payload_bytes'])/1024:.1f}KB) "
          f"s2={a1:.4f} (payload {float(m1[snr]['mean_payload_bytes'])/1024:.1f}KB){flag}")

# ---------- 3. W2 presence accuracy(t, snr) pivot from token_budget_full ----------
tbf = read(f"{REP}/token_budget_full.csv")
pres = [r for r in tbf if r["qtype"]=="presence"]
snrs = sorted({r["snr_db"] for r in pres}, key=float)
def tkey(v):
    try: return (0, float(v))
    except: return (1, v)
ts = sorted({r["t_budget"] for r in pres}, key=tkey)
pivot = {(r["t_budget"], r["snr_db"]): r["accuracy"] for r in pres}
with open(f"{REP}/w2_presence_accuracy_t_snr.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_budget"]+[f"snr_{s}" for s in snrs])
    for t in ts:
        w.writerow([t]+[pivot.get((t,s),"") for s in snrs])
print("\n=== W2 presence accuracy(t_budget x snr) ===")
print("t\\snr " + " ".join(f"{s:>7}" for s in snrs))
for t in ts:
    print(f"{t:>4} " + " ".join(f"{pivot.get((t,s),'-'):>7}" for s in snrs))

# ---------- 4. confidence reliability + AUC from raw ----------
raw = read(f"{REP}/presence_token_budget_raw.csv")
def auc(pairs):  # pairs: list of (score, label 0/1)
    pos = [s for s,l in pairs if l==1]; neg = [s for s,l in pairs if l==0]
    if not pos or not neg: return None
    ranked = sorted(pairs, key=lambda x: x[0])
    # Mann-Whitney U via rank sum with tie handling
    scores = [s for s,_ in ranked]
    ranks = [0.0]*len(ranked); i=0
    while i < len(ranked):
        j=i
        while j<len(ranked) and scores[j]==scores[i]: j+=1
        avg=(i+1+j)/2.0
        for k in range(i,j): ranks[k]=avg
        i=j
    rank_pos = sum(rk for rk,(_,l) in zip(ranks,ranked) if l==1)
    U = rank_pos - len(pos)*(len(pos)+1)/2.0
    return U/(len(pos)*len(neg))

def parse(r):
    try: c=float(r["answer_confidence"])
    except: return None
    lab = 1 if str(r["correct"]).strip().lower() in ("1","true","1.0") else 0
    return (c,lab)

allp=[p for p in (parse(r) for r in raw) if p]
overall=auc(allp)
print(f"\n=== CONFIDENCE AUC (answer_confidence vs correct) ===")
print(f"  overall n={len(allp)} AUC={overall:.4f}")
bt=defaultdict(list)
for r in raw:
    p=parse(r)
    if p: bt[r['t_budget']].append(p)
auc_rows=[["scope","n","auc"],["overall",len(allp),f"{overall:.4f}"]]
for t in sorted(bt, key=tkey):
    a=auc(bt[t]); auc_rows.append([f"t={t}",len(bt[t]),f"{a:.4f}" if a else "NA"])
    print(f"  t={t:>4} n={len(bt[t]):>4} AUC={a:.4f}" if a else f"  t={t}: NA")
with open(f"{REP}/confidence_auc.csv","w",newline="") as f:
    csv.writer(f).writerows(auc_rows)
print("\nWROTE: s3_service_ladder.csv, w2_presence_accuracy_t_snr.csv, confidence_auc.csv")
