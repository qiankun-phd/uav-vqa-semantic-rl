#!/usr/bin/env python3
"""E4 quality-backend A/B aggregation (lut vs persample, dual condition)."""
import csv, os, statistics, glob

BASE = "outputs/rl"
def load_results(d):
    p = os.path.join(d, "v1_9_resource_alloc_results.csv")
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    return rows[0] if rows else None
def final_lambda(d):
    p = os.path.join(d, "ppo_lambda_trace.csv")
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    if not rows: return None
    return float(rows[-1].get("lambda_quality", "nan"))

METRICS = ["average_accuracy","average_accuracy_mean","quality_violation_rate",
           "deadline_violation_rate","utm_conflict_violation_rate","average_payload_kb"]

def collect(group_dir, arm, cond_suffix, seeds):
    recs=[]
    for s in seeds:
        d = os.path.join(group_dir, f"{arm}_{s}{cond_suffix}")
        r = load_results(d)
        if not r: continue
        rec = {"seed": s, "dir": d, "scenario": r["scenario"]}
        for m in METRICS: rec[m]=float(r[m])
        lam = final_lambda(d)
        if lam is not None: rec["lambda_quality"]=lam
        recs.append(rec)
    return recs

def agg(recs, field):
    vals=[r[field] for r in recs if field in r]
    if not vals: return None
    m=statistics.mean(vals)
    sd=statistics.pstdev(vals) if len(vals)>1 else 0.0
    return m, sd, len(vals)

def report(group_dir, seeds, label):
    print(f"\n{'='*72}\n{label}  ({group_dir})\n{'='*72}")
    out_rows=[]
    for cond,suffix in [("utm_conflict(PEAK)",""),("nominal",  "_nom")]:
        for arm in ["lut","persample"]:
            recs=collect(group_dir, arm, suffix, seeds)
            if not recs:
                print(f"  {cond:20} {arm:10}  (no data)"); continue
            line={"condition":cond,"arm":arm,"n_seeds":len(recs)}
            for f in ["average_accuracy","average_accuracy_mean","quality_violation_rate",
                      "utm_conflict_violation_rate","average_payload_kb","lambda_quality"]:
                a=agg(recs,f)
                if a: line[f]=f"{a[0]:.4f}+-{a[1]:.4f}"
            out_rows.append(line)
            print(f"  {cond:20} {arm:10} n={len(recs)}  "
                  f"accLCB={line.get('average_accuracy','-'):>16}  "
                  f"accMean={line.get('average_accuracy_mean','-'):>16}  "
                  f"qvio={line.get('quality_violation_rate','-'):>16}  "
                  f"payloadKB={line.get('average_payload_kb','-'):>16}  "
                  f"lam_q={line.get('lambda_quality','-'):>16}")
    return out_rows

allrows=[]
r5=report(os.path.join(BASE,"e4_quality_backend_ab"), [0,1,2], "E4 500-episode (3 seeds)")
for x in r5: x["group"]="500ep"; allrows.append(x)
r1=report(os.path.join(BASE,"e4_quality_backend_ab_1000"), [0,1,2], "E4 1000-episode (INCOMPLETE: lut seed0 + nom only; persample killed)")
for x in r1: x["group"]="1000ep"; allrows.append(x)

# verdict for 500ep
print(f"\n{'='*72}\nVERDICT (500ep, judge: persample>=lut accuracy & no constraint degradation)\n{'='*72}")
def mean_of(group_dir,arm,suffix,field,seeds=[0,1,2]):
    recs=collect(group_dir,arm,suffix,seeds); a=agg(recs,field); return a[0] if a else None
g=os.path.join(BASE,"e4_quality_backend_ab")
for cond,suffix in [("utm_conflict(PEAK)",""),("nominal","_nom")]:
    for field in ["average_accuracy","average_accuracy_mean"]:
        pl=mean_of(g,"persample",suffix,field); lt=mean_of(g,"lut",suffix,field)
        if pl is None or lt is None: continue
        delta=pl-lt
        verd="persample>=lut" if delta>=-1e-4 else "LUT WINS"
        print(f"  {cond:20} {field:24} persample={pl:.4f} lut={lt:.4f} delta={delta:+.4f}  {verd}")
    # constraint check
    plq=mean_of(g,"persample",suffix,"quality_violation_rate"); ltq=mean_of(g,"lut",suffix,"quality_violation_rate")
    print(f"  {cond:20} {'quality_violation_rate':24} persample={plq:.4f} lut={ltq:.4f} "
          f"({'no degrade' if plq<=ltq+1e-4 else 'WORSE'})")

# write CSV
fields=["group","condition","arm","n_seeds","average_accuracy","average_accuracy_mean",
        "quality_violation_rate","utm_conflict_violation_rate","average_payload_kb","lambda_quality"]
with open("outputs/rl/e4_quality_backend_ab_summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(allrows)
print("\nWROTE outputs/rl/e4_quality_backend_ab_summary.csv")
