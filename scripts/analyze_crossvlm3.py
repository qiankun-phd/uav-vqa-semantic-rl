import csv, zlib
from collections import defaultdict

def is_test(i, frac=0.2): return (zlib.crc32(str(i).encode()) % 100) < int(frac*100)
def ok(v): return str(v).strip().lower() in ("true","1","yes")

def per_qtype(paths):
    acc = defaultdict(lambda: defaultdict(lambda: [0,0]))
    for p in paths:
        try: rows = list(csv.DictReader(open(p)))
        except FileNotFoundError: continue
        for r in rows:
            if not is_test(r["image_id"]): continue
            s = r["service_level"]
            if s not in ("1","2"): continue
            a = acc[r["question_type"]][s]; a[1]+=1; a[0]+=int(ok(r["correct"]))
    return acc

b2 = per_qtype(["outputs/vlm/v3_0_rician_predictions.csv"])
b3 = per_qtype(["outputs/vlm/v25_rician_main_predictions.csv","outputs/vlm/v25_rician_cmp_predictions.csv","outputs/vlm/v25_rician_extra_predictions.csv"])
sm = per_qtype(["outputs/vlm/v26_rician_main_predictions.csv","outputs/vlm/v26_rician_cmp_predictions.csv","outputs/vlm/v26_rician_extra_predictions.csv"])

def rate(a,s): k,n=a[s]; return k/n if n else None
def f(x): return "%.3f"%x if x is not None else "  -  "
def r(t,i):
    if t is None or i is None: return "-"
    return "token" if t>=i else "image"
print("%-12s | %-15s | %-15s | %-15s | routing" % ("qtype","Qwen2-VL-2B t/i","Qwen2.5-VL-3B t/i","SmolVLM-2B t/i"))
print("-"*95)
for qt in ["presence","counting","comparison","co_presence","threshold"]:
    t2,i2 = rate(b2[qt],"1"), rate(b2[qt],"2")
    t3,i3 = rate(b3[qt],"1"), rate(b3[qt],"2")
    ts,is_ = rate(sm[qt],"1"), rate(sm[qt],"2")
    print("%-12s | %s / %s | %s / %s | %s / %s | 2B:%s 3B:%s Smol:%s" % (
        qt, f(t2),f(i2), f(t3),f(i3), f(ts),f(is_), r(t2,i2), r(t3,i3), r(ts,is_)))
