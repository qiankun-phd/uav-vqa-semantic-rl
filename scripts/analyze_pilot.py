import csv
from collections import defaultdict
rows=list(csv.DictReader(open("outputs/vlm/v2_0_snr_predictions.csv")))
print("rows:", len(rows), "cols:", [c for c in rows[0].keys()][:12])
def istrue(v): return str(v).lower() in ("1","true","yes")
def snrkey(r): return r.get("snr_bin") or r.get("channel_bin")
SN=["-5dB","0dB","5dB","10dB","15dB","20dB"]
for q in ["presence","counting","ALL"]:
    print(f"\n=== question_type={q} : accuracy by service x SNR ===")
    print("svc | " + " ".join(f"{s:>7}" for s in SN) + " |  span")
    for sl in ["1","2"]:
        acc={}
        for s in SN:
            sub=[r for r in rows if r["service_level"]==sl and snrkey(r)==s and (q=="ALL" or r["question_type"]==q)]
            if sub: acc[s]=sum(istrue(r["correct"]) for r in sub)/len(sub)
        if acc:
            vals=[acc.get(s,float('nan')) for s in SN]
            span=max(v for v in acc.values())-min(v for v in acc.values())
            print(f" s{sl} | " + " ".join(f"{v:7.3f}" if v==v else "   -   " for v in vals) + f" | {span:.3f}")
