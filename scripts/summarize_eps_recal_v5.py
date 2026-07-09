#!/usr/bin/env python
"""Task #28 iteration 5: 6-generation comparison legacy/v1/v2/v3/v4/v5 + v5 criteria.

v5 = outputs/rl/eps_recal_v5  (attainability_v5 two-key {0.464/0.696/0.529} on the
lut_v5 backend, cache_quality=entry_v2, escalation layer, cache FORBIDDEN).

Criteria (semSucc / accuracy measured on the ADMITTED = non-escalated set):
  1. PEAK oracle semSucc(admitted) >= 0.85
  2. PEAK proposed critical-escalation rate in [spec_unattainable +- 0.05] and
     escalated tasks carry zero silent quality violations
  3. PEAK proposed cacheR<0.40 and rejectR<0.20 and semSucc(admitted)>=0.6*oracle
     and acc>=0.7*oracle_acc
  4. lambda_quality_critical active: terminal in (0.1, 0.9*lambda_max) OR
     quality_cost_critical in [delta-0.02, delta+0.05]
  5. NOMINAL proposed semSucc(admitted)>=0.89 and task success>=0.75
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path.home() / "phd_research/vqa_main/outputs/rl"
CALIB = Path.home() / "phd_research/vqa_main/outputs/rl/eps_v5_calib.json"
GENS = [("legacy", ROOT / "matrix_v1"), ("v1", ROOT / "eps_recal_v1"),
        ("v2", ROOT / "eps_recal_v2"), ("v3", ROOT / "eps_recal_v3"),
        ("v4", ROOT / "eps_recal_v4"), ("v5", ROOT / "eps_recal_v5")]
LAMBDA_MAX = 20.0
QUALITY_LIMIT_CRIT = 0.02

METRICS = ["average_epsilon_k", "semantic_success_rate", "quality_violation_rate",
           "task_success_rate", "average_accuracy", "semantic_path_cache_ratio",
           "deadline_violation_rate", "reject_ratio",
           "escalation_rate", "critical_escalation_rate", "spec_attainable_rate",
           "admitted_semantic_success_rate", "admitted_average_accuracy",
           "admitted_quality_violation_rate"]
POLICY_OF = {"proposed": "ppo", "no_lagrangian": "ppo", "fixed_penalty": "ppo", "e4lut": "ppo",
             "bl_oracle_best_feasible_evidence": "oracle_best_feasible_evidence",
             "bl_semantic_greedy": "semantic_greedy", "bl_always_cache": "always_cache",
             "bl_random": "random"}


def _read_row(d: Path, policy: str):
    f = d / "v1_9_resource_alloc_results.csv"
    if not f.exists():
        return None
    for row in csv.DictReader(f.open()):
        if str(row.get("policy", "")) == policy:
            return {k: float(row.get(k, 0.0) or 0.0) for k in METRICS}
    return None


def collect(root: Path):
    out: dict[str, dict[str, list]] = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.endswith("_nomtrain"):
            cond, base = "nomtrain", name[:-9]
        elif name.endswith("_nom"):
            cond, base = "nominal", name[:-4]
        else:
            cond, base = "peak", name
        if "_" not in base:
            continue
        arm, _seed = base.rsplit("_", 1)
        if arm not in POLICY_OF:
            continue
        r = _read_row(d, POLICY_OF[arm])
        if r is not None:
            out.setdefault(arm, {}).setdefault(cond, []).append(r)
    return out


DATA = {tag: collect(root) for tag, root in GENS}
CAL = json.loads(CALIB.read_text()) if CALIB.exists() else {}


def agg(rows, key):
    vals = [r[key] for r in rows if key in r]
    return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), 0.0)


def m(tag, arm, cond, key):
    rows = DATA[tag].get(arm, {}).get(cond)
    return agg(rows, key)[0] if rows else float("nan")


ARMS = ["bl_oracle_best_feasible_evidence", "bl_always_cache", "bl_random",
        "proposed", "no_lagrangian", "fixed_penalty", "e4lut"]

for cond in ("peak", "nominal", "nomtrain"):
    print("=" * 130)
    print(f"6-GEN COMPARISON -- {cond} condition")
    print("=" * 130)
    hdr = (f"{'arm':32} {'gen':6} {'eps':>6} {'semSucc':>8} {'admSucc':>8} {'qViol':>7} "
           f"{'taskSucc':>9} {'acc':>6} {'admAcc':>7} {'cacheR':>7} {'rejectR':>8} {'escR':>6} {'critEsc':>8} {'specAtt':>8}")
    print(hdr)
    print("-" * len(hdr))
    for arm in ARMS:
        for tag, _ in GENS:
            rows = DATA[tag].get(arm, {}).get(cond)
            if not rows:
                continue
            g = lambda k: agg(rows, k)[0]
            print(f"{arm:32} {tag:6} {g('average_epsilon_k'):6.3f} {g('semantic_success_rate'):8.3f} "
                  f"{g('admitted_semantic_success_rate'):8.3f} {g('quality_violation_rate'):7.3f} "
                  f"{g('task_success_rate'):9.3f} {g('average_accuracy'):6.3f} {g('admitted_average_accuracy'):7.3f} "
                  f"{g('semantic_path_cache_ratio'):7.3f} {g('reject_ratio'):8.3f} {g('escalation_rate'):6.3f} "
                  f"{g('critical_escalation_rate'):8.3f} {g('spec_attainable_rate'):8.3f}")
        print()


def lam(arm_dir: Path):
    f = arm_dir / "ppo_lambda_trace.csv"
    if not f.exists():
        return None
    q, cost, esc, lesc = [], [], [], []
    for row in csv.DictReader(f.open()):
        try:
            q.append(float(row["lambda_quality_critical"]))
            cost.append(float(row.get("quality_cost_critical", 0.0)))
            esc.append(float(row.get("escalation_cost", 0.0)))
            lesc.append(float(row.get("lambda_escalation", 0.0)))
        except (KeyError, ValueError):
            pass
    if not q:
        return None
    pk = max(range(len(q)), key=lambda i: q[i])
    span = max(q) - min(q)
    pinned = span < 1e-3 or (q[pk] - q[-1]) < 0.15 * span
    return dict(first=q[0], last=q[-1], mn=min(q), mx=max(q), pinned=pinned,
                costmean=mean(cost[-50:] if len(cost) >= 50 else cost),
                escmean=mean(esc[-50:] if len(esc) >= 50 else esc),
                lesc_last=lesc[-1])


print("=" * 130)
print("LAMBDA / ESCALATION TRAJECTORY (v5 peak training, terminal-window means)")
print(f"{'arm/seed':16} {'lamQC_first':>11} {'lamQC_last':>10} {'lamQC_max':>10} {'pinned?':>8} "
      f"{'qCostCrit':>9} {'escCost':>8} {'lamEsc':>7}")
for arm in ["proposed", "no_lagrangian", "fixed_penalty", "e4lut"]:
    for seed in (0, 1, 2):
        sh = lam(ROOT / "eps_recal_v5" / f"{arm}_{seed}")
        if sh is None:
            continue
        print(f"{arm+'_'+str(seed):16} {sh['first']:11.3f} {sh['last']:10.3f} {sh['mx']:10.3f} "
              f"{str(sh['pinned']):>8} {sh['costmean']:9.3f} {sh['escmean']:8.3f} {sh['lesc_last']:7.3f}")

print("\n" + "=" * 130)
print("V5 SUCCESS CRITERIA")
print("=" * 130)
spec_unattain = float(CAL.get("peak_decomposition", {}).get("spec_unattainable", 0.90))
checks = []

o = m("v5", "bl_oracle_best_feasible_evidence", "peak", "admitted_semantic_success_rate")
oa = m("v5", "bl_oracle_best_feasible_evidence", "peak", "admitted_average_accuracy")
checks.append(("1. PEAK oracle semSucc(admitted) >= 0.85", o, o >= 0.85))

pesc = m("v5", "proposed", "peak", "critical_escalation_rate")
lo, hi = spec_unattain - 0.05, min(1.0, spec_unattain + 0.05)
checks.append((f"2. PEAK proposed crit-escalation in [{lo:.2f},{hi:.2f}] (spec_unattain {spec_unattain:.2f})",
               pesc, lo <= pesc <= hi))

pc = m("v5", "proposed", "peak", "semantic_path_cache_ratio")
pr = m("v5", "proposed", "peak", "reject_ratio")
pa = m("v5", "proposed", "peak", "admitted_average_accuracy")
ps = m("v5", "proposed", "peak", "admitted_semantic_success_rate")
checks.append(("3a. PEAK proposed cacheR < 0.40", pc, pc < 0.40))
checks.append(("3b. PEAK proposed rejectR < 0.20", pr, pr < 0.20))
checks.append((f"3c. PEAK proposed semSucc(admitted) >= 0.6*oracle ({0.6*o:.3f})", ps, ps >= 0.6 * o))
checks.append((f"3d. PEAK proposed acc(admitted) >= 0.7*oracle_acc ({0.7*oa:.3f})", pa, pa >= 0.7 * oa))

active = []
for seed in (0, 1, 2):
    sh = lam(ROOT / "eps_recal_v5" / f"proposed_{seed}")
    if sh:
        band = (0.1 < sh["last"] < 0.9 * LAMBDA_MAX)
        cost_ok = (QUALITY_LIMIT_CRIT - 0.02) <= sh["costmean"] <= (QUALITY_LIMIT_CRIT + 0.05)
        active.append(band or cost_ok)
checks.append(("4. lambda_quality_critical active (>=2/3 seeds)", sum(active), sum(active) >= 2))

ns = m("v5", "proposed", "nominal", "admitted_semantic_success_rate")
nt = m("v5", "proposed", "nominal", "task_success_rate")
checks.append(("5a. NOMINAL proposed semSucc(admitted) >= 0.89", ns, ns >= 0.89))
checks.append(("5b. NOMINAL proposed task success >= 0.75", nt, nt >= 0.75))

npass = 0
for name, val, ok in checks:
    npass += bool(ok)
    v = f"{val:.4f}" if isinstance(val, float) else str(val)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  (obs {v})")
print(f"\n{npass}/{len(checks)} criteria PASS  ->  {'PASS' if npass == len(checks) else 'FAIL'}")
