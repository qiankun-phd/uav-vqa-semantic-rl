#!/usr/bin/env python
"""Task #28 iteration 2: 3-generation comparison legacy / v1 / v2 + v2 criteria.

legacy = outputs/rl/matrix_v1 (0.82/0.65)
v1     = outputs/rl/eps_recal_v1 (attainability_v1, 0.615/0.166)
v2     = outputs/rl/eps_recal_v2 (attainability_v2, 0.633/0.297)
"""
from __future__ import annotations
import csv
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path.home() / "phd_research/vqa_main/outputs/rl"
GENS = [("legacy", ROOT / "matrix_v1"),
        ("v1", ROOT / "eps_recal_v1"),
        ("v2", ROOT / "eps_recal_v2")]

METRICS = ["average_epsilon_k", "semantic_success_rate", "quality_violation_rate",
           "task_success_rate", "average_accuracy", "semantic_path_cache_ratio",
           "service_level_0_ratio", "service_level_1_ratio",
           "service_level_2_ratio", "deadline_violation_rate", "average_payload_kb"]
POLICY_OF = {"proposed": "ppo", "no_lagrangian": "ppo", "fixed_penalty": "ppo",
             "e4lut": "ppo",
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
        cond = "nominal" if name.endswith("_nom") else "peak"
        base = name[:-4] if name.endswith("_nom") else name
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


def agg(rows, key):
    vals = [r[key] for r in rows]
    return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)


def m(tag, arm, cond, key):
    rows = DATA[tag].get(arm, {}).get(cond)
    return agg(rows, key)[0] if rows else float("nan")


ARMS = ["bl_oracle_best_feasible_evidence", "bl_always_cache", "bl_random",
        "proposed", "no_lagrangian", "fixed_penalty", "e4lut"]

for cond in ("peak", "nominal"):
    print("=" * 104)
    print(f"3-GEN COMPARISON -- {cond} condition")
    print("=" * 104)
    hdr = f"{'arm':32} {'gen':6} {'eps':>6} {'semSucc':>8} {'qViol':>7} {'taskSucc':>9} {'acc':>6} {'cacheR':>7} {'ddlVio':>7}"
    print(hdr)
    print("-" * len(hdr))
    for arm in ARMS:
        for tag, _ in GENS:
            rows = DATA[tag].get(arm, {}).get(cond)
            if not rows:
                continue
            print(f"{arm:32} {tag:6} "
                  f"{agg(rows,'average_epsilon_k')[0]:6.3f} "
                  f"{agg(rows,'semantic_success_rate')[0]:8.3f} "
                  f"{agg(rows,'quality_violation_rate')[0]:7.3f} "
                  f"{agg(rows,'task_success_rate')[0]:9.3f} "
                  f"{agg(rows,'average_accuracy')[0]:6.3f} "
                  f"{agg(rows,'semantic_path_cache_ratio')[0]:7.3f} "
                  f"{agg(rows,'deadline_violation_rate')[0]:7.3f}")
        print()

# lambda trajectory shape (v2 peak training)
print("=" * 104)
print("LAMBDA_QUALITY_CRITICAL TRAJECTORY (v2 peak training)")
print("=" * 104)
print(f"{'arm/seed':18} {'first':>7} {'last':>7} {'min':>7} {'max':>7} {'peakEp':>7} {'dropAfter':>9} {'pinned?':>8} {'costMean':>8}")


def lam(arm_dir: Path):
    f = arm_dir / "ppo_lambda_trace.csv"
    if not f.exists():
        return None
    q, cost = [], []
    for row in csv.DictReader(f.open()):
        try:
            q.append(float(row["lambda_quality_critical"]))
            cost.append(float(row["quality_cost_critical"]))
        except (KeyError, ValueError):
            pass
    if not q:
        return None
    pk = max(range(len(q)), key=lambda i: q[i])
    drop = q[pk] - q[-1]
    span = max(q) - min(q)
    # "pinned/diverged": climbed but never meaningfully descended -> monotone
    # ramp to the dual ceiling (infeasible-constraint signature). Non-pinned
    # requires an interior peak with a real turn-around (drop >= 15% of span).
    pinned = span < 1e-3 or drop < 0.15 * span
    return dict(first=q[0], last=q[-1], mn=min(q), mx=max(q), pk=pk,
                drop=drop, pinned=pinned, costmean=mean(cost))


for arm in ["proposed", "no_lagrangian", "fixed_penalty", "e4lut"]:
    for seed in (0, 1, 2):
        sh = lam(ROOT / "eps_recal_v2" / f"{arm}_{seed}")
        if sh is None:
            continue
        print(f"{arm+'_'+str(seed):18} {sh['first']:7.3f} {sh['last']:7.3f} {sh['mn']:7.3f} "
              f"{sh['mx']:7.3f} {sh['pk']:7d} {sh['drop']:9.3f} {str(sh['pinned']):>8} {sh['costmean']:8.3f}")

print("\n" + "=" * 104)
print("V2 SUCCESS CRITERIA")
print("=" * 104)
checks = []
o = m("v2", "bl_oracle_best_feasible_evidence", "peak", "semantic_success_rate")
checks.append(("PEAK oracle semSucc >= 0.85 (construction guarantee)", o, o >= 0.85))
pc = m("v2", "proposed", "peak", "semantic_path_cache_ratio")
pa = m("v2", "proposed", "peak", "average_accuracy")
checks.append(("PEAK proposed cache ratio < 0.40", pc, pc < 0.40))
checks.append(("PEAK proposed average_accuracy >= 0.45", pa, pa >= 0.45))
turned = []
for seed in (0, 1, 2):
    sh = lam(ROOT / "eps_recal_v2" / f"proposed_{seed}")
    if sh:
        turned.append(not sh["pinned"])
checks.append(("PEAK proposed lambda_critical not pinned (>=2/3 seeds)", sum(turned), sum(turned) >= 2))
ns = m("v2", "proposed", "nominal", "semantic_success_rate")
nt = m("v2", "proposed", "nominal", "task_success_rate")
checks.append(("NOMINAL proposed semSucc >= 0.90", ns, ns >= 0.90))
checks.append(("NOMINAL proposed task success >= 0.75", nt, nt >= 0.75))
npass = 0
for name, val, ok in checks:
    npass += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  (obs {val:.4f})")
print(f"\n{npass}/{len(checks)} criteria PASS")
