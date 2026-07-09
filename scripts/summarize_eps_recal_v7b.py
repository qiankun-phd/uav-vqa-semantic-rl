#!/usr/bin/env python
"""Task #36 v7b: mission-aligned reward WIRED INTO THE LEARNER + judgment.

v7b re-runs the v7 matrix after fixing the reward-wiring bug: mission_aligned
now flows into the CONTROLLER reward (the two-timescale PPO gradient), not
just the env-layer raw_reward log.  Reads outputs/rl/eps_recal_v7b.  Same 13
v6/v7 criteria (the 6 that hung in v7 -- 3a cacheR, 3b rejectR, 3c mission,
3d acc, 4 lambda_QC, 5b nominal task -- are the ones the wiring should move).

Original v7 docstring:
Task #35/#36 v7: mission-aligned reward + link-guard matrix summary + judgment.

Reads outputs/rl/eps_recal_v7 (v6 comm-window gates + mission_aligned reward +
fair-share reference + LUT outage guard, delta_esc peak 0.155 / nominal 0.05
from outputs/rl/eps_v7_calib_corrected.json) and reports the v6 DOUBLE metric
(mission_success = quality AND deadline, on the admitted set) alongside the
quality-only semSucc kept for cross-generation comparison, plus escalation stats.

v6 judgment (mission on ADMITTED = non-escalated; all PASS => PASS):
  1. PEAK oracle mission(admitted) >= 0.85 AND admitted deadline-violation ~ 0
  2. PEAK proposed critical-escalation in [delta_esc +- 0.05] AND escalated tasks
     carry zero silent quality violations; NOMINAL escalation < 0.15
  3. PEAK proposed cacheR<0.40 AND non-escalated-rejectR<0.20 AND
     mission(admitted) >= 0.6*oracle AND acc >= 0.7*oracle_acc
  4. lambda_quality_critical (or a deadline-class channel) not pinned/zero on
     >=2/3 seeds (terminal in (0.1, 0.9*lambda_max) OR cost in [delta-0.02, delta+0.05])
  5. NOMINAL proposed mission(admitted) >= 0.85*oracle AND task >= 0.75
  6. hygiene: spec_attainable non-constant under nominal; cache-ban engages (a
     cache-only critical event judged non-compliant appears in the trace).
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

BASE = Path.home() / "phd_research/vqa_main/outputs/rl"
ROOT = BASE / "eps_recal_v7b"
CALIB = BASE / "eps_v7_calib_corrected.json"
LAMBDA_MAX = 20.0
QUALITY_LIMIT_CRIT = 0.02

METRICS = ["average_epsilon_k", "semantic_success_rate", "quality_violation_rate",
           "task_success_rate", "average_accuracy", "semantic_path_cache_ratio",
           "deadline_violation_rate", "reject_ratio",
           "escalation_rate", "critical_escalation_rate", "spec_attainable_rate",
           "admitted_semantic_success_rate", "admitted_average_accuracy",
           "admitted_quality_violation_rate",
           "mission_success_rate", "admitted_mission_success_rate",
           "admitted_deadline_violation_rate"]
POLICY_OF = {"proposed": "ppo", "no_lagrangian": "ppo", "fixed_penalty": "ppo", "e4lut": "ppo",
             "bl_oracle_best_feasible_evidence": "oracle_best_feasible_evidence",
             "bl_oracle_escalation_aware": "oracle_escalation_aware",
             "bl_semantic_greedy": "semantic_greedy", "bl_always_cache": "always_cache",
             "bl_random": "random"}


def _non_escalated_reject(d: Path, policy: str) -> float:
    f = d / "v1_9_resource_alloc_rollout.csv"
    if not f.exists():
        return float("nan")
    n = bad = 0
    for r in csv.DictReader(f.open()):
        if str(r.get("policy", "")) != policy:
            continue
        n += 1
        if str(r.get("rejected", "")).lower() in ("true", "1") and str(r.get("escalated", "")).lower() not in ("true", "1"):
            bad += 1
    return bad / n if n else float("nan")


def _cache_ban_events(d: Path, policy: str) -> int:
    """Hygiene #6: count trace events where a cache-only (s0) critical/high task
    is judged non-compliant (quality_violation True) -- proof the ban engages."""
    f = d / "v1_9_resource_alloc_rollout.csv"
    if not f.exists():
        return 0
    n = 0
    for r in csv.DictReader(f.open()):
        if str(r.get("policy", "")) != policy:
            continue
        if (str(r.get("semantic_path", "")) == "cache"
                and str(r.get("risk_level", "")) in ("critical", "high")
                and str(r.get("quality_violation", "")).lower() in ("true", "1")):
            n += 1
    return n


def _spec_attainable_nonconstant(d: Path, policy: str) -> bool:
    """Hygiene #6: spec_attainable is a non-constant distribution (both True and
    False appear) in the trace."""
    f = d / "v1_9_resource_alloc_rollout.csv"
    if not f.exists():
        return False
    seen = set()
    for r in csv.DictReader(f.open()):
        if str(r.get("policy", "")) != policy:
            continue
        seen.add(str(r.get("spec_attainable", "")).lower() in ("true", "1"))
        if len(seen) > 1:
            return True
    return False


def _read_row(d: Path, policy: str):
    f = d / "v1_9_resource_alloc_results.csv"
    if not f.exists():
        return None
    for row in csv.DictReader(f.open()):
        if str(row.get("policy", "")) == policy:
            out = {k: float(row.get(k, 0.0) or 0.0) for k in METRICS}
            out["non_escalated_reject_ratio"] = _non_escalated_reject(d, policy)
            out["cache_ban_events"] = _cache_ban_events(d, policy)
            out["spec_attainable_nonconstant"] = _spec_attainable_nonconstant(d, policy)
            return out
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


DATA = collect(ROOT)
CAL = json.loads(CALIB.read_text()) if CALIB.exists() else {}


def agg(rows, key):
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), 0.0)


def m(arm, cond, key):
    rows = DATA.get(arm, {}).get(cond)
    return agg(rows, key)[0] if rows else float("nan")


ARMS = ["bl_oracle_best_feasible_evidence", "bl_oracle_escalation_aware",
        "bl_always_cache", "bl_random",
        "proposed", "no_lagrangian", "fixed_penalty", "e4lut"]

# Criterion-1 ceiling: the escalation-aware oracle (honours the certificate);
# fall back to the plain best-feasible oracle if the aware arm is absent.
ORACLE_CEIL = ("bl_oracle_escalation_aware"
               if DATA.get("bl_oracle_escalation_aware") else "bl_oracle_best_feasible_evidence")

for cond in ("peak", "nominal", "nomtrain"):
    print("=" * 140)
    print(f"V6 MATRIX -- {cond} condition  (mission = quality AND deadline, on admitted)")
    print("=" * 140)
    hdr = (f"{'arm':34} {'mission':>8} {'admMiss':>8} {'admDdlV':>8} {'semSucc':>8} {'admSucc':>8} "
           f"{'task':>6} {'acc':>6} {'admAcc':>7} {'cacheR':>7} {'neRejR':>7} {'escR':>6} {'critEsc':>8} {'specAtt':>8}")
    print(hdr)
    print("-" * len(hdr))
    for arm in ARMS:
        rows = DATA.get(arm, {}).get(cond)
        if not rows:
            continue
        g = lambda k: agg(rows, k)[0]
        print(f"{arm:34} {g('mission_success_rate'):8.3f} {g('admitted_mission_success_rate'):8.3f} "
              f"{g('admitted_deadline_violation_rate'):8.3f} {g('semantic_success_rate'):8.3f} "
              f"{g('admitted_semantic_success_rate'):8.3f} {g('task_success_rate'):6.3f} "
              f"{g('average_accuracy'):6.3f} {g('admitted_average_accuracy'):7.3f} "
              f"{g('semantic_path_cache_ratio'):7.3f} {g('non_escalated_reject_ratio'):7.3f} "
              f"{g('escalation_rate'):6.3f} {g('critical_escalation_rate'):8.3f} {g('spec_attainable_rate'):8.3f}")
    print()


def lam(arm_dir: Path):
    f = arm_dir / "ppo_lambda_trace.csv"
    if not f.exists():
        return None
    q, cost, esc, lesc, ld, ldcost = [], [], [], [], [], []
    for row in csv.DictReader(f.open()):
        try:
            q.append(float(row["lambda_quality_critical"]))
            cost.append(float(row.get("quality_cost_critical", 0.0)))
            esc.append(float(row.get("escalation_cost", 0.0)))
            lesc.append(float(row.get("lambda_escalation", 0.0)))
            ld.append(float(row.get("lambda_deadline_critical", 0.0)))
            ldcost.append(float(row.get("deadline_cost_critical", row.get("q_deadline", 0.0))))
        except (KeyError, ValueError):
            pass
    if not q:
        return None
    span = max(q) - min(q)
    pk = max(range(len(q)), key=lambda i: q[i])
    pinned = span < 1e-3 or (q[pk] - q[-1]) < 0.15 * span
    return dict(first=q[0], last=q[-1], mn=min(q), mx=max(q), pinned=pinned,
                costmean=mean(cost[-50:] if len(cost) >= 50 else cost),
                escmean=mean(esc[-50:] if len(esc) >= 50 else esc),
                lesc_last=lesc[-1], ld_last=ld[-1] if ld else 0.0,
                ldmax=max(ld) if ld else 0.0)


print("=" * 140)
print("LAMBDA / ESCALATION TRAJECTORY (v7b peak training, terminal-window means)")
print(f"{'arm/seed':16} {'lamQC_first':>11} {'lamQC_last':>10} {'lamQC_max':>10} {'pinned?':>8} "
      f"{'qCostCrit':>9} {'lamDdlC_last':>12} {'escCost':>8} {'lamEsc':>7}")
for arm in ["proposed", "no_lagrangian", "fixed_penalty", "e4lut"]:
    for seed in (0, 1, 2):
        sh = lam(ROOT / f"{arm}_{seed}")
        if sh is None:
            continue
        print(f"{arm+'_'+str(seed):16} {sh['first']:11.3f} {sh['last']:10.3f} {sh['mx']:10.3f} "
              f"{str(sh['pinned']):>8} {sh['costmean']:9.3f} {sh['ld_last']:12.3f} {sh['escmean']:8.3f} {sh['lesc_last']:7.3f}")

print("\n" + "=" * 140)
print("V6 SUCCESS CRITERIA")
print("=" * 140)
delta_peak = float(CAL.get("delta_esc_peak", 0.155))
spec_unattain = float(CAL.get("peak_decomposition", {}).get("spec_unattainable", 0.105))
checks = []

# 1. PEAK oracle mission(admitted) >= 0.85 AND admitted deadline-violation ~ 0
# (ceiling = escalation-aware oracle, which honours the spec-attainability cert).
o_m = m(ORACLE_CEIL, "peak", "admitted_mission_success_rate")
o_ddl = m(ORACLE_CEIL, "peak", "admitted_deadline_violation_rate")
o_acc = m(ORACLE_CEIL, "peak", "admitted_average_accuracy")
print(f"(criterion-1 oracle ceiling = {ORACLE_CEIL})")
checks.append(("1a. PEAK oracle mission(admitted) >= 0.85", o_m, o_m >= 0.85))
# v7 task criterion 1b: admitted deadline-violation must NOT worsen from the
# v6 escalation-aware oracle edge (0.084); persisting at ~0.084 is allowed
# per the v7 brief ("v6 0.084 marginal, hold-even OK, do not worsen").
checks.append(("1b. PEAK oracle admitted deadline-violation <= 0.085 (v6 0.084 hold-even)", o_ddl, o_ddl <= 0.085))

# 2. escalation budget + nominal escalation
pesc = m("proposed", "peak", "critical_escalation_rate")
lo, hi = max(0.0, delta_peak - 0.05), min(1.0, delta_peak + 0.05)
checks.append((f"2a. PEAK proposed crit-escalation in [{lo:.2f},{hi:.2f}] (delta_esc {delta_peak:.2f})",
               pesc, lo <= pesc <= hi))
nesc = m("proposed", "nominal", "escalation_rate")
checks.append(("2b. NOMINAL proposed escalation < 0.15", nesc, nesc < 0.15))

# 3. proposed peak behaviour
pc = m("proposed", "peak", "semantic_path_cache_ratio")
pr = m("proposed", "peak", "non_escalated_reject_ratio")
pa = m("proposed", "peak", "admitted_average_accuracy")
pm = m("proposed", "peak", "admitted_mission_success_rate")
checks.append(("3a. PEAK proposed cacheR < 0.40", pc, pc < 0.40))
checks.append(("3b. PEAK proposed non-escalated-rejectR < 0.20", pr, pr < 0.20))
checks.append((f"3c. PEAK proposed mission(admitted) >= 0.6*oracle ({0.6*o_m:.3f})", pm, pm >= 0.6 * o_m))
checks.append((f"3d. PEAK proposed acc(admitted) >= 0.7*oracle_acc ({0.7*o_acc:.3f})", pa, pa >= 0.7 * o_acc))

# 4. lambda channel NON-PINNED, NON-ZERO (quality_critical OR deadline_critical)
# on >= 2/3 seeds.  "non-pinned" = terminal in (0.1, 0.9*lambda_max) AND not
# stuck at the max; OR the quality-critical cost sits in the target band.  A
# lambda that saturates at ~lambda_max is a FAIL (the v5/v6 pin), so ldmax alone
# does not rescue it.
active = []
for seed in (0, 1, 2):
    sh = lam(ROOT / f"proposed_{seed}")
    if sh:
        band_q = (0.1 < sh["last"] < 0.9 * LAMBDA_MAX) and not sh["pinned"]
        cost_q = (QUALITY_LIMIT_CRIT - 0.02) <= sh["costmean"] <= (QUALITY_LIMIT_CRIT + 0.05)
        band_d = (0.1 < sh["ld_last"] < 0.9 * LAMBDA_MAX)
        active.append(band_q or cost_q or band_d)
checks.append(("4. lambda quality/deadline-critical NON-pinned, active (>=2/3 seeds)", sum(active), sum(active) >= 2))

# 5. nominal proposed vs the NOMINAL oracle ceiling (condition-consistent:
# comparing a nominal-condition policy against the peak ceiling conflates the
# two operating conditions).
o_m_nom = m(ORACLE_CEIL, "nominal", "admitted_mission_success_rate")
nm = m("proposed", "nominal", "admitted_mission_success_rate")
nt = m("proposed", "nominal", "task_success_rate")
checks.append((f"5a. NOMINAL proposed mission(admitted) >= 0.85*nominal-oracle ({0.85*o_m_nom:.3f})",
               nm, nm >= 0.85 * o_m_nom if o_m_nom == o_m_nom else False))
checks.append(("5b. NOMINAL proposed task success >= 0.75", nt, nt >= 0.75))

# 6. hygiene
sa_nc = m("proposed", "nominal", "spec_attainable_nonconstant")
ban_ev = m("proposed", "peak", "cache_ban_events")
checks.append(("6a. spec_attainable non-constant under nominal", sa_nc, bool(sa_nc)))
checks.append(("6b. cache-ban engages (>=1 cache-only critical non-compliant event, peak)", ban_ev, (ban_ev or 0) >= 1))

npass = 0
for name, val, ok in checks:
    npass += bool(ok)
    v = f"{val:.4f}" if isinstance(val, float) else str(val)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  (obs {v})")
print(f"\n{npass}/{len(checks)} criteria PASS  ->  {'PASS' if npass == len(checks) else 'FAIL'}")
