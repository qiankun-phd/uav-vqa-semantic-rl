#!/usr/bin/env python
"""Paper II: single table-ready numbers file for the v8 frozen environment.

Aggregates
  * outputs/rl/eps_recal_v8        main-table arms (36 arms, task #37)
  * outputs/rl/flat_ppo_v8         flat single-head PPO baseline (P1)
  * outputs/rl/m3_ablation_v8      dual-machinery ablations
  * outputs/rl/m4_generalization_v8  eval-only generalization sweeps

into outputs/rl/paper2_numbers_v8.json with mean/std over seeds in a format
that can be pasted straight into the paper tables:
  main_table.{peak,nominal}.<arm>.<metric> = {mean, std, n}
  constraint_table.{peak,nominal}.<arm>    = per-channel realized eval cost
                                             vs limit (7 dual channels +
                                             escalation budget)
  m3_ablation.<arm>                        = peak metrics + lambda_QC terminal
  m4_generalization.<axis>.<point>.<arm>   = sweep metrics
  sample_efficiency.<arm>                  = episodes-to-90% terminal return
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path.home() / "phd_research/vqa_main"
RL = ROOT / "outputs/rl"
V8 = RL / "eps_recal_v8"
FLAT = RL / "flat_ppo_v8"
M3 = RL / "m3_ablation_v8"
M4 = RL / "m4_generalization_v8"
OUT = RL / "paper2_numbers_v8.json"

SEEDS = (0, 1, 2)

RESULT_METRICS = [
    "admitted_mission_success_rate",
    "mission_success_rate",
    "task_success_rate",
    "admitted_average_accuracy",
    "average_accuracy",
    "semantic_success_rate",
    "admitted_semantic_success_rate",
    "semantic_path_cache_ratio",
    "semantic_path_token_ratio",
    "semantic_path_image_ratio",
    "reject_ratio",
    "escalation_rate",
    "critical_escalation_rate",
    "spec_attainable_rate",
    "quality_violation_rate",
    "admitted_quality_violation_rate",
    "deadline_violation_rate",
    "admitted_deadline_violation_rate",
    "airspace_conflict_rate",
    "utm_constraint_violation_rate",
    "average_delay",
    "average_energy",
]

CHANNELS = {
    "quality_normal": 0.05,
    "quality_critical": 0.02,
    "deadline_normal": 0.05,
    "deadline_critical": 0.02,
    "conflict": 0.08,
    "battery": 0.0,
    "gpu": 0.0,
}


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1")


def read_results_row(d: Path, policy: str) -> dict[str, float] | None:
    f = d / "v1_9_resource_alloc_results.csv"
    if not f.exists():
        return None
    for r in csv.DictReader(f.open()):
        if str(r.get("policy", "")) == policy:
            out = {}
            for k in RESULT_METRICS:
                try:
                    out[k] = float(r.get(k, "nan"))
                except ValueError:
                    out[k] = float("nan")
            return out
    return None


def rollout_derived(d: Path, policy: str, delta_esc: float | None = None) -> dict[str, float]:
    """Non-escalated reject rate, conflict OR-rate, per-channel eval costs.

    Channel costs use the IMPLEMENTED constraint semantics (the quantity the
    dual compares against d_i): numerator = joint event
    1{violation AND risk-class}, denominator = ADMITTED (non-escalated) steps.
    Escalated steps carry no violation by construction and are excluded from
    the admitted set of Eq. (cmdp); under zero escalation this equals the
    training-time all-steps mean exactly.
    """
    f = d / "v1_9_resource_alloc_rollout.csv"
    out: dict[str, float] = {}
    if not f.exists():
        return out
    n = 0
    n_adm = 0
    ne_rej = 0
    conflict = 0
    esc = 0
    crit_n = 0
    ch = {k: 0 for k in CHANNELS}  # joint-event counts on the admitted set
    for r in csv.DictReader(f.open()):
        if str(r.get("policy", "")) != policy:
            continue
        n += 1
        escd = _truthy(r.get("escalated", ""))
        esc += int(escd)
        if _truthy(r.get("rejected", "")) and not escd:
            ne_rej += 1
        conf = _truthy(r.get("airspace_conflict", "")) or _truthy(r.get("utm_constraint_violation", ""))
        conflict += int(conf)
        risk = str(r.get("risk_level", "normal"))
        crit = risk in ("critical", "high")
        crit_n += int(crit)
        if escd:
            continue
        n_adm += 1
        q = _truthy(r.get("quality_violation", ""))
        dl = _truthy(r.get("deadline_violation", ""))
        ch["quality_critical" if crit else "quality_normal"] += int(q)
        ch["deadline_critical" if crit else "deadline_normal"] += int(dl)
        ch["conflict"] += int(conf)
        ch["battery"] += int(_truthy(r.get("battery_violation", "")))
        ch["gpu"] += int(_truthy(r.get("resource_violation", "")))
    if n:
        out["non_escalated_reject_ratio"] = ne_rej / n
        out["conflict_rate"] = conflict / n
        out["escalation_rate_check"] = esc / n
        out["critical_share"] = crit_n / n
        for k, c in ch.items():
            out[f"cost_{k}"] = (c / n_adm) if n_adm else float("nan")
        if delta_esc is not None:
            out["escalation_budget"] = float(delta_esc)
    return out


def lambda_terminal(d: Path) -> dict[str, float]:
    f = d / "ppo_training_trace.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(f.open()))
    if not rows:
        return {}
    last = rows[-1]
    tail = rows[-50:]
    out = {}
    for k in ("lambda_quality_critical", "lambda_quality_normal", "lambda_conflict",
              "lambda_deadline_critical", "lambda_escalation"):
        try:
            out[f"terminal_{k}"] = float(last.get(k, "nan"))
        except ValueError:
            out[f"terminal_{k}"] = float("nan")
    try:
        out["tail50_quality_cost_critical"] = mean(float(r["quality_cost_critical"]) for r in tail)
    except (ValueError, KeyError):
        pass
    return out


def episodes_to_90pct(d: Path) -> float:
    """First episode where the 9-ep-smoothed return reaches 90% of the span
    min->terminal (robust to negative returns)."""
    f = d / "ppo_training_trace.csv"
    if not f.exists():
        return float("nan")
    rows = list(csv.DictReader(f.open()))
    ret = [float(r["raw_return"]) for r in rows]
    if len(ret) < 20:
        return float("nan")
    w = 9
    sm = [mean(ret[max(0, i - w + 1): i + 1]) for i in range(len(ret))]
    terminal = mean(sm[-25:])
    lo = min(sm)
    if terminal <= lo:
        return float("nan")
    thresh = lo + 0.9 * (terminal - lo)
    for i, v in enumerate(sm):
        if v >= thresh:
            return float(i)
    return float("nan")


def agg(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({k for d in per_seed for k in d})
    out = {}
    for k in keys:
        vals = [d[k] for d in per_seed if k in d and d[k] == d[k]]
        if not vals:
            continue
        out[k] = {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}
    return out


def arm_dirs(root: Path, arm: str, suffix: str = "", seeds=SEEDS) -> list[Path]:
    return [root / f"{arm}_{s}{suffix}" for s in seeds if (root / f"{arm}_{s}{suffix}").exists()]


def collect_arm(root: Path, arm: str, policy: str, suffix: str = "", seeds=SEEDS,
                delta_esc: float | None = None, with_lambda: bool = False) -> dict:
    per_seed = []
    lam = []
    for d in arm_dirs(root, arm, suffix, seeds):
        row = read_results_row(d, policy)
        if row is None:
            continue
        row.update(rollout_derived(d, policy, delta_esc))
        per_seed.append(row)
        if with_lambda:
            lam.append({"dir": d.name, **lambda_terminal(d), "episodes_to_90pct_return": episodes_to_90pct(d)})
    out = {"metrics": agg(per_seed), "seeds": len(per_seed)}
    if with_lambda and lam:
        out["per_seed_dual"] = lam
    return out


DELTA_PEAK, DELTA_NOM = 0.155, 0.05

result: dict = {"meta": {
    "env": "v8 frozen (comm_window + attainability_v5 + spec_attainable esc + entry_v2 cache + "
           "forbidden critical-cache + lut_v5 + mission_aligned + fair_share + LUT outage guard + "
           "lambda_max_quality 8 + dual warmup 150)",
    "delta_esc": {"peak": DELTA_PEAK, "nominal": DELTA_NOM},
    "seeds": list(SEEDS),
    "train_episodes": 500, "eval_episodes": 50, "tasks_per_episode": 20,
    "channel_limits": CHANNELS,
}}

# ------------------------------------------------------------- main table
main: dict = {"peak": {}, "nominal": {}}
for arm, pol in (("proposed", "ppo"), ("no_lagrangian", "ppo"), ("fixed_penalty", "ppo"), ("e4lut", "ppo")):
    seeds = (0, 1) if arm == "e4lut" else SEEDS
    main["peak"][arm] = collect_arm(V8, arm, pol, seeds=seeds, delta_esc=DELTA_PEAK, with_lambda=True)
    main["nominal"][arm] = collect_arm(V8, arm, pol, suffix="_nom", seeds=seeds, delta_esc=DELTA_NOM)
main["nominal"]["proposed_nomtrain"] = collect_arm(V8, "proposed", "ppo", suffix="_nomtrain",
                                                   delta_esc=DELTA_NOM, with_lambda=True)
for bl, pol in (("bl_oracle_escalation_aware", "oracle_escalation_aware"),
                ("bl_oracle_best_feasible_evidence", "oracle_best_feasible_evidence"),
                ("bl_semantic_greedy", "semantic_greedy"),
                ("bl_always_cache", "always_cache")):
    main["peak"][bl] = collect_arm(V8, bl, pol, seeds=(0,), delta_esc=DELTA_PEAK)
    main["nominal"][bl] = collect_arm(V8, bl, pol, suffix="_nom", seeds=(0,), delta_esc=DELTA_NOM)
main["peak"]["bl_random"] = collect_arm(V8, "bl_random", "random", delta_esc=DELTA_PEAK)
if FLAT.exists():
    main["peak"]["flat_ppo"] = collect_arm(FLAT, "flat_ppo", "ppo", delta_esc=DELTA_PEAK, with_lambda=True)
    main["nominal"]["flat_ppo"] = collect_arm(FLAT, "flat_ppo_nom", "ppo", delta_esc=DELTA_NOM)
result["main_table"] = main

# -------------------------------------------------- constraint satisfaction
# per-channel realized eval cost vs limit for the headline arms.
constraint = {}
for cond, suffix, delta in (("peak", "", DELTA_PEAK), ("nominal", "_nom", DELTA_NOM)):
    constraint[cond] = {}
    for arm in ("proposed",):
        block = main[cond].get(arm if suffix != "_nomtrain" else "proposed_nomtrain", {})
        met = block.get("metrics", {})
        rows = {}
        for chkey, limit in CHANNELS.items():
            cell = met.get(f"cost_{chkey}")
            if cell:
                rows[chkey] = {"cost": cell["mean"], "std": cell["std"], "limit": limit,
                               "satisfied": bool(cell["mean"] <= limit + 1e-9) if limit > 0 else None}
        esc = met.get("critical_escalation_rate")
        if esc:
            rows["escalation_budget"] = {"cost": esc["mean"], "std": esc["std"], "limit": delta,
                                         "satisfied": bool(esc["mean"] <= delta + 1e-9)}
        constraint[cond][arm] = rows
constraint["nominal"]["proposed_nomtrain"] = {}
met = main["nominal"].get("proposed_nomtrain", {}).get("metrics", {})
for chkey, limit in CHANNELS.items():
    cell = met.get(f"cost_{chkey}")
    if cell:
        constraint["nominal"]["proposed_nomtrain"][chkey] = {
            "cost": cell["mean"], "std": cell["std"], "limit": limit,
            "satisfied": bool(cell["mean"] <= limit + 1e-9) if limit > 0 else None}
esc = met.get("critical_escalation_rate")
if esc:
    constraint["nominal"]["proposed_nomtrain"]["escalation_budget"] = {
        "cost": esc["mean"], "std": esc["std"], "limit": DELTA_NOM,
        "satisfied": bool(esc["mean"] <= DELTA_NOM + 1e-9)}
result["constraint_table"] = constraint

# ------------------------------------------------------------- M3 ablation
if M3.exists():
    m3 = {}
    for arm in ("quality_off", "warm_start", "no_slow_head"):
        m3[arm] = collect_arm(M3, arm, "ppo", delta_esc=DELTA_PEAK, with_lambda=True)
    m3["reference_proposed_peak"] = {
        "metrics": main["peak"]["proposed"]["metrics"],
        "per_seed_dual": main["peak"]["proposed"].get("per_seed_dual", []),
    }
    result["m3_ablation"] = m3

# ----------------------------------------------------- M4 generalization
if M4.exists():
    m4: dict = {}
    for axis_dir in sorted(p for p in M4.iterdir() if p.is_dir()):
        axis = axis_dir.name
        m4[axis] = {}
        tags = sorted({d.name.rsplit("_", 1)[0] for d in axis_dir.iterdir() if d.is_dir()})
        # tags look like: n2_proposed, t12_no_lagrangian, x4_bl_semantic_greedy ...
        points = sorted({t.split("_", 1)[0] for t in tags})
        for pt in points:
            m4[axis][pt] = {}
            for arm, pol in (("proposed", "ppo"), ("no_lagrangian", "ppo"), ("fixed_penalty", "ppo")):
                block = collect_arm(axis_dir, f"{pt}_{arm}", pol, delta_esc=None)
                if block["seeds"]:
                    m4[axis][pt][arm] = block
            for bl, pol in (("bl_semantic_greedy", "semantic_greedy"),
                            ("bl_oracle_escalation_aware", "oracle_escalation_aware")):
                block = collect_arm(axis_dir, f"{pt}_{bl}", pol, seeds=(0,))
                if block["seeds"]:
                    m4[axis][pt][bl] = block
    result["m4_generalization"] = m4

# -------------------------------------------------------- sample efficiency
se = {}
for label, root, arm, suffix in (
    ("proposed_peak", V8, "proposed", ""),
    ("no_lagrangian_peak", V8, "no_lagrangian", ""),
    ("fixed_penalty_peak", V8, "fixed_penalty", ""),
    ("e4lut_peak", V8, "e4lut", ""),
    ("proposed_nomtrain", V8, "proposed", "_nomtrain"),
    ("flat_ppo_peak", FLAT, "flat_ppo", ""),
    ("m3_quality_off", M3, "quality_off", ""),
    ("m3_warm_start", M3, "warm_start", ""),
    ("m3_no_slow_head", M3, "no_slow_head", ""),
):
    if not root.exists():
        continue
    vals = [episodes_to_90pct(d) for d in arm_dirs(root, arm, suffix, (0, 1, 2))]
    vals = [v for v in vals if v == v]
    if vals:
        se[label] = {"episodes_to_90pct_return": {"mean": mean(vals),
                                                  "std": pstdev(vals) if len(vals) > 1 else 0.0,
                                                  "n": len(vals)},
                     "train_episodes": 500}
result["sample_efficiency"] = se

OUT.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print("wrote", OUT)

# quick console digest
def cell(cond, arm, key):
    c = result["main_table"].get(cond, {}).get(arm, {}).get("metrics", {}).get(key)
    return f"{c['mean']:.3f}+-{c['std']:.3f}" if c else "--"

print("\n== main table digest (admitted mission / task / cacheR / critEsc) ==")
for cond in ("peak", "nominal"):
    for arm in result["main_table"][cond]:
        print(f"{cond:8s} {arm:34s} "
              f"admMiss {cell(cond, arm, 'admitted_mission_success_rate'):>16s}  "
              f"task {cell(cond, arm, 'task_success_rate'):>16s}  "
              f"cacheR {cell(cond, arm, 'semantic_path_cache_ratio'):>16s}  "
              f"critEsc {cell(cond, arm, 'critical_escalation_rate'):>16s}")
