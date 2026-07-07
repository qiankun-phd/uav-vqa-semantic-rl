#!/usr/bin/env python3
"""Paper-2 experiment matrix -- batch 4: figures + tables + summary.

Consumes:
  outputs/rl/matrix_v1/{arm}_{seed}[,_nom]/   batch-2 training + dual-condition eval
  outputs/rl/matrix_v1/sweep/{axis}/{point}/  batch-3 zero-shot sweeps
  outputs/rl/ab_bubbles_v3/                   v3 ablation runs (Table II)

Produces (PNG + PDF, markdown + csv):
  outputs/rl/matrix_v1/figures/fig1_convergence, fig2_multiscale, fig3_uav,
    fig4_load, fig5_snr, fig6_violation_pareto, fig7_zeroshot
  outputs/rl/matrix_v1/tables/table1_main, table2_ablation, table3_sample_efficiency
  outputs/rl/matrix_v1/matrix_summary.md
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "outputs" / "rl" / "matrix_v1"
SWEEP = MATRIX / "sweep"
V3 = ROOT / "outputs" / "rl" / "ab_bubbles_v3"
FIG_DIR = MATRIX / "figures"
TAB_DIR = MATRIX / "tables"

SEEDS = (0, 1, 2)
LEARN_ARMS = ("proposed", "no_lagrangian", "fixed_penalty", "flat_ppo", "service_only")
BASELINE_ARMS = {
    "bl_semantic_greedy": ("semantic_greedy", (0,)),
    "bl_always_cache": ("always_cache", (0,)),
    "bl_oracle_best_feasible_evidence": ("oracle_best_feasible_evidence", (0,)),
    "bl_random": ("random", SEEDS),
}
ARM_LABEL = {
    "proposed": "Proposed (constrained two-timescale)",
    "no_lagrangian": "No-Lagrangian (unconstrained)",
    "fixed_penalty": "Fixed penalty (frozen duals)",
    "flat_ppo": "Flat PPO (conventional)",
    "service_only": "Service-only PPO (discrete)",
    "bl_semantic_greedy": "Semantic greedy",
    "bl_always_cache": "Always cache",
    "bl_oracle_best_feasible_evidence": "Oracle (best feasible)",
    "bl_random": "Random",
}
ARM_COLOR = {
    "proposed": "#d62728",
    "no_lagrangian": "#1f77b4",
    "fixed_penalty": "#2ca02c",
    "flat_ppo": "#9467bd",
    "service_only": "#8c564b",
    "bl_semantic_greedy": "#ff7f0e",
    "bl_always_cache": "#7f7f7f",
    "bl_oracle_best_feasible_evidence": "#000000",
    "bl_random": "#bcbd22",
}
ARM_MARKER = {
    "proposed": "o", "no_lagrangian": "s", "fixed_penalty": "^",
    "flat_ppo": "v", "service_only": "D", "bl_semantic_greedy": "P",
    "bl_always_cache": "X", "bl_oracle_best_feasible_evidence": "*", "bl_random": "x",
}
TABLE_METRICS = (
    "average_accuracy", "semantic_success_rate", "task_success_rate",
    "airspace_conflict_rate", "deadline_violation_rate", "average_delay",
    "average_payload_kb", "average_reward",
)
CONFLICT_LIMIT = 0.08
SMOOTH_WINDOW = 25
PLATEAU_TAIL = 100


# ---------------------------------------------------------------- io helpers
def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def result_row(run_dir: Path, policy: str) -> dict[str, float] | None:
    for row in read_rows(run_dir / "v1_9_resource_alloc_results.csv"):
        if str(row.get("policy", "")) == policy:
            return {k: _f(v) for k, v in row.items() if _isnum(v)}
    return None


def trace_series(run_dir: Path, column: str) -> list[float]:
    return [_f(row.get(column, 0.0)) for row in read_rows(run_dir / "ppo_training_trace.csv")]


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _isnum(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def moving_avg(values: list[float], window: int = SMOOTH_WINDOW) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def seed_band(series: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    """Align seeds to the shortest length; return (mean, mean-std, mean+std)."""
    n = min(len(s) for s in series)
    mid, lo, hi = [], [], []
    for i in range(n):
        vals = [s[i] for s in series]
        m, sd = mean(vals), std(vals)
        mid.append(m)
        lo.append(m - sd)
        hi.append(m + sd)
    return mid, lo, hi


def convergence_episode(raw: list[float]) -> int:
    """Episode where the smoothed return first reaches 95% of the way from its
    start value to the last-100-episode plateau (robust to negative returns)."""
    if len(raw) < PLATEAU_TAIL + 1:
        return len(raw)
    sm = moving_avg(raw)
    plateau = mean(sm[-PLATEAU_TAIL:])
    start = mean(sm[: SMOOTH_WINDOW]) if len(sm) >= SMOOTH_WINDOW else sm[0]
    thr = start + 0.95 * (plateau - start)
    if plateau >= start:
        for i, v in enumerate(sm):
            if v >= thr:
                return i
    else:
        for i, v in enumerate(sm):
            if v <= thr:
                return i
    return len(raw)


def save_fig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.png/.pdf")


def fmt(m: float, s: float, digits: int = 3) -> str:
    return f"{m:.{digits}f}±{s:.{digits}f}"


def collect_condition(arm_dirs: dict[str, list[Path]], policy_of: dict[str, str]) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for arm, dirs in arm_dirs.items():
        rows = [r for r in (result_row(d, policy_of[arm]) for d in dirs) if r]
        if not rows:
            continue
        out[arm] = {
            k: (mean([r.get(k, 0.0) for r in rows]), std([r.get(k, 0.0) for r in rows]))
            for k in TABLE_METRICS
        }
        out[arm]["runs"] = (float(len(rows)), 0.0)
    return out


# ------------------------------------------------------------------- figures
def fig1_convergence(summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    conv_notes = []
    for arm in LEARN_ARMS:
        dirs = [MATRIX / f"{arm}_{s}" for s in SEEDS]
        rewards = [trace_series(d, "raw_return") for d in dirs if (d / "ppo_training_trace.csv").exists()]
        rewards = [r for r in rewards if r]
        if not rewards:
            continue
        color = ARM_COLOR[arm]
        mid, lo, hi = seed_band([moving_avg(r) for r in rewards])
        x = range(len(mid))
        axes[0].plot(x, mid, color=color, label=ARM_LABEL[arm], lw=1.4)
        axes[0].fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
        conflicts = [trace_series(d, "conflict_cost") for d in dirs if (d / "ppo_training_trace.csv").exists()]
        mid_c, lo_c, hi_c = seed_band([moving_avg(c) for c in conflicts if c])
        axes[1].plot(range(len(mid_c)), mid_c, color=color, lw=1.4)
        axes[1].fill_between(range(len(mid_c)), lo_c, hi_c, color=color, alpha=0.15, lw=0)
        lambdas = [trace_series(d, "lambda_conflict") for d in dirs if (d / "ppo_training_trace.csv").exists()]
        mid_l, lo_l, hi_l = seed_band([l for l in lambdas if l])
        axes[2].plot(range(len(mid_l)), mid_l, color=color, lw=1.4)
        axes[2].fill_between(range(len(mid_l)), lo_l, hi_l, color=color, alpha=0.15, lw=0)
        eps = [convergence_episode(r) for r in rewards]
        conv_notes.append((arm, mean([float(e) for e in eps])))
    if conv_notes:
        prop = next((e for a, e in conv_notes if a == "proposed"), None)
        if prop is not None:
            axes[0].axvline(prop, color=ARM_COLOR["proposed"], ls=":", lw=1.0)
            axes[0].annotate(f"proposed converges ~ep {prop:.0f}",
                             xy=(prop, axes[0].get_ylim()[0]),
                             xytext=(prop + 30, axes[0].get_ylim()[0]),
                             fontsize=8, color=ARM_COLOR["proposed"], va="bottom")
    axes[1].axhline(CONFLICT_LIMIT, color="k", ls="--", lw=1.0)
    axes[1].annotate(f"conflict limit {CONFLICT_LIMIT}", xy=(0.55, CONFLICT_LIMIT),
                     xycoords=("axes fraction", "data"), fontsize=8, va="bottom")
    axes[0].set_xlabel("training episode"); axes[0].set_ylabel("episode return (raw env reward)")
    axes[0].set_title("(a) Reward convergence (3 seeds, ±std)")
    axes[1].set_xlabel("training episode"); axes[1].set_ylabel("conflict cost (episode rate)")
    axes[1].set_title("(b) Constraint cost vs budget")
    axes[2].set_xlabel("training episode"); axes[2].set_ylabel(r"$\lambda_{conflict}$")
    axes[2].set_title("(c) Conflict dual variable trajectory")
    axes[0].legend(fontsize=8, loc="lower right")
    save_fig(fig, "fig1_convergence")
    summary["convergence_ep"] = {a: e for a, e in conv_notes}


def fig2_multiscale(summary: dict) -> None:
    runs = [("mscale_uav2_0", "2 UAVs"), ("proposed_0", "4 UAVs (default)"), ("mscale_uav6_0", "6 UAVs")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    notes = {}
    for name, label in runs:
        raw = trace_series(MATRIX / name, "raw_return")
        if not raw:
            continue
        sm = moving_avg(raw)
        ax.plot(range(len(sm)), sm, lw=1.4, label=label)
        notes[label] = convergence_episode(raw)
    ax.set_xlabel("training episode"); ax.set_ylabel("episode return (raw env reward)")
    ax.set_title("Proposed controller convergence vs UAV-count scale (seed 0)")
    ax.legend(fontsize=9)
    save_fig(fig, "fig2_multiscale")
    summary["multiscale_convergence_ep"] = notes


SWEEP_AXES = {
    "uav": ("fig3_uav", "number of UAVs", [("uav/uav2", 2), ("uav/uav3", 3), ("uav/uav4", 4), ("uav/uav6", 6)]),
    "load": ("fig4_load", "tasks per episode (arrival load)", [("load/load10", 10), ("load/load20", 20), ("load/load30", 30), ("load/load40", 40)]),
    "snr": ("fig5_snr", "sensed SNR band center (dB)", [("snr/snr_m5_0", -2.5), ("snr/snr_0_5", 2.5), ("snr/snr_5_10", 7.5), ("snr/snr_10_15", 12.5), ("snr/snr_15_20", 17.5)]),
}


def sweep_point_stats(point: str, arm: str) -> dict[str, tuple[float, float]] | None:
    if arm in LEARN_ARMS or arm.startswith("retrain_"):
        policy, seeds = "ppo", SEEDS if arm in LEARN_ARMS else (0,)
    else:
        policy, seeds = BASELINE_ARMS[arm]
    rows = []
    for seed in seeds:
        row = result_row(SWEEP / point / f"{arm}_{seed}", policy)
        if row:
            rows.append(row)
    if not rows:
        return None
    return {k: (mean([r.get(k, 0.0) for r in rows]), std([r.get(k, 0.0) for r in rows])) for k in TABLE_METRICS}


def figs_3_4_5(summary: dict) -> None:
    all_arms = list(LEARN_ARMS) + list(BASELINE_ARMS)
    sweep_stats: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for axis, (figname, xlabel, points) in SWEEP_AXES.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for arm in all_arms:
            xs, acc_m, acc_s, con_m, con_s = [], [], [], [], []
            for point, xval in points:
                stats = sweep_point_stats(point, arm)
                if stats is None:
                    continue
                sweep_stats.setdefault(point, {})[arm] = stats
                xs.append(xval)
                acc_m.append(stats["average_accuracy"][0]); acc_s.append(stats["average_accuracy"][1])
                con_m.append(stats["airspace_conflict_rate"][0]); con_s.append(stats["airspace_conflict_rate"][1])
            if not xs:
                continue
            kw = dict(color=ARM_COLOR[arm], marker=ARM_MARKER[arm], ms=5, lw=1.3, capsize=2)
            axes[0].errorbar(xs, acc_m, yerr=acc_s, label=ARM_LABEL[arm], **kw)
            axes[1].errorbar(xs, con_m, yerr=con_s, **kw)
        axes[0].set_xlabel(xlabel); axes[0].set_ylabel("average accuracy (LCB)")
        axes[0].set_title("(a) Semantic accuracy")
        axes[1].set_xlabel(xlabel); axes[1].set_ylabel("airspace conflict rate")
        axes[1].set_title("(b) Airspace conflict rate")
        axes[1].axhline(CONFLICT_LIMIT, color="k", ls="--", lw=0.9)
        axes[0].legend(fontsize=7, ncol=2)
        save_fig(fig, figname)
    summary["sweep_stats"] = sweep_stats


def fig6_violation_pareto(peak: dict, nominal: dict) -> None:
    arms = [a for a in list(LEARN_ARMS) + list(BASELINE_ARMS) if a in peak]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    width = 0.38
    xs = range(len(arms))
    peak_v = [peak[a]["airspace_conflict_rate"][0] for a in arms]
    peak_e = [peak[a]["airspace_conflict_rate"][1] for a in arms]
    nom_v = [nominal.get(a, {}).get("airspace_conflict_rate", (0.0, 0.0))[0] for a in arms]
    nom_e = [nominal.get(a, {}).get("airspace_conflict_rate", (0.0, 0.0))[1] for a in arms]
    axes[0].bar([x - width / 2 for x in xs], peak_v, width, yerr=peak_e, capsize=2,
                label="peak (utm_conflict)", color="#d62728", alpha=0.85)
    axes[0].bar([x + width / 2 for x in xs], nom_v, width, yerr=nom_e, capsize=2,
                label="nominal", color="#1f77b4", alpha=0.85)
    axes[0].axhline(CONFLICT_LIMIT, color="k", ls="--", lw=1.0, label=f"limit {CONFLICT_LIMIT}")
    axes[0].set_xticks(list(xs))
    axes[0].set_xticklabels([ARM_LABEL[a].split(" (")[0] for a in arms], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("airspace conflict rate")
    axes[0].set_title("(a) Constraint violation, dual conditions")
    axes[0].legend(fontsize=8)
    for arm in arms:
        x = peak[arm]["airspace_conflict_rate"][0]
        y = peak[arm]["average_accuracy"][0]
        axes[1].scatter(x, y, color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                        s=110 if arm == "proposed" else 70, zorder=3,
                        label=ARM_LABEL[arm].split(" (")[0])
        axes[1].annotate(ARM_LABEL[arm].split(" (")[0], xy=(x, y), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
    axes[1].axvline(CONFLICT_LIMIT, color="k", ls="--", lw=0.9)
    axes[1].set_xlabel("airspace conflict rate (peak)")
    axes[1].set_ylabel("average accuracy (peak)")
    axes[1].set_title("(b) Utility-violation Pareto view")
    save_fig(fig, "fig6_violation_pareto")


ZEROSHOT_POINTS = (
    ("zeroshot/zs_legacy", "unseen profile (legacy)", "retrain_legacy"),
    ("load/load30", "x1.5 load (tpe=30)", "retrain_load15"),
    ("snr/snr_m5_0", "low SNR (-5..0 dB)", "retrain_lowsnr"),
)


def fig7_zeroshot(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    groups = ["proposed (zero-shot)", "retrained reference", "semantic greedy", "random"]
    colors = ["#d62728", "#2ca02c", "#ff7f0e", "#bcbd22"]
    notes = {}
    for metric_idx, metric in enumerate(("average_accuracy", "airspace_conflict_rate")):
        ax = axes[metric_idx]
        n_groups = len(groups)
        width = 0.8 / n_groups
        for gi, (label, color) in enumerate(zip(groups, colors)):
            xs, vals, errs = [], [], []
            for pi, (point, plabel, retrain_arm) in enumerate(ZEROSHOT_POINTS):
                if label == "proposed (zero-shot)":
                    stats = sweep_point_stats(point, "proposed")
                elif label == "retrained reference":
                    stats = sweep_point_stats(point, retrain_arm)
                elif label == "semantic greedy":
                    stats = sweep_point_stats(point, "bl_semantic_greedy")
                else:
                    stats = sweep_point_stats(point, "bl_random")
                if stats is None:
                    continue
                xs.append(pi + (gi - (n_groups - 1) / 2) * width)
                vals.append(stats[metric][0])
                errs.append(stats[metric][1])
                if metric == "average_accuracy":
                    notes.setdefault(plabel, {})[label] = stats[metric][0]
            ax.bar(xs, vals, width * 0.92, yerr=errs, capsize=2, color=color,
                   label=label if metric_idx == 0 else None)
        ax.set_xticks(range(len(ZEROSHOT_POINTS)))
        ax.set_xticklabels([p[1] for p in ZEROSHOT_POINTS], fontsize=8)
        ax.set_ylabel(metric.replace("_", " "))
        if metric == "airspace_conflict_rate":
            ax.axhline(CONFLICT_LIMIT, color="k", ls="--", lw=0.9)
    axes[0].set_title("(a) Zero-shot vs retrained: accuracy")
    axes[1].set_title("(b) Zero-shot vs retrained: conflict rate")
    axes[0].legend(fontsize=8)
    save_fig(fig, "fig7_zeroshot")
    summary["zeroshot"] = notes


# -------------------------------------------------------------------- tables
def write_table(path_base: Path, header: list[str], rows: list[list[str]], title: str) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    md = [f"# {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    (path_base.with_suffix(".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    with path_base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote tables/{path_base.name}.md/.csv")


def table1(peak: dict, nominal: dict) -> None:
    header = ["condition", "method", "runs"] + list(TABLE_METRICS)
    rows = []
    for cond_name, cond in (("peak", peak), ("nominal", nominal)):
        for arm in list(LEARN_ARMS) + list(BASELINE_ARMS):
            stats = cond.get(arm)
            if not stats:
                continue
            rows.append([cond_name, ARM_LABEL[arm], str(int(stats["runs"][0]))]
                        + [fmt(*stats[m]) for m in TABLE_METRICS])
    write_table(TAB_DIR / "table1_main", header, rows,
                "Table I -- Main comparison (dual conditions, mean±std across seeds/runs)")


V3_ARMS = {
    "A2": ("ppo", "Proposed (full, v3 500ep)"),
    "B2": ("ppo", "w/o conflict dual channel"),
    "B1": ("ppo", "w/o slow mobility head"),
    "A2warm": ("ppo", "dual warm-start probe"),
    "A1": ("ppo", "legacy profile reference"),
    "Cgreedy": ("semantic_greedy", "Semantic greedy"),
    "Ccache": ("always_cache", "Always cache"),
}


def table2_ablation() -> None:
    header = ["arm", "runs"] + list(TABLE_METRICS)
    rows = []
    for arm, (policy, label) in V3_ARMS.items():
        arm_rows = []
        for seed in SEEDS:
            row = result_row(V3 / f"{arm}_{seed}", policy)
            if row:
                arm_rows.append(row)
        if not arm_rows:
            continue
        rows.append([label, str(len(arm_rows))]
                    + [fmt(mean([r.get(m, 0.0) for r in arm_rows]), std([r.get(m, 0.0) for r in arm_rows]))
                       for m in TABLE_METRICS])
    write_table(TAB_DIR / "table2_ablation", header, rows,
                "Table II -- Structural ablation (v3 BUBBLES chain, peak condition, mean±std over 3 seeds)")


def table3_sample_efficiency(summary: dict) -> None:
    wall = {}
    status = MATRIX / "train_status.tsv"
    if status.exists():
        for line in status.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 6 and parts[2] == "0" and parts[3] != "-":
                try:
                    t0 = datetime.fromisoformat(parts[3])
                    t1 = datetime.fromisoformat(parts[4])
                    wall.setdefault(parts[0], []).append((t1 - t0).total_seconds() / 60.0)
                except ValueError:
                    continue
    header = ["method", "episodes to 95% plateau (mean±std)", "final return (mean±std)", "wall-clock min/run (mean)"]
    rows = []
    eff = {}
    for arm in LEARN_ARMS:
        eps, finals = [], []
        for seed in SEEDS:
            raw = trace_series(MATRIX / f"{arm}_{seed}", "raw_return")
            if not raw:
                continue
            eps.append(float(convergence_episode(raw)))
            finals.append(mean(moving_avg(raw)[-PLATEAU_TAIL:]))
        if not eps:
            continue
        wc = mean(wall.get(arm, [0.0]))
        rows.append([ARM_LABEL[arm], fmt(mean(eps), std(eps), 0), fmt(mean(finals), std(finals), 2), f"{wc:.1f}"])
        eff[arm] = {"ep95": mean(eps), "final_return": mean(finals), "wall_min": wc}
    write_table(TAB_DIR / "table3_sample_efficiency", header, rows,
                "Table III -- Sample efficiency (episodes to 95% of plateau; wall-clock per 1000-ep run)")
    summary["sample_efficiency"] = eff


# ------------------------------------------------------------------- summary
def criteria_block(peak: dict, nominal: dict, summary: dict) -> list[str]:
    lines = ["## Criteria check", ""]

    def check(name: str, ok: bool, observed: str) -> None:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name} (observed {observed})")

    conv = summary.get("convergence_ep", {})
    if "proposed" in conv:
        check("proposed plateaus before 1000 ep (ep95 < 900)", conv["proposed"] < 900, f"ep95≈{conv['proposed']:.0f}")
    p, f_, s_ = peak.get("proposed"), peak.get("flat_ppo"), peak.get("service_only")
    if p and f_:
        d = p["average_accuracy"][0] - f_["average_accuracy"][0]
        check("proposed accuracy >= flat PPO accuracy (peak)", d >= 0.0, f"delta {d:+.4f}")
        d = f_["airspace_conflict_rate"][0] - p["airspace_conflict_rate"][0]
        check("flat PPO conflict - proposed conflict >= 0 (structure carries safety)", d >= 0.0, f"delta {d:+.4f}")
    if p and s_:
        d = p["average_accuracy"][0] - s_["average_accuracy"][0]
        check("proposed accuracy >= service-only accuracy (peak)", d >= 0.0, f"delta {d:+.4f}")
    nl = peak.get("no_lagrangian")
    if p and nl:
        d = nl["airspace_conflict_rate"][0] - p["airspace_conflict_rate"][0]
        check("no-Lagrangian conflict - proposed conflict >= 0.05 (dual load-bearing; v3 found the dual inert -- expect FAIL)",
              d >= 0.05, f"delta {d:+.4f}")
    fx = peak.get("fixed_penalty")
    if p and fx:
        d = p["average_accuracy"][0] - fx["average_accuracy"][0]
        check("proposed accuracy >= fixed-penalty accuracy (adaptive dual not worse)", d >= -0.01, f"delta {d:+.4f}")
    orc = peak.get("bl_oracle_best_feasible_evidence")
    if p and orc:
        d = orc["average_accuracy"][0] - p["average_accuracy"][0]
        check("oracle accuracy >= proposed accuracy (upper bound sanity)", d >= -0.005, f"delta {d:+.4f}")
    rnd = peak.get("bl_random")
    if p and rnd:
        d = p["average_accuracy"][0] - rnd["average_accuracy"][0]
        check("proposed accuracy > random accuracy", d > 0.0, f"delta {d:+.4f}")
    pn = nominal.get("proposed")
    if pn:
        check("nominal proposed semantic success >= 0.92 (v3 criterion 6)",
              pn["semantic_success_rate"][0] >= 0.92, f"{pn['semantic_success_rate'][0]:.4f}")
        check("nominal proposed task success >= 0.30 (v3 criterion 6)",
              pn["task_success_rate"][0] >= 0.30, f"{pn['task_success_rate'][0]:.4f}")
    for plabel, vals in summary.get("zeroshot", {}).items():
        zs, rt = vals.get("proposed (zero-shot)"), vals.get("retrained reference")
        if zs is not None and rt is not None:
            gap = zs - rt
            check(f"zero-shot accuracy gap vs retrained @ {plabel} >= -0.05", gap >= -0.05, f"gap {gap:+.4f}")
    return lines


def main() -> int:
    summary: dict = {}
    peak_dirs = {arm: [MATRIX / f"{arm}_{s}" for s in SEEDS] for arm in LEARN_ARMS}
    nom_dirs = {arm: [MATRIX / f"{arm}_{s}_nom" for s in SEEDS] for arm in LEARN_ARMS}
    policy_of = {arm: "ppo" for arm in LEARN_ARMS}
    for bl, (policy, seeds) in BASELINE_ARMS.items():
        peak_dirs[bl] = [MATRIX / f"{bl}_{s}" for s in seeds]
        nom_dirs[bl] = [MATRIX / f"{bl}_{s}_nom" for s in seeds]
        policy_of[bl] = policy
    peak = collect_condition(peak_dirs, policy_of)
    nominal = collect_condition(nom_dirs, policy_of)

    fig1_convergence(summary)
    fig2_multiscale(summary)
    figs_3_4_5(summary)
    if peak:
        fig6_violation_pareto(peak, nominal)
    fig7_zeroshot(summary)
    table1(peak, nominal)
    table2_ablation()
    table3_sample_efficiency(summary)

    lines = [
        "# matrix_v1 experiment summary",
        "",
        f"> generated {datetime.now().isoformat(timespec='seconds')} by scripts/report_matrix_v1.py",
        "> spec: docs_spec/RL_Experiment_Standards_Survey.md (section 3 final matrix)",
        "",
        "## Artifact index",
        "",
        "| artifact | content |",
        "|---|---|",
        "| figures/fig1_convergence | reward / constraint cost + 0.08 budget / lambda_conflict, 5 learning arms, 3 seeds ±std |",
        "| figures/fig2_multiscale | proposed convergence at 2/4/6 UAVs |",
        "| figures/fig3_uav, fig4_load, fig5_snr | zero-shot sweeps, all 9 methods |",
        "| figures/fig6_violation_pareto | dual-condition violation bars + utility-violation Pareto |",
        "| figures/fig7_zeroshot | zero-shot vs retrained reference at 3 unseen points |",
        "| tables/table1_main | dual-condition main comparison, mean±std |",
        "| tables/table2_ablation | v3 structural ablation with ±std |",
        "| tables/table3_sample_efficiency | ep-to-95%-plateau + wall clock |",
        "",
        "## Headline numbers (peak condition)",
        "",
    ]
    for arm in list(LEARN_ARMS) + list(BASELINE_ARMS):
        stats = peak.get(arm)
        if not stats:
            continue
        lines.append(
            f"- {ARM_LABEL[arm]}: acc {fmt(*stats['average_accuracy'])}, "
            f"conflict {fmt(*stats['airspace_conflict_rate'])}, "
            f"deadline-vio {fmt(*stats['deadline_violation_rate'])}, "
            f"reward {fmt(*stats['average_reward'])}"
        )
    lines.append("")
    conv = summary.get("convergence_ep", {})
    if conv:
        lines.append("## Convergence readings")
        lines.append("")
        for arm, ep in conv.items():
            lines.append(f"- {ARM_LABEL.get(arm, arm)}: ep95 ≈ {ep:.0f}")
        for label, ep in summary.get("multiscale_convergence_ep", {}).items():
            lines.append(f"- multiscale {label}: ep95 ≈ {ep}")
        lines.append("")
    lines += criteria_block(peak, nominal, summary)
    out = MATRIX / "matrix_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (MATRIX / "matrix_summary.json").write_text(
        json.dumps({"peak": {a: {m: list(v) for m, v in s.items()} for a, s in peak.items()},
                    "nominal": {a: {m: list(v) for m, v in s.items()} for a, s in nominal.items()},
                    "summary": {k: v for k, v in summary.items() if k != "sweep_stats"}},
                   indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
