#!/usr/bin/env python3
"""paper1_stats.py --- Statistical package for Paper 1 (P0-3).

Log-level statistical analysis on the 160-server prediction CSVs:
  (1) image-clustered bootstrap CIs for the five per-type token-image gaps
      Delta (Table III), the M4/M3/M1/M5 main comparison (Table IV), and the
      cross-VLM token/image accuracies (Table V);
  (2) paired McNemar tests (rule vs calibrated LCB; M4 vs M1; M4 vs M3;
      truncated-token vs full-token where relevant);
  (3) the true evaluated n and its decomposition (unique questions x SNR);
  (4) the rule-vs-calibrated aggregate comparison (statistically
      indistinguishable check).

All resampling clusters on image_id (a question and its 6 SNR replays share an
image), which is the correct unit given the crc32(image_id) test split.

Usage:
  python scripts/paper1_stats.py --pred-dir outputs/vlm \
      --out outputs/reports/paper1_stats.json [--boot 5000] [--seed 0]

Outputs a JSON report + a human-readable stdout summary. Read-only on inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

CHANNELS = ("awgn", "rayleigh", "rician")
TAGS = ("", "_cmp", "_extra")  # main(presence,counting) + comparison + (co_presence,threshold)
QTYPES = ("presence", "counting", "comparison", "co_presence", "threshold")


def is_test(image_id, frac: float = 0.2) -> bool:
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def cc(v) -> int:
    return 1 if str(v).strip().lower() in ("true", "1", "yes") else 0


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - m), min(1.0, c + m)


# --------------------------------------------------------------------------
# Data loading: build a tidy per-task table with token/image correctness.
# One row per (image_id, question, snr_bin, question_type, channel): s1_ok, s2_ok.
# --------------------------------------------------------------------------
def load_dataset(pred_dir: str, prefix: str, channels=CHANNELS, tags=TAGS,
                 test_only: bool = True) -> pd.DataFrame:
    frames = []
    for ch in channels:
        for tg in tags:
            f = os.path.join(pred_dir, f"{prefix}_{ch}{tg}_predictions.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(
                f,
                usecols=["image_id", "question", "snr_bin", "question_type",
                         "service_level", "correct"],
                dtype={"image_id": str},
            )
            df["channel"] = ch
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)
    if not test_only:  # _pivot_ok always applies the test filter; guard callers
        big["__all__"] = True
    return _pivot_ok(big)


# --------------------------------------------------------------------------
# Clustered bootstrap: resample image_ids with replacement, recompute stat.
# --------------------------------------------------------------------------
def clustered_bootstrap_ci(df: pd.DataFrame, stat_fn, n_boot: int, seed: int,
                           alpha: float = 0.05):
    """df must have an 'image_id' column. stat_fn(sub_df)->float (nan-safe).
    Returns (point, lo, hi, mean_boot). Clusters on image_id."""
    rng = np.random.default_rng(seed)
    point = stat_fn(df)
    ids = df["image_id"].unique()
    # pre-group rows by image_id for fast resampling
    groups = {iid: g for iid, g in df.groupby("image_id")}
    boot = np.empty(n_boot, dtype=float)
    n_ids = len(ids)
    for b in range(n_boot):
        pick = rng.choice(ids, size=n_ids, replace=True)
        sub = pd.concat([groups[i] for i in pick], ignore_index=True)
        boot[b] = stat_fn(sub)
    boot = boot[~np.isnan(boot)]
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(point), lo, hi, float(np.mean(boot))


# stat functions -----------------------------------------------------------
def stat_delta(qt):
    def f(g):
        gg = g[g.question_type == qt]
        if len(gg) == 0:
            return np.nan
        return gg["s1_ok"].mean() - gg["s2_ok"].mean()
    return f


def stat_acc(col, qt=None):
    def f(g):
        gg = g if qt is None else g[g.question_type == qt]
        if len(gg) == 0:
            return np.nan
        return gg[col].mean()
    return f


# rule accuracy: presence->s2, others->s1.  calibrated LCB is emulated below.
def rule_pick_col(row):
    return "s2_ok" if row["question_type"] == "presence" else "s1_ok"


def stat_rule_acc(g):
    if len(g) == 0:
        return np.nan
    sel = np.where(g["question_type"].values == "presence",
                   g["s2_ok"].values, g["s1_ok"].values)
    return float(np.nanmean(sel))


# --------------------------------------------------------------------------
# Calibrated LCB policy (M4): learn per (qtype, snr) best-LCB service on TRAIN,
# apply on TEST. Mirrors build_comparison_v2.learn_policy exactly.
# --------------------------------------------------------------------------
def learn_lcb_policy(pred_dir, prefix, channels=CHANNELS, tags=TAGS):
    """channel -> {(qtype, snr_bin): '1'|'2'} learned on TRAIN split."""
    pol = {}
    for ch in channels:
        cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (qt,snr)->svc->[k,n]
        for tg in tags:
            f = os.path.join(pred_dir, f"{prefix}_{ch}{tg}_predictions.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f, usecols=["image_id", "question", "snr_bin",
                                         "question_type", "service_level", "correct"],
                         dtype={"image_id": str})
            df = df[~df.image_id.map(is_test)]  # TRAIN only
            df["ok"] = df.correct.map(cc)
            for _, r in df.iterrows():
                s = str(r.service_level)
                if s in ("1", "2"):
                    key = (r.question_type, r.snr_bin)
                    cell[key][s][1] += 1
                    cell[key][s][0] += int(r.ok)
        p = {}
        for key, sv in cell.items():
            best, blcb = None, -1.0
            for s, (k, n) in sv.items():
                _, lcb, _ = wilson(k, n)
                if lcb > blcb:
                    blcb, best = lcb, s
            p[key] = best
        pol[ch] = p
    return pol


def apply_lcb_col(df: pd.DataFrame, pol: dict) -> np.ndarray:
    """Return per-row correctness of the LCB policy selection."""
    out = np.full(len(df), np.nan)
    qt = df["question_type"].values
    snr = df["snr_bin"].values
    ch = df["channel"].values
    s1 = df["s1_ok"].values
    s2 = df["s2_ok"].values
    for i in range(len(df)):
        sel = pol.get(ch[i], {}).get((qt[i], snr[i]))
        if sel == "1":
            out[i] = s1[i]
        elif sel == "2":
            out[i] = s2[i]
        else:  # fall back to majority-ish: if unseen cell, use rule
            out[i] = s2[i] if qt[i] == "presence" else s1[i]
    return out


# --------------------------------------------------------------------------
# McNemar exact paired test on two binary-correctness vectors (aligned rows).
# --------------------------------------------------------------------------
def mcnemar(a: np.ndarray, b: np.ndarray):
    """a,b: 0/1 correctness, paired & aligned. Returns dict with b,c,p (exact
    binomial two-sided for n=b+c<=25 else continuity-corrected chi-square)."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask].astype(int)
    b_ = b[mask].astype(int)
    n01 = int(np.sum((a == 0) & (b_ == 1)))  # a wrong, b right
    n10 = int(np.sum((a == 1) & (b_ == 0)))  # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {"n_discordant": 0, "a_right_b_wrong": n10, "a_wrong_b_right": n01,
                "chi2_cc": 0.0, "p_exact": 1.0,
                "acc_a": float(a.mean()) if len(a) else float("nan"),
                "acc_b": float(b_.mean()) if len(b_) else float("nan")}
    # exact two-sided binomial (scipy for numerical stability at large n)
    try:
        from scipy.stats import binomtest
        p_exact = float(binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        # normal approximation fallback (large n)
        z = (abs(n01 - n10) - 1) / math.sqrt(n)
        p_exact = math.erfc(z / math.sqrt(2))
    p_exact = min(1.0, p_exact)
    chi2 = (abs(n01 - n10) - 1) ** 2 / n if n > 0 else 0.0
    return {"n_discordant": n, "a_right_b_wrong": n10, "a_wrong_b_right": n01,
            "chi2_cc": round(chi2, 3), "p_exact": p_exact,
            "acc_a": float(a.mean()), "acc_b": float(b_.mean())}


# --------------------------------------------------------------------------
def true_n_decomposition(pred_dir, prefix, channel="rician", tags=TAGS):
    """Report unique questions per qtype (single SNR) and total test tasks."""
    per_snr = {}
    tasks = {}
    for tg in tags:
        f = os.path.join(pred_dir, f"{prefix}_{channel}{tg}_predictions.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, usecols=["image_id", "question", "snr_bin",
                                     "question_type", "service_level"],
                             dtype={"image_id": str})
        df = df[df.service_level == 2]
        df = df[df.image_id.map(is_test)]
        for qt, g in df.groupby("question_type"):
            ntask = g[["image_id", "question", "snr_bin"]].drop_duplicates().shape[0]
            nq = g[["image_id", "question"]].drop_duplicates().shape[0]
            tasks[qt] = tasks.get(qt, 0) + ntask
            per_snr[qt] = per_snr.get(qt, 0) + nq
    n_snr = 6
    return {"unique_questions_per_type": per_snr,
            "unique_questions_total": sum(per_snr.values()),
            "test_tasks_per_type_one_channel": tasks,
            "test_tasks_total_one_channel": sum(tasks.values()),
            "snr_grid_size": n_snr}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/vlm")
    ap.add_argument("--prefix", default="v2_0", help="VisDrone primary prefix")
    ap.add_argument("--dv-prefix", default="dv_rician", help="DroneVehicle prefix (rician only)")
    ap.add_argument("--v25-prefix", default="v25_rician", help="Qwen2.5-VL-3B cross-VLM")
    ap.add_argument("--v26-prefix", default="v26_rician", help="SmolVLM cross-VLM")
    ap.add_argument("--out", default="outputs/reports/paper1_stats.json")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = {"config": vars(args)}
    print("=" * 78)
    print("PAPER 1 STATISTICAL PACKAGE (P0-3)  boot=%d seed=%d" % (args.boot, args.seed))
    print("=" * 78)

    # ---- (3) true n decomposition ----
    nd = true_n_decomposition(args.pred_dir, args.prefix, "rician")
    report["true_n"] = nd
    print("\n[3] TRUE n (VisDrone, single channel = Rician test set):")
    print("    unique questions/type:", nd["unique_questions_per_type"],
          "-> total", nd["unique_questions_total"])
    print("    x 6 SNR = test tasks/type:", nd["test_tasks_per_type_one_channel"],
          "-> total", nd["test_tasks_total_one_channel"])

    # ---- load VisDrone pooled-3-channel test table ----
    print("\nLoading VisDrone (3 channels pooled, test)...")
    vd = load_dataset(args.pred_dir, args.prefix, CHANNELS, TAGS, test_only=True)
    print("    rows:", len(vd), " qtypes:", sorted(vd.question_type.unique()),
          " images:", vd.image_id.nunique())

    # ---- (1a) Table III: Delta per type, clustered bootstrap CI ----
    print("\n[1a] Table III --- token-image gap Delta by type (VisDrone, 3ch pooled)")
    report["table3_delta_visdrone"] = {}
    for qt in QTYPES:
        sub = vd[vd.question_type == qt]
        pt, lo, hi, mb = clustered_bootstrap_ci(sub, stat_delta(qt), args.boot, args.seed)
        tok = sub["s1_ok"].mean(); img = sub["s2_ok"].mean()
        crosses = (lo < 0 < hi)
        report["table3_delta_visdrone"][qt] = {
            "delta": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "token_acc": round(float(tok), 4), "image_acc": round(float(img), 4),
            "n_tasks": int(len(sub)), "n_images": int(sub.image_id.nunique()),
            "ci_crosses_zero": bool(crosses)}
        print(f"    {qt:12s} d={pt:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  "
              f"tok={tok:.3f} img={img:.3f}  n={len(sub)}  "
              f"{'CROSSES 0' if crosses else 'sign-consistent'}")

    # DroneVehicle (rician only). Authoritative source = the MERGED
    # dv_rician_predictions.csv (490 images), which the paper's
    # build_evidence_complementarity_dv.py uses; the _main/_cmp/_extra parts are
    # a smaller earlier subset and must NOT be used for the DV column.
    print("\n[1a] Table III --- Delta by type (DroneVehicle, Rician only; MERGED file)")
    dv = load_merged_single(args.pred_dir, "dv_rician_predictions.csv")
    report["table3_delta_dronevehicle"] = {}
    if len(dv):
        print("    rows:", len(dv), " images:", dv.image_id.nunique())
        for qt in QTYPES:
            sub = dv[dv.question_type == qt]
            if len(sub) == 0:
                continue
            pt, lo, hi, mb = clustered_bootstrap_ci(sub, stat_delta(qt), args.boot, args.seed)
            crosses = (lo < 0 < hi)
            report["table3_delta_dronevehicle"][qt] = {
                "delta": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "n_tasks": int(len(sub)), "ci_crosses_zero": bool(crosses)}
            print(f"    {qt:12s} d={pt:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  n={len(sub)}  "
                  f"{'CROSSES 0' if crosses else 'sign-consistent'}")
    else:
        print("    (DroneVehicle CSVs not found)")

    # ---- (4) rule vs calibrated LCB aggregate ----
    print("\n[4] Rule vs calibrated LCB (aggregate test accuracy)")
    print("    Learning LCB policy on TRAIN (VisDrone)...")
    pol_vd = learn_lcb_policy(args.pred_dir, args.prefix)
    vd = vd.copy()
    vd["rule_ok"] = np.where(vd["question_type"].values == "presence",
                             vd["s2_ok"].values, vd["s1_ok"].values)
    vd["lcb_ok"] = apply_lcb_col(vd, pol_vd)
    rule_acc = float(np.nanmean(vd["rule_ok"]))
    lcb_acc = float(np.nanmean(vd["lcb_ok"]))
    report["rule_vs_lcb_visdrone"] = {
        "rule_acc": round(rule_acc, 4), "lcb_acc": round(lcb_acc, 4),
        "n": int(len(vd)),
        "policy_per_channel": {ch: {f"{k[0]}|{k[1]}": v for k, v in p.items()}
                               for ch, p in pol_vd.items()}}
    print(f"    rule={rule_acc:.4f}  lcb={lcb_acc:.4f}  n={len(vd)}")
    mc = mcnemar(vd["rule_ok"].values, vd["lcb_ok"].values)
    report["mcnemar_rule_vs_lcb_visdrone"] = mc
    print(f"    McNemar rule-vs-lcb: discordant={mc['n_discordant']} "
          f"p_exact={mc['p_exact']:.4g}  (acc_rule={mc['acc_a']:.4f} acc_lcb={mc['acc_b']:.4f})")

    # DroneVehicle rule vs lcb (merged file; LCB policy learned on DV TRAIN split)
    if len(dv):
        pol_dv = learn_lcb_policy_merged(args.pred_dir, "dv_rician_predictions.csv")
        dv = dv.copy()
        dv["rule_ok"] = np.where(dv["question_type"].values == "presence",
                                 dv["s2_ok"].values, dv["s1_ok"].values)
        dv["lcb_ok"] = apply_lcb_col(dv, {"rician": pol_dv})
        r2 = float(np.nanmean(dv["rule_ok"])); l2 = float(np.nanmean(dv["lcb_ok"]))
        # restrict McNemar to rows where BOTH s1 and s2 exist (paired routing)
        both = ~(np.isnan(dv["s1_ok"].values) | np.isnan(dv["s2_ok"].values))
        mc2 = mcnemar(dv["rule_ok"].values[both], dv["lcb_ok"].values[both])
        report["rule_vs_lcb_dronevehicle"] = {"rule_acc": round(r2, 4),
                                              "lcb_acc": round(l2, 4), "n": int(len(dv))}
        report["mcnemar_rule_vs_lcb_dronevehicle"] = mc2
        print(f"    [DroneVehicle] rule={r2:.4f} lcb={l2:.4f} n={len(dv)} "
              f"McNemar p_exact={mc2['p_exact']:.4g} discordant={mc2['n_discordant']}")

    # ---- (2) McNemar M4 vs M1, M4 vs M3 (VisDrone pooled) ----
    print("\n[2] McNemar: M4(rule) vs M1(image) and M4(rule) vs M3(token)  [VisDrone]")
    m4 = vd["rule_ok"].values
    m1 = vd["s2_ok"].values  # image everywhere
    m3 = vd["s1_ok"].values  # token everywhere
    mc_m4_m1 = mcnemar(m4, m1)
    mc_m4_m3 = mcnemar(m4, m3)
    report["mcnemar_m4_vs_m1_visdrone"] = mc_m4_m1
    report["mcnemar_m4_vs_m3_visdrone"] = mc_m4_m3
    print(f"    M4 vs M1(image): acc {mc_m4_m1['acc_a']:.4f} vs {mc_m4_m1['acc_b']:.4f} "
          f"discordant={mc_m4_m1['n_discordant']} p_exact={mc_m4_m1['p_exact']:.4g}")
    print(f"    M4 vs M3(token): acc {mc_m4_m3['acc_a']:.4f} vs {mc_m4_m3['acc_b']:.4f} "
          f"discordant={mc_m4_m3['n_discordant']} p_exact={mc_m4_m3['p_exact']:.4g}")

    # ---- (1b) Table IV: M4/M3/M1/M5 aggregate accuracy with clustered CI ----
    print("\n[1b] Table IV --- aggregate method accuracy + clustered CI (VisDrone pooled)")
    vd["oracle_ok"] = np.nanmax(np.vstack([vd["s1_ok"].values, vd["s2_ok"].values]), axis=0)
    report["table4_methods_visdrone"] = {}
    for name, col in [("M3_token", "s1_ok"), ("M1_image", "s2_ok"),
                      ("M4_rule", "rule_ok"), ("M4_lcb", "lcb_ok"),
                      ("M5_oracle", "oracle_ok")]:
        pt, lo, hi, mb = clustered_bootstrap_ci(vd, stat_acc(col), args.boot, args.seed)
        report["table4_methods_visdrone"][name] = {
            "acc": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": int(len(vd))}
        print(f"    {name:11s} acc={pt:.4f}  CI95=[{lo:.4f},{hi:.4f}]")

    # ---- (1c) Table V: cross-VLM token/image per type ----
    # Authoritative sources (match scripts/analyze_crossvlm3.py):
    #   Qwen2-VL-2B   -> v3_0_rician_predictions.csv (single merged file)
    #   Qwen2.5-VL-3B -> v25_rician_{main,cmp,extra}
    #   SmolVLM-Instr -> v26_rician_{main,cmp,extra}
    print("\n[1c] Table V --- cross-VLM per-type token/image (Rician)")
    report["table5_crossvlm"] = {}
    vlm_files = {
        "qwen2vl_2b": ["v3_0_rician_predictions.csv"],
        "qwen25vl_3b": ["v25_rician_main_predictions.csv", "v25_rician_cmp_predictions.csv",
                        "v25_rician_extra_predictions.csv"],
        "smolvlm": ["v26_rician_main_predictions.csv", "v26_rician_cmp_predictions.csv",
                    "v26_rician_extra_predictions.csv"],
    }
    for vlm, files in vlm_files.items():
        tbl = load_files(args.pred_dir, files)
        if not len(tbl):
            print(f"    {vlm}: CSVs not found ({files[0]})")
            continue
        report["table5_crossvlm"][vlm] = {}
        line = []
        for qt in QTYPES:
            sub = tbl[tbl.question_type == qt]
            if len(sub) == 0:
                continue
            tok = float(sub["s1_ok"].mean()); img = float(sub["s2_ok"].mean())
            report["table5_crossvlm"][vlm][qt] = {
                "token": round(tok, 4), "image": round(img, 4), "n": int(len(sub))}
            line.append(f"{qt[:4]}:{tok:.3f}/{img:.3f}")
        print(f"    {vlm:12s} " + "  ".join(line))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote", args.out)


def _pivot_ok(big: pd.DataFrame) -> pd.DataFrame:
    """Dedup to one row per (image_id, question, snr_bin) task, keeping the FIRST
    CSV occurrence of each service level (matches the paper build scripts, which
    use dict.setdefault; some merged CSVs replay a task several times with
    channel-seed noise). Preserves CSV row order for a deterministic 'first'."""
    big = big[big.image_id.map(is_test)].copy()
    big["ok"] = big.correct.map(cc)
    big["service_level"] = big["service_level"].astype(str)
    big = big[big.service_level.isin(["0", "1", "2"])]
    # keep first occurrence of each (task, service)
    big = big.drop_duplicates(
        subset=["image_id", "question", "snr_bin", "service_level"], keep="first")
    piv = big.pivot_table(
        index=["image_id", "question", "snr_bin", "question_type", "channel"],
        columns="service_level", values="ok", aggfunc="first", sort=False).reset_index()
    piv.columns = [str(c) for c in piv.columns]
    for lv in ("0", "1", "2"):
        if lv not in piv.columns:
            piv[lv] = np.nan
    return piv.rename(columns={"0": "s0_ok", "1": "s1_ok", "2": "s2_ok"})


def load_files(pred_dir, files, channel="rician") -> pd.DataFrame:
    """Load an explicit list of prediction CSVs (single channel), TEST split."""
    frames = []
    for fn in files:
        f = os.path.join(pred_dir, fn)
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, usecols=["image_id", "question", "snr_bin",
                                     "question_type", "service_level", "correct"],
                         dtype={"image_id": str})
        df["channel"] = channel
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _pivot_ok(pd.concat(frames, ignore_index=True))


def load_merged_single(pred_dir, fname, channel="rician") -> pd.DataFrame:
    """Load one merged prediction CSV (e.g. dv_rician_predictions.csv), TEST split."""
    return load_files(pred_dir, [fname], channel=channel)


def learn_lcb_policy_merged(pred_dir, fname):
    """Learn per (qtype, snr)->best-LCB service on the TRAIN split of a single
    merged CSV. Returns {(qtype, snr_bin): '1'|'2'}."""
    f = os.path.join(pred_dir, fname)
    df = pd.read_csv(f, usecols=["image_id", "question", "snr_bin",
                                 "question_type", "service_level", "correct"],
                         dtype={"image_id": str})
    df = df[~df.image_id.map(is_test)]
    df["ok"] = df.correct.map(cc)
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for _, r in df.iterrows():
        s = str(r.service_level)
        if s in ("1", "2"):
            key = (r.question_type, r.snr_bin)
            cell[key][s][1] += 1
            cell[key][s][0] += int(r.ok)
    pol = {}
    for key, sv in cell.items():
        best, blcb = None, -1.0
        for s, (k, n) in sv.items():
            _, lcb, _ = wilson(k, n)
            if lcb > blcb:
                blcb, best = lcb, s
        pol[key] = best
    return pol


# helper to load a dataset given explicit prefix/tags/channels (DroneVehicle & cross-VLM)
def load_dv(pred_dir, pfx=None, prefix=None, tags=("_main", "_cmp", "_extra"),
            channels=None):
    prefix = prefix or "dv_rician"
    frames = []
    if channels is None:
        # DroneVehicle & cross-VLM: files like <prefix>_main_predictions.csv
        for tg in tags:
            f = os.path.join(pred_dir, f"{prefix}{tg}_predictions.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f, usecols=["image_id", "question", "snr_bin",
                                         "question_type", "service_level", "correct"],
                         dtype={"image_id": str})
            df["channel"] = "rician"
            frames.append(df)
    else:
        for ch in channels:
            for tg in ("", "_cmp", "_extra"):
                f = os.path.join(pred_dir, f"{prefix}_{ch}{tg}_predictions.csv")
                if not os.path.exists(f):
                    continue
                df = pd.read_csv(f, usecols=["image_id", "question", "snr_bin",
                                             "question_type", "service_level", "correct"],
                         dtype={"image_id": str})
                df["channel"] = ch
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)
    big = big[big.image_id.map(is_test)].copy()
    big["ok"] = big.correct.map(cc)
    piv = big.pivot_table(
        index=["image_id", "question", "snr_bin", "question_type", "channel"],
        columns="service_level", values="ok", aggfunc="first").reset_index()
    piv.columns = [str(c) for c in piv.columns]
    for lv in ("0", "1", "2"):
        if lv not in piv.columns:
            piv[lv] = np.nan
    return piv.rename(columns={"0": "s0_ok", "1": "s1_ok", "2": "s2_ok"})


if __name__ == "__main__":
    main()
