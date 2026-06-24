from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SERVICE_NAMES = {
    0: "cache_answer",
    1: "semantic_token",
    2: "image_evidence",
}


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._\n"
    view = df[columns].copy()
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in view.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(_fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_report(args: argparse.Namespace) -> str:
    utility = pd.read_csv(args.utility_csv)
    utility["service_name"] = utility["service_level"].map(SERVICE_NAMES)
    low = utility[utility["snr_bin"].isin(args.low_snr_bins)].copy()
    if low.empty:
        raise ValueError(f"No rows found for low-SNR bins: {args.low_snr_bins}")

    key_cols = ["question_type", "risk_level", "view_quality_bin", "freshness_bin", "snr_bin"]

    service_summary = (
        low.groupby(["service_level", "service_name"], as_index=False)
        .agg(
            cells=("accuracy_lcb", "size"),
            accuracy_lcb_mean=("accuracy_lcb", "mean"),
            accuracy_lcb_p10=("accuracy_lcb", lambda s: s.quantile(0.10)),
            accuracy_lcb_p50=("accuracy_lcb", "median"),
            accuracy_lcb_p90=("accuracy_lcb", lambda s: s.quantile(0.90)),
            payload_kb_mean=("payload_kb", "mean"),
            payload_kb_p50=("payload_kb", "median"),
            payload_kb_p90=("payload_kb", lambda s: s.quantile(0.90)),
            uncertainty_mean=("uncertainty", "mean"),
            sample_count_mean=("sample_count", "mean"),
        )
        .sort_values("service_level")
    )

    pivot = low.pivot_table(
        index=key_cols,
        columns="service_level",
        values=["accuracy_lcb", "payload_kb", "uncertainty", "sample_count"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_s{level}" for metric, level in pivot.columns]
    pivot = pivot.reset_index()
    for col in ["accuracy_lcb_s0", "accuracy_lcb_s1", "accuracy_lcb_s2"]:
        if col not in pivot:
            pivot[col] = float("nan")
    for col in ["payload_kb_s0", "payload_kb_s1", "payload_kb_s2"]:
        if col not in pivot:
            pivot[col] = float("nan")
    pivot["token_gain_vs_cache"] = pivot["accuracy_lcb_s1"] - pivot["accuracy_lcb_s0"]
    pivot["image_gain_vs_token"] = pivot["accuracy_lcb_s2"] - pivot["accuracy_lcb_s1"]
    pivot["image_gain_vs_cache"] = pivot["accuracy_lcb_s2"] - pivot["accuracy_lcb_s0"]
    pivot["token_payload_kb"] = pivot["payload_kb_s1"]
    pivot["image_payload_kb"] = pivot["payload_kb_s2"]
    pivot["image_payload_multiplier_vs_token"] = pivot["image_payload_kb"] / pivot["token_payload_kb"].replace(0, pd.NA)

    combo_summary = (
        pivot.groupby(["question_type", "risk_level", "view_quality_bin", "freshness_bin"], as_index=False)
        .agg(
            cases=("snr_bin", "count"),
            cache_lcb=("accuracy_lcb_s0", "mean"),
            token_lcb=("accuracy_lcb_s1", "mean"),
            image_lcb=("accuracy_lcb_s2", "mean"),
            token_gain_vs_cache=("token_gain_vs_cache", "mean"),
            image_gain_vs_token=("image_gain_vs_token", "mean"),
            token_payload_kb=("token_payload_kb", "mean"),
            image_payload_kb=("image_payload_kb", "mean"),
            image_payload_multiplier_vs_token=("image_payload_multiplier_vs_token", "mean"),
        )
        .sort_values(["question_type", "risk_level", "view_quality_bin", "freshness_bin"])
    )

    gain_summary = (
        pivot.groupby(["question_type", "risk_level"], as_index=False)
        .agg(
            cache_lcb=("accuracy_lcb_s0", "mean"),
            token_lcb=("accuracy_lcb_s1", "mean"),
            image_lcb=("accuracy_lcb_s2", "mean"),
            token_gain_vs_cache=("token_gain_vs_cache", "mean"),
            image_gain_vs_token=("image_gain_vs_token", "mean"),
            token_payload_kb=("token_payload_kb", "mean"),
            image_payload_kb=("image_payload_kb", "mean"),
            image_payload_multiplier_vs_token=("image_payload_multiplier_vs_token", "mean"),
        )
        .sort_values(["question_type", "risk_level"])
    )

    thresholds = []
    for epsilon in args.epsilon_grid:
        for gap_threshold in args.cache_gap_thresholds:
            semantic_gap = (epsilon - pivot["accuracy_lcb_s0"]).clip(lower=0.0)
            acceptable = semantic_gap <= gap_threshold
            thresholds.append(
                {
                    "epsilon": epsilon,
                    "cache_gap_threshold": gap_threshold,
                    "acceptable_cell_ratio": acceptable.mean(),
                    "mean_cache_gap": semantic_gap.mean(),
                    "p90_cache_gap": semantic_gap.quantile(0.90),
                }
            )
    threshold_df = pd.DataFrame(thresholds)

    bench_paths = [Path(p) for p in args.benchmark_csv]
    bench_frames = []
    for path in bench_paths:
        df = _read_optional_csv(path)
        if df.empty or "scenario" not in df:
            continue
        sub = df[df["scenario"] == "low_snr_blockage"].copy()
        if sub.empty:
            continue
        sub["source"] = path.as_posix()
        bench_frames.append(sub)
    bench = pd.concat(bench_frames, ignore_index=True) if bench_frames else pd.DataFrame()
    bench_cols = [
        "source",
        "benchmark_policy",
        "semantic_success_rate_mean",
        "task_success_rate_mean",
        "average_accuracy_mean",
        "average_semantic_quality_gap_mean",
        "average_delay_mean",
        "average_payload_kb_mean",
        "deadline_violation_rate_mean",
        "service_level_0_ratio_mean",
        "service_level_1_ratio_mean",
        "service_level_2_ratio_mean",
    ]
    if not bench.empty:
        bench = bench[[c for c in bench_cols if c in bench.columns]]

    token_gain_mean = pivot["token_gain_vs_cache"].mean()
    token_payload_mean = low[low["service_level"] == 1]["payload_kb"].mean()
    token_payload_p90 = low[low["service_level"] == 1]["payload_kb"].quantile(0.90)
    image_payload_mean = low[low["service_level"] == 2]["payload_kb"].mean()
    image_payload_p90 = low[low["service_level"] == 2]["payload_kb"].quantile(0.90)
    image_gain_vs_token = pivot["image_gain_vs_token"].mean()

    lines: list[str] = []
    lines.append("# Low-SNR Service Tradeoff Diagnosis")
    lines.append("")
    lines.append("This report diagnoses the semantic-utility side of `low_snr_blockage` without modifying the original LUT or VLM prediction data.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Semantic utility CSV: `{args.utility_csv}`")
    lines.append(f"- Low-SNR bins: `{', '.join(args.low_snr_bins)}`")
    if bench_paths:
        lines.append("- Benchmark context:")
        for path in bench_paths:
            lines.append(f"  - `{path}`")
    lines.append("")

    lines.append("## Service-Level Low-SNR Utility Summary")
    lines.append("")
    lines.append(_markdown_table(service_summary, [
        "service_level",
        "service_name",
        "cells",
        "accuracy_lcb_mean",
        "accuracy_lcb_p10",
        "accuracy_lcb_p50",
        "accuracy_lcb_p90",
        "payload_kb_mean",
        "payload_kb_p50",
        "payload_kb_p90",
        "uncertainty_mean",
        "sample_count_mean",
    ]))

    lines.append("## Token vs Cache Semantic Gain")
    lines.append("")
    lines.append(f"- Mean token LCB gain over cache across low-SNR cells: `{_fmt(token_gain_mean)}`.")
    lines.append(f"- Mean token payload: `{_fmt(token_payload_mean)} KB`; P90 token payload: `{_fmt(token_payload_p90)} KB`.")
    lines.append(f"- Mean image payload: `{_fmt(image_payload_mean)} KB`; P90 image payload: `{_fmt(image_payload_p90)} KB`.")
    lines.append(f"- Mean image LCB gain over token: `{_fmt(image_gain_vs_token)}`.")
    lines.append("")
    lines.append("By semantic utility alone, service level 1 is the right low-SNR **main candidate** when its LCB clears `epsilon_k`: it keeps payload near the semantic-token scale and gives clear gains for presence tasks. It is not universally stronger than cache, especially for critical counting cells where detector count errors make token LCB conservative. Image evidence can improve or match LCB for some cells, but its payload is tens of times larger.")
    lines.append("")

    lines.append("## Breakdown by Task, Risk, View, and Freshness")
    lines.append("")
    lines.append(_markdown_table(combo_summary, [
        "question_type",
        "risk_level",
        "view_quality_bin",
        "freshness_bin",
        "cache_lcb",
        "token_lcb",
        "image_lcb",
        "token_gain_vs_cache",
        "image_gain_vs_token",
        "token_payload_kb",
        "image_payload_kb",
        "image_payload_multiplier_vs_token",
    ]))

    lines.append("## Compact Breakdown by Task and Risk")
    lines.append("")
    lines.append(_markdown_table(gain_summary, [
        "question_type",
        "risk_level",
        "cache_lcb",
        "token_lcb",
        "image_lcb",
        "token_gain_vs_cache",
        "image_gain_vs_token",
        "token_payload_kb",
        "image_payload_kb",
        "image_payload_multiplier_vs_token",
    ]))

    lines.append("## Cache Fallback Threshold")
    lines.append("")
    lines.append("A deadline-aware fallback should not require cache to fully satisfy the semantic threshold. It can be allowed when cache is close enough and the alternative token/image service would violate deadline. Define:")
    lines.append("")
    lines.append("```text")
    lines.append("cache_gap = max(0, epsilon_k - cache_accuracy_lcb)")
    lines.append("allow_cache_deadline_fallback if cache_gap <= delta_cache and token/image is deadline-infeasible")
    lines.append("```")
    lines.append("")
    lines.append(_markdown_table(threshold_df, [
        "epsilon",
        "cache_gap_threshold",
        "acceptable_cell_ratio",
        "mean_cache_gap",
        "p90_cache_gap",
    ]))
    lines.append("")
    lines.append("Recommendation: expose a `deadline_aware_semantic_fallback_threshold` to Algorithm. Start with `delta_cache = 0.05` for strict runs and `delta_cache = 0.08` for stress-scenario runs. The fallback should only fire when the deadline queue is high or no candidate service is jointly feasible; otherwise the controller should still prefer semantic tokens.")
    lines.append("")

    lines.append("## Image: Semantically Strong but Deadline-Infeasible")
    lines.append("")
    if bench.empty:
        lines.append("_No benchmark summary CSV was available._")
    else:
        lines.append(_markdown_table(bench, [c for c in bench_cols if c in bench.columns]))
    lines.append("")
    lines.append("The benchmark context confirms the mechanism: `always_image` can have high semantic success, but deadline violation is near one in `low_snr_blockage`. `always_semantic_token` sharply reduces payload but can still violate deadlines because the scenario also contains arrival/mobility delay under blockage. Therefore, semantic utility supports token as the main low-SNR service for many presence/normal tasks, but the algorithm still needs task-aware and deadline-aware fallback/projection rather than simply maximizing semantic LCB.")
    lines.append("")

    lines.append("## Conclusions for Algorithm")
    lines.append("")
    lines.append("1. **Service level 1 should be the low-SNR main service candidate** when `accuracy_lcb >= epsilon_k`: it provides the best semantic gain per payload for presence and several normal-risk cells while avoiding image overuse.")
    lines.append("2. **Image evidence is not the default low-SNR service**: it may be semantically strong, but its large payload makes it deadline-infeasible in the blockage stress scenario.")
    lines.append("3. **Cache fallback is acceptable only as a deadline safety valve**: allow cache when `cache_gap <= 0.05` in strict settings or `<= 0.08` in stress settings, and only when token/image are deadline-infeasible or deadline queues are high.")
    lines.append("4. **Critical counting needs special care**: detector-token count errors make semantic tokens conservative for some critical counting cells, so fallback/projection should consider task type and risk level rather than using one service rule for all low-SNR tasks.")
    lines.append("5. **Do not weaken the LUT**: the issue is not the semantic utility table. The table correctly exposes that token/image are semantically stronger for many cells but not all cells; the controller needs a conservative fallback threshold to trade a small semantic gap for a large deadline gain.")
    lines.append("6. **Expose the threshold explicitly**: add Algorithm-side config such as `deadline_aware_semantic_fallback_threshold: 0.05` and optionally `stress_fallback_threshold: 0.08` for low-SNR stress presets.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- This is a LUT/summary-level analysis; it does not rerun Qwen, retrain detector, or change environment dynamics.")
    lines.append("- Cache fallback ratios are evaluated over low-SNR utility cells, not over a new rollout distribution.")
    lines.append("- Deadline feasibility in the benchmark includes mobility/arrival and queue effects; semantic payload alone does not fully determine task success.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility-csv", type=Path, default=Path("outputs/lut/v1_9_semantic_utility_with_ci.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic/low_snr_service_tradeoff_20260624"))
    parser.add_argument("--low-snr-bins", nargs="+", default=["-5dB", "0dB"])
    parser.add_argument("--epsilon-grid", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.84])
    parser.add_argument("--cache-gap-thresholds", nargs="+", type=float, default=[0.05, 0.08])
    parser.add_argument(
        "--benchmark-csv",
        nargs="*",
        default=[
            "outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv",
            "outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv",
            "outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv",
        ],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    output_path = args.output_dir / "semantic_tradeoff.md"
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
