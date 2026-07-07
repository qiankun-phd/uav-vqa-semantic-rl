from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.semantic.utility import SemanticUtilityModel


SCENARIOS = (
    "normal_patrol",
    "disaster_hotspot",
    "low_snr_soft",
    "low_snr_blockage",
    "edge_overload",
    "utm_conflict",
)

SCENARIO_ALIASES = {
    "normal_patrol": "nominal_patrol",
    "low_snr_soft": "low_snr_blockage",
}

SCENARIO_SNR_BINS = {
    "normal_patrol": ["10dB", "15dB", "20dB"],
    "disaster_hotspot": ["5dB", "10dB", "15dB"],
    "low_snr_soft": ["0dB", "5dB", "10dB"],
    "low_snr_blockage": ["-5dB", "0dB"],
    "edge_overload": ["10dB", "15dB", "20dB"],
    "utm_conflict": ["5dB", "10dB", "15dB"],
}

FALLBACK_LAYOUTS = {
    "normal_patrol": {
        "risk_cycle": ["normal", "normal", "critical", "normal"],
        "freshness_cycle": ["fresh", "stale", "fresh", "stale"],
        "view_quality_cycle": ["medium", "good", "medium", "good"],
        "semantic_threshold_by_risk": {"normal": 0.56, "critical": 0.78, "high": 0.78},
    },
    "disaster_hotspot": {
        "risk_cycle": ["critical", "critical", "normal", "critical", "critical"],
        "freshness_cycle": ["stale", "expired", "fresh", "stale"],
        "view_quality_cycle": ["medium", "poor", "medium", "good"],
        "semantic_threshold_by_risk": {"normal": 0.62, "critical": 0.84, "high": 0.84},
    },
    "low_snr_blockage": {
        "risk_cycle": ["normal", "critical", "normal", "normal"],
        "freshness_cycle": ["fresh", "stale", "expired", "fresh"],
        "view_quality_cycle": ["medium", "poor", "medium", "good"],
        "semantic_threshold_by_risk": {"normal": 0.56, "critical": 0.78, "high": 0.78},
    },
    "edge_overload": {
        "risk_cycle": ["normal", "normal", "normal", "critical", "normal", "normal"],
        "freshness_cycle": ["fresh", "fresh", "fresh", "stale"],
        "view_quality_cycle": ["good", "good", "medium", "good"],
        "semantic_threshold_by_risk": {"normal": 0.54, "critical": 0.64, "high": 0.64},
    },
    "utm_conflict": {
        "risk_cycle": ["critical", "normal", "critical", "normal"],
        "freshness_cycle": ["stale", "fresh", "expired", "stale"],
        "view_quality_cycle": ["medium", "good", "medium", "poor"],
        "semantic_threshold_by_risk": {"normal": 0.60, "critical": 0.82, "high": 0.82},
    },
}


@dataclass
class PathRecord:
    scenario: str
    semantic_path: str
    task_type: str
    risk_level: str
    freshness_bin: str
    view_quality_bin: str
    snr_bin: str
    epsilon_k: float
    accuracy_lcb: float
    uncertainty: float
    quality_gap: float
    payload_kb: float
    cache_recommended: bool
    path_recommended: bool
    critical_task_recommended: bool
    stale_or_expired_cache_recommended: bool
    future_reuse_value_available: bool
    cache_update_limitation: bool


def _scenario_presets() -> dict[str, dict[str, Any]]:
    try:
        from vqa_semcom.sim.multi_uav_env import SEMANTIC_SCENARIO_PRESETS

        return SEMANTIC_SCENARIO_PRESETS
    except Exception:
        return {}


def _layout_for_scenario(scenario: str, presets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = SCENARIO_ALIASES.get(scenario, scenario)
    preset = presets.get(base, {})
    env = preset.get("env", {}) if isinstance(preset, dict) else {}
    layout = preset.get("task_layout", {}) if isinstance(preset, dict) else {}
    fallback = FALLBACK_LAYOUTS.get(scenario) or FALLBACK_LAYOUTS.get(base) or FALLBACK_LAYOUTS["normal_patrol"]
    return {
        "risk_cycle": list(layout.get("risk_cycle", fallback["risk_cycle"])),
        "freshness_cycle": list(layout.get("freshness_cycle", fallback["freshness_cycle"])),
        "view_quality_cycle": list(layout.get("view_quality_cycle", fallback["view_quality_cycle"])),
        "semantic_threshold_by_risk": dict(env.get("semantic_threshold_by_risk", fallback["semantic_threshold_by_risk"])),
    }


def _task_types(model: SemanticUtilityModel) -> list[str]:
    values = sorted({cell.question_type for cell in model.cells if cell.question_type})
    return values or ["presence", "counting"]


def _context_grid(scenario: str, model: SemanticUtilityModel) -> list[dict[str, Any]]:
    presets = _scenario_presets()
    layout = _layout_for_scenario(scenario, presets)
    risks = layout["risk_cycle"]
    freshness_values = layout["freshness_cycle"]
    view_values = layout["view_quality_cycle"]
    thresholds = layout["semantic_threshold_by_risk"]
    contexts: list[dict[str, Any]] = []
    for task_type in _task_types(model):
        for risk in sorted(set(risks), key=risks.index):
            epsilon = float(thresholds.get(risk, thresholds.get("normal", 0.7)))
            for freshness in sorted(set(freshness_values), key=freshness_values.index):
                for view in sorted(set(view_values), key=view_values.index):
                    for snr_bin in SCENARIO_SNR_BINS[scenario]:
                        contexts.append(
                            {
                                "question_type": task_type,
                                "risk_level": risk,
                                "freshness_bin": freshness,
                                "view_quality_bin": view,
                                "snr_bin": snr_bin,
                                "epsilon_k": epsilon,
                            }
                        )
    return contexts


def _path_recommended(path: str, utility: Any, cache_recommended: bool, risk_level: str, freshness_bin: str) -> bool:
    if path == "cache":
        return bool(cache_recommended)
    if path == "cache_update":
        return False
    if utility.semantic_quality_gap > 0.0:
        return False
    if risk_level == "critical" and utility.uncertainty > 0.45:
        return False
    if freshness_bin == "expired" and path == "image":
        return True
    return True


def build_records(model: SemanticUtilityModel) -> list[PathRecord]:
    records: list[PathRecord] = []
    for scenario in SCENARIOS:
        for ctx in _context_grid(scenario, model):
            cache = model.cache_quality_metrics(
                ctx["question_type"],
                ctx["snr_bin"],
                ctx["view_quality_bin"],
                ctx["freshness_bin"],
                ctx["risk_level"],
                ctx["epsilon_k"],
            )
            for path in ("cache", "token", "image", "cache_update"):
                utility = model.path_utility(
                    path,
                    ctx["question_type"],
                    ctx["snr_bin"],
                    ctx["view_quality_bin"],
                    ctx["freshness_bin"],
                    ctx["risk_level"],
                    ctx["epsilon_k"],
                )
                recommended = _path_recommended(path, utility, cache.recommended, ctx["risk_level"], ctx["freshness_bin"])
                stale_expired_cache_recommended = bool(
                    path == "cache"
                    and ctx["freshness_bin"] in {"stale", "expired"}
                    and cache.recommended
                )
                records.append(
                    PathRecord(
                        scenario=scenario,
                        semantic_path=path,
                        task_type=ctx["question_type"],
                        risk_level=ctx["risk_level"],
                        freshness_bin=ctx["freshness_bin"],
                        view_quality_bin=ctx["view_quality_bin"],
                        snr_bin=ctx["snr_bin"],
                        epsilon_k=ctx["epsilon_k"],
                        accuracy_lcb=float(utility.accuracy_lcb),
                        uncertainty=float(utility.uncertainty),
                        quality_gap=float(utility.semantic_quality_gap),
                        payload_kb=float(utility.payload_kb),
                        cache_recommended=bool(cache.recommended),
                        path_recommended=bool(recommended),
                        critical_task_recommended=bool(ctx["risk_level"] == "critical" and recommended),
                        stale_or_expired_cache_recommended=stale_expired_cache_recommended,
                        future_reuse_value_available=False,
                        cache_update_limitation=path == "cache_update",
                    )
                )
    return records


def _rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.astype(bool).mean())


def summarize(records: list[PathRecord]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(row) for row in records])
    grouped = df.groupby(["scenario", "semantic_path"], as_index=False)
    summary = grouped.agg(
        cells=("accuracy_lcb", "size"),
        accuracy_lcb=("accuracy_lcb", "mean"),
        uncertainty=("uncertainty", "mean"),
        quality_gap=("quality_gap", "mean"),
        payload_kb=("payload_kb", "mean"),
        path_recommended_rate=("path_recommended", _rate),
        cache_recommended_rate=("cache_recommended", _rate),
        critical_task_recommendation_rate=("critical_task_recommended", _rate),
        stale_expired_cache_recommendation_rate=("stale_or_expired_cache_recommended", _rate),
        future_reuse_value_available=("future_reuse_value_available", _rate),
        cache_update_limitation_rate=("cache_update_limitation", _rate),
    )
    order = {name: idx for idx, name in enumerate(SCENARIOS)}
    path_order = {"cache": 0, "token": 1, "image": 2, "cache_update": 3}
    summary["_scenario_order"] = summary["scenario"].map(order)
    summary["_path_order"] = summary["semantic_path"].map(path_order)
    return summary.sort_values(["_scenario_order", "_path_order"]).drop(columns=["_scenario_order", "_path_order"])


def _read_short_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = [
        "scenario",
        "benchmark_policy",
        "semantic_path_cache_ratio_mean",
        "semantic_path_token_ratio_mean",
        "semantic_path_image_ratio_mean",
        "semantic_path_cache_update_ratio_mean",
        "cache_eligible_ratio_mean",
        "deadline_violation_rate_mean",
        "task_success_rate_mean",
    ]
    return df[[col for col in cols if col in df.columns]]


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No data._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_report(summary: pd.DataFrame, records: pd.DataFrame, benchmark: pd.DataFrame, output_path: Path) -> None:
    critical_cache_violations = records[
        (records["semantic_path"] == "cache")
        & (records["risk_level"] == "critical")
        & (records["freshness_bin"].isin(["stale", "expired"]))
        & (records["path_recommended"])
    ]
    expired_cache_violations = records[
        (records["semantic_path"] == "cache")
        & (records["freshness_bin"] == "expired")
        & (records["path_recommended"])
    ]
    cache_update = summary[summary["semantic_path"] == "cache_update"]
    edge_utm = benchmark[benchmark["scenario"].isin(["edge_overload", "utm_conflict"])] if not benchmark.empty else pd.DataFrame()

    lines: list[str] = []
    lines.append("# Semantic Path Utility Diagnosis")
    lines.append("")
    lines.append("This report checks cache/path utility recommendations for semantic-path cache-defer control. It does not modify the original VQA predictions, semantic LUT, environment dynamics, or PPO logic.")
    lines.append("")
    lines.append("## Rule Checks")
    lines.append("")
    lines.append(f"- Critical stale/expired cache recommendation violations: `{len(critical_cache_violations)}`.")
    lines.append(f"- Expired cache recommendation violations: `{len(expired_cache_violations)}`.")
    lines.append("- Cache update future reuse value available in utility layer: `False`.")
    lines.append("- Conclusion: `cache_update` must not be recommended solely because cache is missing. The utility layer can score the current token/image update path, but it has no future reuse value model. PPO/environment logic must supply reuse value, cache pressure, or expected future hit probability before actively preferring `cache_update`.")
    lines.append("")
    lines.append("## Scenario Path Utility Summary")
    lines.append("")
    lines.append(_md_table(summary, [
        "scenario",
        "semantic_path",
        "cells",
        "accuracy_lcb",
        "uncertainty",
        "quality_gap",
        "payload_kb",
        "path_recommended_rate",
        "cache_recommended_rate",
        "critical_task_recommendation_rate",
        "stale_expired_cache_recommendation_rate",
        "cache_update_limitation_rate",
    ]))
    lines.append("## Edge/UTM Path-Ratio Context from Existing Short Runs")
    lines.append("")
    if edge_utm.empty:
        lines.append("_No existing semantic-path short-run summary found._")
    else:
        lines.append(_md_table(edge_utm, list(edge_utm.columns)))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `edge_overload`: token paths should remain attractive because image payload is high and cache update has no utility-layer future reuse estimate. If PPO selects `cache_update`, that should be justified by environment-side future cache value, not by this utility score alone.")
    lines.append("- `utm_conflict`: critical/stale/expired cache is blocked by the cache recommendation rule. Cache update may be useful operationally only if it reduces future UTM-constrained revisits or cache misses; that value is outside the current utility API.")
    lines.append("- `low_snr_blockage` and `low_snr_soft`: token is the lightweight evidence path when LCB clears epsilon; cache is a deadline fallback only when fresh and semantically eligible.")
    lines.append("- `disaster_hotspot`: stricter epsilon makes cache less eligible; cache update should be treated as a deliberate future-cache investment, not as an automatic replacement for missing cache.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("Keep cache/path utility as a conservative evaluator. Add a separate Algorithm/Environment-side `future_reuse_value` or `cache_update_value` field before using `cache_update` as an actively recommended path. Until then, reports and policies should treat `cache_update` as a candidate action with current-task token/image utility plus an explicit limitation.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility-csv", type=Path, default=Path("outputs/lut/v1_9_semantic_utility_with_ci.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic/semantic_path_utility_diagnosis_20260624"))
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=Path("outputs/rl/semantic_path_cache_defer_short_20260624/scenario_comparison_summary.csv"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = SemanticUtilityModel.from_csv(args.utility_csv)
    records = build_records(model)
    records_df = pd.DataFrame([asdict(row) for row in records])
    summary = summarize(records)
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    benchmark = _read_short_benchmark(args.benchmark_summary)
    write_report(summary, records_df, benchmark, args.output_dir / "report.md")
    print(args.output_dir / "report.md")
    print(summary_path)


if __name__ == "__main__":
    main()
