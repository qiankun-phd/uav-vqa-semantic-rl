#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import ensure_parent, load_config, resolve_path
from vqa_semcom.degradation.channel import degrade_image
from vqa_semcom.evidence.builder import (
    build_lightweight_evidence,
    build_vlm_prompt,
    image_path_for_task,
    load_objects_by_image,
    read_tasks_csv,
    select_vlm_tasks,
)
from vqa_semcom.quality.lut_builder import estimate_accuracy
from vqa_semcom.vlm.answer import check_answer
from vqa_semcom.vlm.evaluator import evaluate_prediction, make_evaluator


PREDICTION_FIELDNAMES = [
    "image_id",
    "question_type",
    "question",
    "ground_truth_answer",
    "target_class",
    "object_count",
    "risk_level",
    "epsilon_k",
    "tau_k",
    "service_level",
    "channel_bin",
    "view_quality_bin",
    "freshness_bin",
    "evidence_type",
    "evidence_repr",
    "payload_bytes",
    "image_path",
    "predicted_answer",
    "normalized_prediction",
    "correct",
    "latency_sec",
    "model_name",
]


def _cache_prediction(task: dict[str, str], channel_bin: str, freshness_bin: str, cfg: dict) -> tuple[str, str, bool]:
    acc = estimate_accuracy(
        question_type=task["question_type"],
        service_level=0,
        channel_bin=channel_bin,
        view_quality_bin=task["view_quality_bin"],
        freshness_bin=freshness_bin,
        risk_level=task["risk_level"],
        evaluator_cfg=cfg["evaluator"],
    )
    from vqa_semcom.vlm.evaluator import deterministic_unit

    unit = deterministic_unit(task["image_id"], task["question"], channel_bin, freshness_bin, "cache")
    if unit < acc:
        prediction = task["answer"]
    elif task["question_type"] == "presence":
        prediction = "no" if task["answer"].strip().lower() == "yes" else "yes"
    else:
        try:
            count = int(float(task["answer"]))
        except ValueError:
            count = 0
        prediction = str(max(0, count + 2))
    checked = check_answer(
        task["question_type"],
        prediction,
        task["answer"],
        tolerance_ratio=float(cfg.get("vlm", {}).get("count_tolerance_ratio", 0.10)),
    )
    return prediction, checked.normalized_prediction, checked.correct


def _payload_bytes(service_level: int, evidence_type: str, evidence_repr: str, image_path: str = "") -> int:
    if service_level == 0 or evidence_type == "cache":
        return 0
    if evidence_type == "lightweight":
        return len(evidence_repr.encode("utf-8"))
    if evidence_type in {"image", "roi_image"}:
        candidate_text = image_path or evidence_repr
        if candidate_text:
            candidate = Path(candidate_text)
            if candidate.exists():
                return candidate.stat().st_size
        return 0
    return len(evidence_repr.encode("utf-8"))


def _backfill_payload_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row.get("payload_bytes") not in {None, ""}:
            continue
        try:
            service_level = int(row.get("service_level", "0") or 0)
        except ValueError:
            service_level = 0
        row["payload_bytes"] = str(
            _payload_bytes(
                service_level,
                row.get("evidence_type", ""),
                row.get("evidence_repr", ""),
                row.get("image_path", ""),
            )
        )


def _write_rows(rows: list[dict[str, str]], path: Path) -> None:
    ensure_parent(path)
    _backfill_payload_rows(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    _backfill_payload_rows(rows)
    return rows


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row["image_id"],
        row["question"],
        str(row["service_level"]),
        row["channel_bin"],
        row["freshness_bin"],
        row["evidence_type"],
    )


def _merge_vlm_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = dict(cfg)
    vlm_cfg = dict(cfg.get("vlm", {}))
    if args.model_name:
        vlm_cfg["model_name"] = args.model_name
    if args.fallback_model_name:
        vlm_cfg["fallback_model_name"] = args.fallback_model_name
    if args.max_pixels is not None:
        vlm_cfg["max_pixels"] = args.max_pixels
    if args.min_pixels is not None:
        vlm_cfg["min_pixels"] = args.min_pixels
    if args.max_new_tokens is not None:
        vlm_cfg["max_new_tokens"] = args.max_new_tokens
    cfg["vlm"] = vlm_cfg
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--evaluator", choices=["mock", "qwen"], default="mock")
    parser.add_argument("--service-levels", default="0,1,2")
    parser.add_argument("--channels", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--fallback-model-name", default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _merge_vlm_overrides(load_config(args.config), args)
    vlm_cfg = cfg.get("vlm", {})
    service_levels = [int(x.strip()) for x in args.service_levels.split(",") if x.strip()]
    channels = [x.strip() for x in args.channels.split(",")] if args.channels else list(cfg["bins"]["channel"])
    freshness_bins = list(cfg["bins"]["freshness"])
    question_types = set(vlm_cfg.get("question_types", ["presence", "counting"]))
    tasks = read_tasks_csv(resolve_path(cfg["paths"]["tasks_csv"]))
    selected_tasks = select_vlm_tasks(
        tasks,
        question_types=question_types,
        limit_images=args.limit_images,
        max_tasks=args.max_tasks,
        max_tasks_per_image=int(vlm_cfg.get("max_tasks_per_image", 4)),
    )
    if not selected_tasks:
        raise RuntimeError("No V1 VLM tasks selected. Build V0 tasks first or adjust question types.")

    visdrone_root = resolve_path(cfg["paths"]["visdrone_val"])
    objects_by_image = load_objects_by_image(visdrone_root, [task["image_id"] for task in selected_tasks])
    evaluator = make_evaluator(args.evaluator, cfg) if any(level in service_levels for level in [1, 2]) else None
    out_path = resolve_path(cfg["paths"]["vlm_predictions_csv"])
    existing_rows = _read_existing_rows(out_path) if args.resume else []
    existing_keys = {_row_key(row) for row in existing_rows}
    prediction_cache: dict[tuple[str, str, int, str], tuple[str, str, bool, float, str, str, str, int]] = {}
    rows: list[dict[str, str]] = list(existing_rows)
    degraded_dir = resolve_path(cfg["paths"].get("degraded_image_dir", "outputs/vlm/degraded_images"))

    for task in selected_tasks:
        for service_level in service_levels:
            for channel_bin in channels:
                for freshness_bin in freshness_bins:
                    evidence_repr = ""
                    evidence_type = "cache"
                    image_path = ""
                    row_stub = {
                        "image_id": task["image_id"],
                        "question": task["question"],
                        "service_level": str(service_level),
                        "channel_bin": channel_bin,
                        "freshness_bin": freshness_bin,
                        "evidence_type": "cache" if service_level == 0 else "lightweight" if service_level == 1 else "image",
                    }
                    if args.resume and _row_key(row_stub) in existing_keys:
                        continue
                    if service_level == 0:
                        predicted, normalized, correct = _cache_prediction(task, channel_bin, freshness_bin, cfg)
                        latency = 0.0
                        model_name = "cache-simulator"
                        payload_bytes = 0
                    else:
                        cache_key = (task["image_id"], task["question"], service_level, channel_bin)
                        if cache_key not in prediction_cache:
                            if evaluator is None:
                                raise RuntimeError("Evaluator is required for service levels 1 and 2.")
                            if service_level == 1:
                                objects = objects_by_image.get(task["image_id"], [])
                                evidence_repr = build_lightweight_evidence(task, objects, channel_bin, cfg)
                                prompt = build_vlm_prompt(task, evidence_text=evidence_repr)
                                prompt = f"service_level=1 channel={channel_bin}\n{prompt}"
                                image_input = None
                                evidence_type = "lightweight"
                            elif service_level == 2:
                                source_image = image_path_for_task(visdrone_root, task["image_id"])
                                image_input = degrade_image(source_image, degraded_dir, channel_bin, cfg)
                                evidence_repr = str(image_input)
                                prompt = build_vlm_prompt(task)
                                prompt = f"service_level=2 channel={channel_bin}\n{prompt}"
                                evidence_type = "image"
                            else:
                                raise ValueError(f"Unsupported service level for V1: {service_level}")
                            pred = evaluate_prediction(evaluator, task, prompt, image_input, cfg)
                            payload_bytes = _payload_bytes(
                                service_level,
                                evidence_type,
                                evidence_repr,
                                str(image_input) if image_input is not None else "",
                            )
                            prediction_cache[cache_key] = (
                                pred.predicted_answer,
                                pred.normalized_prediction,
                                pred.correct,
                                pred.latency_sec,
                                pred.model_name,
                                evidence_type,
                                evidence_repr,
                                payload_bytes,
                            )
                        predicted, normalized, correct, latency, model_name, evidence_type, evidence_repr, payload_bytes = prediction_cache[cache_key]
                        if evidence_type == "image":
                            image_path = evidence_repr
                    rows.append(
                        {
                            "image_id": task["image_id"],
                            "question_type": task["question_type"],
                            "question": task["question"],
                            "ground_truth_answer": task["answer"],
                            "target_class": task["target_class"],
                            "object_count": task["object_count"],
                            "risk_level": task["risk_level"],
                            "epsilon_k": task["epsilon_k"],
                            "tau_k": task["tau_k"],
                            "service_level": str(service_level),
                            "channel_bin": channel_bin,
                            "view_quality_bin": task["view_quality_bin"],
                            "freshness_bin": freshness_bin,
                            "evidence_type": evidence_type,
                            "evidence_repr": evidence_repr,
                            "payload_bytes": str(payload_bytes),
                            "image_path": image_path,
                            "predicted_answer": predicted,
                            "normalized_prediction": normalized,
                            "correct": str(bool(correct)),
                            "latency_sec": f"{latency:.6f}",
                            "model_name": model_name,
                        }
                    )

    _write_rows(rows, out_path)
    print(f"selected_tasks={len(selected_tasks)} prediction_rows={len(rows)}")
    if args.resume:
        print(f"resumed_existing_rows={len(existing_rows)}")
    print(f"predictions_csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
