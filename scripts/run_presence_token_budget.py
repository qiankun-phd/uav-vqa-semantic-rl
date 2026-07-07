#!/usr/bin/env python3
"""Q2/W2: presence top-t token-budget sweep through the VLM, with answer confidence.

The transmitter keeps only the top-t detections (confidence order, nested-prefix
style) before building the s1 token evidence; the evidence is degraded by the
same deterministic LDPC/rician channel hash as the logged runs and fed to the
VLM as a text prompt (no image). The VLM's first answer-token probability is
logged as answer_confidence (output_scores / return_dict_in_generate).

Task set: the presence tasks of the logged v26 rician main run, TEST split only
(same crc32 split as build_comparison_v2), so results sit next to the symbolic
token_budget_sweep numbers. Appends per-sample rows to a raw CSV (resume-safe).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_comparison_v2 as bc  # noqa: E402
import build_token_budget_sweep as tbs  # noqa: E402
import run_v1_detector_eval as ev  # noqa: E402
from vqa_semcom.config import load_config, resolve_path  # noqa: E402
from vqa_semcom.detector.visdrone_yolo import build_detector_lightweight_evidence  # noqa: E402
from vqa_semcom.evidence.builder import build_vlm_prompt, read_tasks_csv  # noqa: E402
from vqa_semcom.snr import channel_bin_from_snr, snr_bin_label  # noqa: E402
from vqa_semcom.vlm.answer import check_answer  # noqa: E402
from vqa_semcom.vlm.evaluator import make_evaluator  # noqa: E402

RAW_FIELDS = [
    "channel", "t_budget", "snr_db", "snr_bin", "image_id", "question", "presence_polarity",
    "ground_truth_answer", "predicted_answer", "normalized_prediction", "correct",
    "answer_confidence", "payload_bytes", "tokens_sent", "model_name",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v26_rician_main.json")
    ap.add_argument("--channel", default="rician")
    ap.add_argument("--pred-csv", default="outputs/vlm/v26_rician_main_predictions.csv",
                    help="logged run defining the presence task subset")
    ap.add_argument("--detections", default="outputs/detector/v2_0_snr_detections.csv",
                    help="conf>=0.25 detection cache matching the s1 baseline")
    ap.add_argument("--t-grid", default="1,2,4,8,16,32,0", help="0 = full")
    ap.add_argument("--snr-bins", default="-5,0,5,10,15,20")
    ap.add_argument("--out-raw", default="outputs/reports/presence_token_budget_raw.csv")
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--all-split", action="store_true", help="use all tasks instead of test split")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t_grid = [int(x) for x in args.t_grid.split(",") if x.strip()]
    snr_values = [float(x) for x in args.snr_bins.split(",") if x.strip()]

    # presence tasks restricted to the logged run's selection
    logged = set()
    polarity_hint = {}
    for r in csv.DictReader(open(resolve_path(args.pred_csv))):
        if r["question_type"] == "presence":
            logged.add((r["image_id"], r["question"]))
            polarity_hint[(r["image_id"], r["question"])] = r.get("presence_polarity", "")
    tasks = [t for t in read_tasks_csv(resolve_path(cfg["paths"]["tasks_csv"]))
             if t["question_type"] == "presence" and (t["image_id"], t["question"]) in logged]
    if not args.all_split:
        tasks = [t for t in tasks if bc.is_test(t["image_id"])]
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    print(f"presence tasks: {len(tasks)} (split={'all' if args.all_split else 'test'}) "
          f"x t={t_grid} x snr={snr_values} -> {len(tasks) * len(t_grid) * len(snr_values)} VLM calls")

    det_cache = tbs.load_detections(resolve_path(args.detections))

    out_path = resolve_path(args.out_raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for r in csv.DictReader(open(out_path)):
            done.add((r["image_id"], r["question"], r["t_budget"], r["snr_bin"]))
        print(f"resume: {len(done)} rows already logged")
    write_header = not (args.resume and out_path.exists())

    evaluator = make_evaluator("qwen", cfg)
    n_new = 0
    with open(out_path, "a" if not write_header else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if write_header:
            writer.writeheader()
        for task in tasks:
            recs_full = det_cache.get(task["image_id"], [])
            for t in t_grid:
                recs = recs_full[:t] if t else recs_full
                t_label = str(t) if t else "full"
                for snr_db in snr_values:
                    snr_bin = snr_bin_label(snr_db)
                    if (task["image_id"], task["question"], t_label, snr_bin) in done:
                        continue
                    channel_bin = channel_bin_from_snr(snr_db)
                    evidence = build_detector_lightweight_evidence(task, recs, snr_bin, cfg)
                    prompt = build_vlm_prompt(task, evidence_text=evidence)
                    prompt = (f"service_level=1 snr_bin={snr_bin} channel={channel_bin} "
                              f"evidence_source=detector token_budget={t_label}\n{prompt}")
                    answer, confidence = evaluator.predict_with_confidence(task, prompt, None)
                    chk = check_answer("presence", answer, task["answer"])
                    writer.writerow({
                        "channel": args.channel,
                        "t_budget": t_label,
                        "snr_db": f"{snr_db:g}",
                        "snr_bin": snr_bin,
                        "image_id": task["image_id"],
                        "question": task["question"],
                        "presence_polarity": task.get("presence_polarity", "") or polarity_hint.get((task["image_id"], task["question"]), ""),
                        "ground_truth_answer": task["answer"],
                        "predicted_answer": answer,
                        "normalized_prediction": chk.normalized_prediction,
                        "correct": str(bool(chk.correct)),
                        "answer_confidence": f"{confidence:.6f}",
                        "payload_bytes": str(ev._payload_bytes(1, "lightweight", evidence, "")),
                        "tokens_sent": str(len(recs)),
                        "model_name": evaluator.model_name,
                    })
                    n_new += 1
                    if n_new % 50 == 0:
                        f.flush()
                        print(f"progress: +{n_new} rows (last: t={t_label} snr={snr_bin} img={task['image_id']})", flush=True)
    print(f"done: wrote {n_new} new rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
