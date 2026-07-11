#!/usr/bin/env python3
"""O2 probe step 3 (GPU): re-run the two VLM receive channels with the
free-phrased questions, mirroring the existing pipelines bit-for-bit except
for the Question line:

  --mode tokens : VLM reads the transmitted token text (evidence_repr from the
                  original s1 cell, unchanged) -- mirrors
                  scripts/p1_vlm_reads_tokens.py / the legacy s1 VLM branch.
  --mode image  : VLM reads the channel-degraded image (same degraded jpg the
                  original s2 cell used) -- mirrors the service_level=2 branch
                  of scripts/run_v1_detector_eval.py.

Usage:
  python scripts/o2_vlm_runner.py --mode tokens --out outputs/vlm/o2_vlm_tokens_predictions.csv --resume
  python scripts/o2_vlm_runner.py --mode image  --out outputs/vlm/o2_vlm_image_predictions.csv  --resume
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from vqa_semcom.config import load_config, ensure_parent, resolve_path
from vqa_semcom.evidence.builder import build_vlm_prompt
from vqa_semcom.vlm.answer import check_answer
from vqa_semcom.vlm.evaluator import make_evaluator

FIELDS = [
    "image_id", "question_type", "question_orig", "paraphrase_id", "template_id",
    "question_para", "mode", "ground_truth_answer", "snr_bin", "channel_bin",
    "predicted_answer", "normalized_prediction", "correct", "latency_sec",
    "model_name",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tokens", "image"], required=True)
    ap.add_argument("--in-csv", default="outputs/vlm/o2_paraphrases.csv")
    ap.add_argument("--config", default="configs/v2_0_ldpc_channel.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(resolve_path(args.in_csv), dtype={"image_id": str})
    df = df.sort_values(["question_type", "image_id", "question_orig", "paraphrase_id"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"[{args.mode}] items: {len(df)}", flush=True)

    out_path = Path(resolve_path(args.out))
    rows: list[dict[str, str]] = []
    done: set[tuple[str, str]] = set()
    if args.resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        done = {(r["image_id"], r["question_para"]) for r in rows}
        print(f"resume: {len(done)} items already done", flush=True)

    evaluator = make_evaluator("qwen", cfg)
    print(f"model: {evaluator.model_name}", flush=True)
    tol = float(cfg.get("vlm", {}).get("count_tolerance_ratio", 0.10))

    def flush_rows() -> None:
        ensure_parent(out_path)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    n_new = 0
    t0 = time.time()
    for _, r in df.iterrows():
        key = (r.image_id, r.question_para)
        if key in done:
            continue
        task = {
            "image_id": r.image_id,
            "question": str(r.question_para),
            "question_type": r.question_type,
            "answer": str(r.ground_truth_answer),
        }
        if args.mode == "tokens":
            # exact mirror of the legacy s1 VLM branch / p1_vlm_reads_tokens.py
            prompt = build_vlm_prompt(task, evidence_text=str(r.evidence_repr))
            prompt = (f"service_level=1 snr_bin={r.snr_bin or ''} "
                      f"channel={r.channel_bin} evidence_source=detector\n{prompt}")
            image_path = None
        else:
            # exact mirror of the service_level=2 branch of run_v1_detector_eval.py
            prompt = build_vlm_prompt(task)
            prompt = (f"service_level=2 snr_bin={r.snr_bin or ''} "
                      f"channel={r.channel_bin} evidence_source=image\n{prompt}")
            image_path = Path(str(r.degraded_image_path))
            if not image_path.exists():
                raise FileNotFoundError(image_path)
        t1 = time.perf_counter()
        predicted = evaluator.predict(task, prompt, image_path=image_path)
        latency = time.perf_counter() - t1
        chk = check_answer(r.question_type, predicted, str(r.ground_truth_answer), tolerance_ratio=tol)
        rows.append({
            "image_id": r.image_id,
            "question_type": r.question_type,
            "question_orig": r.question_orig,
            "paraphrase_id": str(r.paraphrase_id),
            "template_id": r.template_id,
            "question_para": r.question_para,
            "mode": args.mode,
            "ground_truth_answer": str(r.ground_truth_answer),
            "snr_bin": r.snr_bin,
            "channel_bin": r.channel_bin,
            "predicted_answer": predicted,
            "normalized_prediction": chk.normalized_prediction,
            "correct": str(bool(chk.correct)),
            "latency_sec": f"{latency:.6f}",
            "model_name": evaluator.model_name,
        })
        n_new += 1
        if n_new % 25 == 0:
            flush_rows()
            rate = n_new / max(1e-9, time.time() - t0)
            print(f"done {n_new} new ({rate:.2f}/s)", flush=True)
    flush_rows()
    print(f"wrote {len(rows)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
