#!/usr/bin/env python3
"""P1-3 ablation: let the receiver VLM read the *transmitted token text* (s1
evidence_repr, which already embeds the channel-degraded detections) and answer
with no image input, mirroring the legacy service_level=1 VLM branch of
run_v1_detector_eval.py bit-for-bit (same prompt prefix, same build_vlm_prompt,
same check_answer tolerance). Compares against the calibrated symbolic decoder
(the `correct` column of the same s1 rows) on identical cells.

Scope control (paper P1-3): Rician, SNR in {-5,5,20} dB, all TEST-split
questions (crc32(image_id)%100 < 20), one inference per unique
(image_id, question, snr_bin) -- freshness replicas share the s1 prediction.

Usage:
  python scripts/p1_vlm_reads_tokens.py \
      --pred-csv outputs/vlm/v3_0_rician_predictions.csv \
      --config configs/v2_0_ldpc_channel.yaml \
      --snr-bins="-5dB,5dB,20dB" \
      --out outputs/vlm/p1_vlm_reads_tokens_rician_predictions.csv --resume
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import zlib
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
    "image_id", "question_type", "question", "ground_truth_answer", "snr_bin",
    "channel_bin", "payload_bytes", "predicted_answer", "normalized_prediction",
    "correct", "latency_sec", "model_name",
    "symbolic_correct", "raw_decoder_correct", "decoder_mode",
]


def is_test(image_id: str, frac: float = 0.2) -> bool:
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", default="outputs/vlm/v3_0_rician_predictions.csv")
    ap.add_argument("--config", default="configs/v2_0_ldpc_channel.yaml")
    ap.add_argument("--snr-bins", default="-5dB,5dB,20dB")
    ap.add_argument("--out", default="outputs/vlm/p1_vlm_reads_tokens_rician_predictions.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    snr_keep = {s.strip() for s in args.snr_bins.split(",") if s.strip()}
    df = pd.read_csv(args.pred_csv, dtype={"image_id": str})
    df = df[(df.service_level == 1) & df.snr_bin.isin(snr_keep)]
    df = df[df.image_id.map(is_test)]
    df = df.drop_duplicates(subset=["image_id", "question", "snr_bin"]).copy()
    df = df.sort_values(["question_type", "image_id", "question", "snr_bin"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"cells to evaluate: {len(df)} "
          f"({df.question_type.value_counts().to_dict()})", flush=True)

    out_path = Path(resolve_path(args.out))
    done: set[tuple[str, str, str]] = set()
    rows: list[dict[str, str]] = []
    if args.resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        done = {(r["image_id"], r["question"], r["snr_bin"]) for r in rows}
        print(f"resume: {len(done)} cells already done", flush=True)

    evaluator = make_evaluator("qwen", cfg)
    print(f"model: {evaluator.model_name}", flush=True)
    tol = float(cfg.get("vlm", {}).get("count_tolerance_ratio", 0.10))

    n_new = 0
    t0 = time.time()
    for _, r in df.iterrows():
        key = (r.image_id, r.question, r.snr_bin)
        if key in done:
            continue
        task = {
            "image_id": r.image_id,
            "question": r.question,
            "question_type": r.question_type,
            "answer": str(r.ground_truth_answer),
        }
        # exact mirror of the legacy s1 VLM branch in run_v1_detector_eval.py
        prompt = build_vlm_prompt(task, evidence_text=str(r.evidence_repr))
        prompt = (f"service_level=1 snr_bin={r.snr_bin or chr(39)+chr(39)} "
                  f"channel={r.channel_bin} evidence_source=detector\n{prompt}")
        t1 = time.perf_counter()
        predicted = evaluator.predict(task, prompt, image_path=None)
        latency = time.perf_counter() - t1
        chk = check_answer(r.question_type, predicted, str(r.ground_truth_answer), tolerance_ratio=tol)
        rows.append({
            "image_id": r.image_id,
            "question_type": r.question_type,
            "question": r.question,
            "ground_truth_answer": str(r.ground_truth_answer),
            "snr_bin": r.snr_bin,
            "channel_bin": r.channel_bin,
            "payload_bytes": str(r.payload_bytes),
            "predicted_answer": predicted,
            "normalized_prediction": chk.normalized_prediction,
            "correct": str(bool(chk.correct)),
            "latency_sec": f"{latency:.6f}",
            "model_name": evaluator.model_name,
            "symbolic_correct": str(bool(r.correct)),
            "raw_decoder_correct": str(r.raw_decoder_correct),
            "decoder_mode": str(r.decoder_mode),
        })
        n_new += 1
        if n_new % 25 == 0:
            ensure_parent(out_path)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            rate = n_new / max(1e-9, time.time() - t0)
            print(f"done {n_new} new ({rate:.2f}/s)", flush=True)
    ensure_parent(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
