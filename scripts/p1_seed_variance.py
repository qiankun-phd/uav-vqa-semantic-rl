#!/usr/bin/env python3
"""P1-4: channel-realization variance for the paper's two most-cited cells,
M1_image and M0_naive at -5 dB Rician, with >=10 independent channel seeds.

The campaign pipeline seeds every transmission deterministically from
(image_id, snr_bin), so run-to-run repeats are bit-identical by design; this
script re-draws the *channel realization* (fading/outage draw for M1's
rate-adaptive path, per-tile frame losses for M0_naive's fixed-rate LDPC path)
under seeds 1..K and reports mean +/- std test accuracy per seed.

Phases (GPU discipline: transmit is CPU-only and may run beside training):
  --phase transmit : regenerate degraded images per seed (CPU)
  --phase answer   : Qwen answers all (question, seed) cells (GPU)

Usage:
  python scripts/p1_seed_variance.py --phase transmit --seeds 10
  python scripts/p1_seed_variance.py --phase answer --seeds 10 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from vqa_semcom.config import load_config, resolve_path, ensure_parent
from vqa_semcom.evidence.builder import build_vlm_prompt, image_path_for_task, read_tasks_csv, select_vlm_tasks
from vqa_semcom.snr import snr_bin_label, channel_bin_from_snr
from vqa_semcom.vlm.evaluator import evaluate_prediction, make_evaluator
from vqa_semcom.degradation.digital_link import (
    _seed_from, build_link_config, load_or_calibrate,
    transmit_image, transmit_image_rate_adaptive,
)

SNR_DB = -5.0
OUT_DIR = "outputs/vlm/p1_seedvar"
FIELDS = ["method", "seed", "image_id", "question_type", "question", "ground_truth_answer",
          "predicted_answer", "normalized_prediction", "correct", "model_name"]

M1_CONFIGS = ["configs/v2_0_ldpc_channel.yaml", "configs/v2_0_rician_cmp.json",
              "configs/v2_0_rician_extra.json"]
NAIVE_CONFIG = "configs/v2_0_rician_naive.json"


def is_test(image_id: str, frac: float = 0.2) -> bool:
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def gather_tasks(config_paths: list[str], test_only: bool = True) -> tuple[list[dict], dict]:
    all_tasks: list[dict] = []
    seen: set[tuple[str, str]] = set()
    base_cfg = None
    for cp in config_paths:
        cfg = load_config(cp)
        if base_cfg is None:
            base_cfg = cfg
        vlm_cfg = cfg.get("vlm", {})
        qtypes = set(vlm_cfg.get("question_types", ["presence", "counting"]))
        tasks = read_tasks_csv(resolve_path(cfg["paths"]["tasks_csv"]))
        sel = select_vlm_tasks(tasks, question_types=qtypes,
                               max_tasks_per_image=int(vlm_cfg.get("max_tasks_per_image", 6)))
        for t in sel:
            key = (t["image_id"], t["question"])
            if key in seen:
                continue
            if test_only and not is_test(t["image_id"]):
                continue
            seen.add(key)
            all_tasks.append(t)
    return all_tasks, base_cfg


def _img_out(method: str, seed: int, image_id: str) -> Path:
    return Path(resolve_path(f"{OUT_DIR}/{method}/seed{seed}/{image_id}_-5dB.jpg"))


def do_transmit(seeds: int) -> None:
    import cv2

    m1_tasks, m1_cfg = gather_tasks(M1_CONFIGS)
    nv_tasks, nv_cfg = gather_tasks([NAIVE_CONFIG])
    m1_link = build_link_config(m1_cfg)
    nv_link_cfg, nv_calib = load_or_calibrate(nv_cfg)
    nv_quality = int((nv_cfg.get("vlm", {}).get("fading_link", {}) or {}).get("jpeg_quality", 90))
    jobs = [("m1", sorted({t["image_id"] for t in m1_tasks})),
            ("naive", sorted({t["image_id"] for t in nv_tasks}))]
    root = resolve_path(m1_cfg["paths"]["visdrone_val"])
    for method, ids in jobs:
        print(f"[{method}] {len(ids)} unique test images x {seeds} seeds", flush=True)
        for k in range(1, seeds + 1):
            n_done = 0
            for iid in ids:
                op = _img_out(method, k, iid)
                if op.exists():
                    continue
                src = image_path_for_task(root, iid)
                img = cv2.imread(str(src))
                if img is None:
                    raise RuntimeError(f"cv2.imread failed: {src}")
                rng = np.random.default_rng(_seed_from(Path(src).stem, "-5dB", "p1seed", k))
                if method == "m1":
                    dec, _meta = transmit_image_rate_adaptive(img, SNR_DB, m1_link, rng)
                else:
                    dec, _meta = transmit_image(img, SNR_DB, nv_calib, nv_link_cfg, rng,
                                                jpeg_quality=nv_quality)
                ensure_parent(op)
                cv2.imwrite(str(op), dec, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                n_done += 1
            print(f"[{method}] seed {k}: wrote {n_done} new", flush=True)
    marker = Path(resolve_path(f"{OUT_DIR}/.transmit_done"))
    ensure_parent(marker)
    marker.write_text("done\n")
    print("transmit phase done", flush=True)


def do_answer(seeds: int, resume: bool, out_csv: str) -> None:
    m1_tasks, m1_cfg = gather_tasks(M1_CONFIGS)
    nv_tasks, _ = gather_tasks([NAIVE_CONFIG])
    out_path = Path(resolve_path(out_csv))
    rows: list[dict] = []
    done: set[tuple[str, int, str, str]] = set()
    if resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        done = {(r["method"], int(r["seed"]), r["image_id"], r["question"]) for r in rows}
        print(f"resume: {len(done)} cells done", flush=True)
    evaluator = make_evaluator("qwen", m1_cfg)
    snr_bin = snr_bin_label(SNR_DB)
    channel_bin = channel_bin_from_snr(SNR_DB)
    n_new = 0
    for method, tasks in (("m1", m1_tasks), ("naive", nv_tasks)):
        for k in range(1, seeds + 1):
            for task in tasks:
                key = (method, k, task["image_id"], task["question"])
                if key in done:
                    continue
                img = _img_out(method, k, task["image_id"])
                # exact mirror of the campaign's s2 prompt
                prompt = build_vlm_prompt(task)
                prompt = f"service_level=2 snr_bin={snr_bin} channel={channel_bin} evidence_source=image\n{prompt}"
                pred = evaluate_prediction(evaluator, task, prompt, img, m1_cfg)
                rows.append({
                    "method": method, "seed": str(k), "image_id": task["image_id"],
                    "question_type": task["question_type"], "question": task["question"],
                    "ground_truth_answer": task["answer"], "predicted_answer": pred.predicted_answer,
                    "normalized_prediction": pred.normalized_prediction, "correct": str(pred.correct),
                    "model_name": pred.model_name,
                })
                n_new += 1
                if n_new % 100 == 0:
                    ensure_parent(out_path)
                    with out_path.open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS)
                        w.writeheader()
                        w.writerows(rows)
                    print(f"...{n_new} new answered ({method} seed {k})", flush=True)
    ensure_parent(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}", flush=True)
    summarize(rows)


def summarize(rows: list[dict]) -> None:
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        c = 1 if str(r["correct"]).lower() == "true" else 0
        per[(r["method"], int(r["seed"]))][0] += c
        per[(r["method"], int(r["seed"]))][1] += 1
    for (m, k), (c, n) in sorted(per.items()):
        acc[(m, "seed_accs")].append(c / max(1, n))
    report = {}
    for m in ("m1", "naive"):
        vals = acc.get((m, "seed_accs"), [])
        if vals:
            report[m] = {"n_seeds": len(vals), "mean": round(float(np.mean(vals)), 4),
                         "std": round(float(np.std(vals, ddof=1)), 4),
                         "min": round(min(vals), 4), "max": round(max(vals), 4),
                         "per_seed": [round(v, 4) for v in vals]}
    out = Path(resolve_path("outputs/reports/p1_seed_variance.json"))
    ensure_parent(out)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["transmit", "answer", "summarize"], required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="outputs/vlm/p1_seedvar_predictions.csv")
    args = ap.parse_args()
    if args.phase == "transmit":
        do_transmit(args.seeds)
    elif args.phase == "answer":
        do_answer(args.seeds, args.resume, args.out)
    else:
        with Path(resolve_path(args.out)).open(newline="", encoding="utf-8") as f:
            summarize(list(csv.DictReader(f)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
