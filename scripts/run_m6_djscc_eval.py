#!/usr/bin/env python3
"""M6 learned baseline eval: compact SNR-conditioned DeepJSCC + same Qwen backend.

Mirrors run_m2_analog_eval.py (same task selection as the v3_0 campaign's three
config runs, same prompt shape, same scoring), but the image is transmitted by
the trained DJSCC codec (Rician block fading, CSI at RX) instead of the analog
mapper. Two phases to respect the 8GB GPU: (1) transmit every unique
(image, SNR) with only the codec resident; (2) load Qwen and answer.

Usage:
  python scripts/run_m6_djscc_eval.py \
      --configs configs/v2_0_ldpc_channel.yaml,configs/v2_0_rician_cmp.json,configs/v2_0_rician_extra.json \
      --ckpt outputs/djscc/djscc_best.pt --test-only --resume \
      --out outputs/vlm/m6_djscc_rician_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from vqa_semcom.config import load_config, resolve_path, ensure_parent
from vqa_semcom.evidence.builder import build_vlm_prompt, image_path_for_task, read_tasks_csv, select_vlm_tasks
from vqa_semcom.snr import parse_snr_bins, snr_bin_label, channel_bin_from_snr
from vqa_semcom.vlm.evaluator import evaluate_prediction, make_evaluator

FIELDS = ["image_id", "question_type", "question", "ground_truth_answer", "snr_db", "snr_bin",
          "channel_bin", "method", "predicted_answer", "normalized_prediction", "correct",
          "payload_bytes", "channel_uses", "eff_snr_db", "fading_gain", "scale", "psnr_db",
          "model_name"]


def _seed(image_id: str, snr: float) -> int:
    return zlib.crc32(f"{image_id}|{snr}".encode()) & 0xFFFFFFFF


def is_test(image_id: str, frac: float = 0.2) -> bool:
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def gather_tasks(config_paths: list[str], test_only: bool) -> tuple[list[dict], dict]:
    """Replicate each campaign config's own task selection, then concat."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=("configs/v2_0_ldpc_channel.yaml,"
                                          "configs/v2_0_rician_cmp.json,"
                                          "configs/v2_0_rician_extra.json"))
    ap.add_argument("--ckpt", default="outputs/djscc/djscc_best.pt")
    ap.add_argument("--evaluator", choices=["mock", "qwen"], default="qwen")
    ap.add_argument("--snr-bins", default="-5,0,5,10,15,20")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--out", default="outputs/vlm/m6_djscc_rician_predictions.csv")
    ap.add_argument("--img-dir", default="outputs/vlm/djscc_images_rician")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import cv2
    import torch

    from vqa_semcom.degradation.djscc import DJSCC, transmit_image_djscc

    cfg_paths = [c.strip() for c in args.configs.split(",") if c.strip()]
    tasks, cfg = gather_tasks(cfg_paths, args.test_only)
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    snr_values = parse_snr_bins(args.snr_bins)
    fading = cfg["vlm"]["fading_link"]["fading"]
    kind, k_db = fading["kind"], float(fading.get("k_factor_db", 6.0))
    visdrone_root = resolve_path(cfg["paths"]["visdrone_val"])
    out_path = Path(resolve_path(args.out))
    img_dir = Path(resolve_path(args.img_dir))
    qt_counts: dict[str, int] = {}
    for t in tasks:
        qt_counts[t["question_type"]] = qt_counts.get(t["question_type"], 0) + 1
    print(f"tasks={len(tasks)} {qt_counts} snrs={snr_values} fading={kind}(K={k_db}dB)", flush=True)

    # ---------------- phase 1: transmit unique (image, snr) ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_ids = sorted({t["image_id"] for t in tasks})
    meta_by_key: dict[tuple[str, float], dict] = {}
    todo = [(iid, float(s)) for iid in image_ids for s in snr_values]
    pending = []
    for iid, s in todo:
        op = img_dir / f"{iid}_snr{s:g}.jpg"
        mp = img_dir / f"{iid}_snr{s:g}.meta.json"
        if args.resume and op.exists() and mp.exists():
            import json
            meta_by_key[(iid, s)] = json.loads(mp.read_text())
        else:
            pending.append((iid, s, op, mp))
    print(f"transmissions: {len(todo)} total, {len(pending)} to do", flush=True)
    if pending:
        model = DJSCC().to(device)
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        print(f"loaded {args.ckpt} (step {state.get('step')}, probe {state.get('probe')})", flush=True)
        for n, (iid, s, op, mp) in enumerate(pending):
            src = image_path_for_task(visdrone_root, iid)
            img = cv2.imread(str(src))
            if img is None:
                raise RuntimeError(f"cv2.imread failed: {src}")
            rec, meta = transmit_image_djscc(model, img, s, kind, k_db,
                                             rng=np.random.default_rng(_seed(iid, s)),
                                             device=device)
            ensure_parent(op)
            cv2.imwrite(str(op), rec, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            import json
            mp.write_text(json.dumps(meta))
            meta_by_key[(iid, s)] = meta
            if (n + 1) % 50 == 0:
                print(f"...transmitted {n+1}/{len(pending)}", flush=True)
        del model
        torch.cuda.empty_cache()
    psnrs: dict[float, list[float]] = {}
    for (iid, s), m in meta_by_key.items():
        psnrs.setdefault(s, []).append(float(m.get("psnr_db", float("nan"))))
    for s in sorted(psnrs):
        print(f"PSNR@{s:g}dB: mean={np.nanmean(psnrs[s]):.2f} dB (n={len(psnrs[s])})", flush=True)

    # ---------------- phase 2: VLM answering ----------------
    rows: list[dict] = []
    done: set[tuple[str, str, str]] = set()
    if args.resume and out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        done = {(r["image_id"], r["question"], r["snr_db"]) for r in rows}
        print(f"resume: {len(done)} rows already done", flush=True)

    evaluator = make_evaluator(args.evaluator, cfg)
    n_new = 0
    for ti, task in enumerate(tasks):
        iid = task["image_id"]
        for s in snr_values:
            s = float(s)
            key = (iid, task["question"], f"{s:g}")
            if key in done:
                continue
            meta = meta_by_key[(iid, s)]
            op = img_dir / f"{iid}_snr{s:g}.jpg"
            prompt = build_vlm_prompt(task)
            prompt = f"service_level=2-djscc snr={s:g} channel={kind} evidence_source=djscc_image\n{prompt}"
            pred = evaluate_prediction(evaluator, task, prompt, op, cfg)
            rows.append({
                "image_id": iid, "question_type": task["question_type"], "question": task["question"],
                "ground_truth_answer": task["answer"], "snr_db": f"{s:g}", "snr_bin": snr_bin_label(s),
                "channel_bin": channel_bin_from_snr(s), "method": "M6_djscc",
                "predicted_answer": pred.predicted_answer, "normalized_prediction": pred.normalized_prediction,
                "correct": str(pred.correct),
                "payload_bytes": meta.get("channel_uses", 0), "channel_uses": meta.get("channel_uses", 0),
                "eff_snr_db": meta.get("eff_snr_db", ""), "fading_gain": meta.get("fading_gain", ""),
                "scale": meta.get("scale", ""), "psnr_db": meta.get("psnr_db", ""),
                "model_name": pred.model_name,
            })
            n_new += 1
            if n_new % 100 == 0:
                ensure_parent(out_path)
                with out_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerows(rows)
                print(f"...answered {n_new} new (task {ti+1}/{len(tasks)})", flush=True)
    ensure_parent(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"method=M6_djscc rows={len(rows)} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
