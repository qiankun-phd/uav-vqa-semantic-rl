#!/usr/bin/env python3
"""Measure RTX 4060 GPU power/energy for paper-1 TGCN energy accounting (B1).

Motivation: on this card/driver (RTX 4060, driver 550.144.03) the NVML power
APIs are unsupported (`nvidia-smi --query-gpu=power.draw` -> [N/A];
nvmlDeviceGetPowerUsage / nvmlDeviceGetTotalEnergyConsumption -> Not
Supported). The driver nevertheless keeps an internal ~8 Hz power sampler
exposed via `nvidia-smi -q -d POWER` as a "Power Samples" block (Duration /
Max / Min / Avg over a ~14.4 s trailing window). We poll that block during
long steady-state phases and average the window averages, using only samples
taken at least `--settle` seconds (> one trailing window) after the phase's
steady-state loop started, which gives an unbiased steady-state mean power.

Phases:
  idle       : no GPU work (baseline board power)
  yolo       : YOLOv8n fine-tuned detector, imgsz=640, batch=1, campaign conf
  vlm        : Qwen2-VL-2B-Instruct VQA, campaign settings (min/max pixels,
               greedy, max_new_tokens=24, real degraded test images +
               real questions via build_vlm_prompt)
  idle_post  : sanity re-check of the baseline

Outputs (under --out-dir):
  gpu_power_phases.json  : per-phase mean power, per-item latency, J/item
  gpu_power_samples.csv  : raw polled samples with phase tags

Usage (full run, ~13 min):
  python scripts/measure_gpu_energy.py
Smoke:
  python scripts/measure_gpu_energy.py --smoke
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


# --------------------------------------------------------------------------
# Power sampling via `nvidia-smi -q -d POWER` "Power Samples" block
# --------------------------------------------------------------------------
class PowerSampler(threading.Thread):
    def __init__(self, poll_sec: float = 4.0):
        super().__init__(daemon=True)
        self.poll_sec = poll_sec
        self.records: list[dict] = []
        self._halt = threading.Event()

    @staticmethod
    def read_once() -> dict | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "-q", "-d", "POWER"],
                capture_output=True, text=True, timeout=10,
            ).stdout
        except Exception:
            return None
        if "Power Samples" not in out:
            return None
        block = out.split("Power Samples", 1)[1]

        def grab(key: str) -> float | None:
            m = re.search(rf"{key}\s*:\s*([\d.]+)\s*W", block)
            return float(m.group(1)) if m else None

        dur = re.search(r"Duration\s*:\s*([\d.]+)\s*sec", block)
        nsm = re.search(r"Number of Samples\s*:\s*(\d+)", block)
        rec = {
            "t": time.time(),
            "avg_w": grab("Avg"),
            "min_w": grab("Min"),
            "max_w": grab("Max"),
            "window_sec": float(dur.group(1)) if dur else None,
            "window_n": int(nsm.group(1)) if nsm else None,
        }
        # GPU utilization works even though power.draw is N/A.
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().split(",")
            rec["util_pct"] = float(q[0])
            rec["sm_mhz"] = float(q[1])
            rec["temp_c"] = float(q[2])
        except Exception:
            rec["util_pct"] = rec["sm_mhz"] = rec["temp_c"] = None
        return rec if rec["avg_w"] is not None else None

    def run(self) -> None:
        while not self._halt.is_set():
            rec = self.read_once()
            if rec is not None:
                self.records.append(rec)
            self._halt.wait(self.poll_sec)

    def stop(self) -> None:
        self._halt.set()


def phase_stats(records: list[dict], t0: float, t1: float, settle: float) -> dict:
    """Steady-state stats for samples inside [t0+settle, t1]."""
    sel = [r for r in records if t0 + settle <= r["t"] <= t1]
    if not sel:
        return {"n_windows": 0}
    avg = [r["avg_w"] for r in sel]
    mean = sum(avg) / len(avg)
    util = [r["util_pct"] for r in sel if r.get("util_pct") is not None]
    return {
        "n_windows": len(sel),
        "power_avg_w": mean,
        "power_win_min_w": min(r["min_w"] for r in sel),
        "power_win_max_w": max(r["max_w"] for r in sel),
        "power_std_between_windows_w": (
            sum((a - mean) ** 2 for a in avg) / len(avg)
        ) ** 0.5,
        "util_avg_pct": (sum(util) / len(util)) if util else None,
        "steady_used_sec": t1 - (t0 + settle),
    }


# --------------------------------------------------------------------------
# Workloads. Each returns meta including its own steady-state window
# (t_steady0/t_steady1) measured around the inference loop only, so model
# loading and warmup never contaminate the power statistics.
# --------------------------------------------------------------------------
def list_val_images() -> list[Path]:
    img_dir = REPO / "data/raw/visdrone/DET/val/images"
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        imgs = sorted((REPO / "data/raw/visdrone/DET/val").rglob("*.jpg"))
    if not imgs:
        raise FileNotFoundError("no VisDrone val images found")
    return imgs


def load_vlm_workload(cfg: dict, snr_tag: str = "5dB") -> list[tuple[Path, dict]]:
    """(degraded image, task) pairs matching the campaign at the given SNR."""
    tasks_csv = REPO / cfg["paths"]["tasks_csv"]
    deg_dir = REPO / cfg["paths"]["degraded_image_dir"] / snr_tag
    tasks = list(csv.DictReader(open(tasks_csv)))
    stem_to_img: dict[str, Path] = {}
    if deg_dir.exists():
        for p in sorted(deg_dir.glob("*.jpg")) + sorted(deg_dir.glob("*.png")):
            stem_to_img[p.stem.split("__")[0]] = p
    pairs = []
    for t in tasks:
        iid = t["image_id"]
        if iid in stem_to_img:
            pairs.append((stem_to_img[iid], t))
    if not pairs:  # fallback: raw val images, tasks matched by image_id
        raw = {p.stem: p for p in list_val_images()}
        pairs = [(raw[t["image_id"]], t) for t in tasks if t["image_id"] in raw]
    if not pairs:
        raise RuntimeError("could not build any (image, task) pairs")
    return pairs


def run_yolo_phase(duration: float, cfg: dict) -> dict:
    from ultralytics import YOLO

    weights = REPO / "outputs/detector/visdrone_yolov8n/weights/best.pt"
    model = YOLO(str(weights))
    imgs = list_val_images()
    for p in imgs[:5]:  # warmup
        model.predict(str(p), imgsz=640, conf=0.25, verbose=False, device=0)
    t_steady0 = time.time()
    lat: list[float] = []
    i = 0
    while time.time() - t_steady0 < duration:
        p = imgs[i % len(imgs)]
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=640, conf=0.25, verbose=False, device=0)
        lat.append(time.perf_counter() - t0)
        i += 1
    t_steady1 = time.time()
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()
    lat_sorted = sorted(lat)
    return {
        "t_steady0": t_steady0,
        "t_steady1": t_steady1,
        "weights": str(weights.relative_to(REPO)),
        "imgsz": 640,
        "n_items": len(lat),
        "sec_per_item_mean": sum(lat) / len(lat),
        "sec_per_item_median": lat_sorted[len(lat) // 2],
    }


def run_vlm_phase(duration: float, cfg: dict) -> dict:
    from vqa_semcom.evidence.builder import build_vlm_prompt
    from vqa_semcom.vlm.evaluator import QwenVLEvaluator
    from vqa_semcom.vlm.model_setup import resolve_model_reference

    vcfg = cfg["vlm"]
    model_ref = resolve_model_reference(vcfg)
    ev = QwenVLEvaluator(
        model_name=model_ref,
        fallback_model_name=vcfg.get("fallback_model_name"),
        device=vcfg.get("device", "cuda"),
        torch_dtype=vcfg.get("torch_dtype", "auto"),
        processor_use_fast=bool(vcfg.get("processor_use_fast", False)),
        min_pixels=vcfg.get("min_pixels"),
        max_pixels=vcfg.get("max_pixels"),
        max_new_tokens=int(vcfg.get("max_new_tokens", 24)),
    )
    pairs = load_vlm_workload(cfg)
    for img, task in pairs[:3]:  # warmup
        ev.predict(task, build_vlm_prompt(task), image_path=img)
    t_steady0 = time.time()
    lat: list[float] = []
    i = 0
    while time.time() - t_steady0 < duration:
        img, task = pairs[i % len(pairs)]
        t0 = time.perf_counter()
        ev.predict(task, build_vlm_prompt(task), image_path=img)
        lat.append(time.perf_counter() - t0)
        i += 1
    t_steady1 = time.time()
    import torch

    del ev
    gc.collect()
    torch.cuda.empty_cache()
    lat_sorted = sorted(lat)
    return {
        "t_steady0": t_steady0,
        "t_steady1": t_steady1,
        "model_ref": model_ref,
        "min_pixels": vcfg.get("min_pixels"),
        "max_pixels": vcfg.get("max_pixels"),
        "max_new_tokens": vcfg.get("max_new_tokens"),
        "n_items": len(lat),
        "sec_per_item_mean": sum(lat) / len(lat),
        "sec_per_item_median": lat_sorted[len(lat) // 2],
        "workload": "campaign degraded images @5dB Rician + real questions",
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_0_rician_cmp.json")
    ap.add_argument("--out-dir", default="outputs/energy")
    ap.add_argument("--poll-sec", type=float, default=4.0)
    ap.add_argument("--settle-sec", type=float, default=20.0)
    ap.add_argument("--idle-sec", type=float, default=120.0)
    ap.add_argument("--yolo-sec", type=float, default=150.0)
    ap.add_argument("--vlm-sec", type=float, default=240.0)
    ap.add_argument("--idle-post-sec", type=float, default=60.0)
    ap.add_argument("--smoke", action="store_true", help="short phases for a dry run")
    args = ap.parse_args()

    if args.smoke:
        args.idle_sec, args.yolo_sec, args.vlm_sec, args.idle_post_sec = 30, 40, 60, 25
        args.settle_sec = 16.0

    cfg = json.loads((REPO / args.config).read_text())
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = PowerSampler.read_once()
    if probe is None:
        raise SystemExit("cannot read Power Samples block from nvidia-smi")
    print(f"[probe] power avg {probe['avg_w']:.1f} W, util {probe['util_pct']}%")

    sampler = PowerSampler(poll_sec=args.poll_sec)
    sampler.start()
    windows: dict[str, tuple[float, float]] = {}  # steady windows per phase
    spans: dict[str, tuple[float, float]] = {}    # full phase spans (for CSV tags)
    meta: dict[str, dict] = {}

    def sleep_phase(name: str, duration: float) -> None:
        print(f"[phase] {name} for {duration:.0f}s ...", flush=True)
        t0 = time.time()
        time.sleep(duration)
        t1 = time.time()
        windows[name] = (t0, t1)
        spans[name] = (t0, t1)
        meta[name] = {}
        print(f"[phase] {name} done", flush=True)

    def work_phase(name: str, fn, duration: float) -> None:
        print(f"[phase] {name} (load + {duration:.0f}s steady) ...", flush=True)
        t0 = time.time()
        m = fn(duration, cfg)
        t1 = time.time()
        windows[name] = (m.pop("t_steady0"), m.pop("t_steady1"))
        spans[name] = (t0, t1)
        meta[name] = m
        print(f"[phase] {name} done ({m['n_items']} items, "
              f"{m['sec_per_item_mean']:.3f}s/item)", flush=True)

    sleep_phase("idle", args.idle_sec)
    work_phase("yolo", run_yolo_phase, args.yolo_sec)
    time.sleep(20)  # drain trailing window between phases
    work_phase("vlm", run_vlm_phase, args.vlm_sec)
    time.sleep(20)
    sleep_phase("idle_post", args.idle_post_sec)

    sampler.stop()
    sampler.join(timeout=10)

    with open(out_dir / "gpu_power_samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "avg_w", "min_w", "max_w", "window_sec",
                    "window_n", "util_pct", "sm_mhz", "temp_c"])
        for r in sampler.records:
            tag = ""
            for name, (t0, t1) in spans.items():
                if t0 <= r["t"] <= t1:
                    tag = name
                    break
            w.writerow([f"{r['t']:.1f}", tag, r["avg_w"], r["min_w"], r["max_w"],
                        r["window_sec"], r["window_n"], r["util_pct"],
                        r["sm_mhz"], r["temp_c"]])

    report: dict = {
        "hardware": "NVIDIA GeForce RTX 4060 (115 W limit), NUC9i7QNX host",
        "driver": "550.144.03; power via nvidia-smi -q -d POWER Power-Samples "
                  "(~8 Hz internal sampler, ~14.4 s trailing window); "
                  "NVML power.draw unsupported on this card",
        "method": f"poll every {args.poll_sec}s; per phase use samples >= "
                  f"{args.settle_sec}s after the steady loop starts; average "
                  "window-averages",
        "config": args.config,
        "phases": {},
    }
    for name, (t0, t1) in windows.items():
        st = phase_stats(sampler.records, t0, t1, args.settle_sec)
        st.update(meta.get(name, {}))
        report["phases"][name] = st

    p_idle = report["phases"]["idle"].get("power_avg_w")
    for name in ("yolo", "vlm"):
        ph = report["phases"].get(name, {})
        if ph.get("power_avg_w") and ph.get("sec_per_item_mean") and p_idle:
            ph["joule_per_item_total"] = ph["power_avg_w"] * ph["sec_per_item_mean"]
            ph["joule_per_item_incremental"] = (
                (ph["power_avg_w"] - p_idle) * ph["sec_per_item_mean"]
            )
    report["idle_baseline_w"] = p_idle

    out_json = out_dir / "gpu_power_phases.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
