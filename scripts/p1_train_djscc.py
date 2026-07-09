#!/usr/bin/env python3
"""P1-2: train the compact SNR-conditioned DeepJSCC codec (M6) on the
VisDrone-DET *train* split (6471 images; zero overlap with the 548 val
images used by every VQA evaluation).

Stage A (once): build a decode-fast cache (max side 1600, JPEG q92).
Stage B: random 128x128 crops (with random 1.0-2.0x downscale + hflip),
Rician(K=6dB) block fading with snr_db ~ U(-7,22), post-ZF AWGN channel,
MSE loss. Checkpoints are resume-safe; a fixed PSNR probe (last 16 train
images, never used as crops) is logged every eval interval.

Usage:
  python scripts/p1_train_djscc.py --steps 30000 [--resume]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from vqa_semcom.degradation.digital_link import sample_power_gain
from vqa_semcom.degradation.djscc import DJSCC

TRAIN_IMAGES = REPO_ROOT / "data/raw/visdrone/DET/train/images"
CACHE_DIR = REPO_ROOT / "outputs/djscc/train_cache"
CKPT_DIR = REPO_ROOT / "outputs/djscc"
N_PROBE = 16


def build_cache(max_side: int = 1600, quality: int = 92) -> list[Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    srcs = sorted(TRAIN_IMAGES.glob("*.jpg"))
    if not srcs:
        raise RuntimeError(f"no train images under {TRAIN_IMAGES}")
    out_paths = []
    t0 = time.time()
    for i, p in enumerate(srcs):
        q = CACHE_DIR / p.name
        out_paths.append(q)
        if q.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            out_paths.pop()
            continue
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(q), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if (i + 1) % 500 == 0:
            print(f"cache {i+1}/{len(srcs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"cache ready: {len(out_paths)} images", flush=True)
    return out_paths


class CropDataset(Dataset):
    def __init__(self, paths: list[Path], crop: int = 128, length: int = 10**9) -> None:
        self.paths = paths
        self.crop = crop
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rnd = random.Random()
        for _ in range(4):
            p = self.paths[rnd.randrange(len(self.paths))]
            img = cv2.imread(str(p))
            if img is not None and min(img.shape[:2]) >= self.crop:
                break
        else:
            raise RuntimeError("could not read a train image")
        s = rnd.uniform(1.0, 2.0)
        if s > 1.001:
            h, w = img.shape[:2]
            nh, nw = max(self.crop, int(h / s)), max(self.crop, int(w / s))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        y = rnd.randrange(0, h - self.crop + 1)
        x = rnd.randrange(0, w - self.crop + 1)
        tile = img[y:y + self.crop, x:x + self.crop]
        if rnd.random() < 0.5:
            tile = tile[:, ::-1]
        rgb = tile[:, :, ::-1].astype(np.float32) / 255.0
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1)


def sample_eff_snr(batch: int, rng: np.random.Generator,
                   kind: str = "rician", k_db: float = 6.0) -> torch.Tensor:
    snr = rng.uniform(-7.0, 22.0, batch)
    g = sample_power_gain(kind, k_db, batch, rng)
    return torch.from_numpy(snr + 10.0 * np.log10(np.maximum(g, 1e-12))).float()


@torch.inference_mode()
def probe_psnr(model: DJSCC, probe_paths: list[Path], device: str) -> dict[str, float]:
    """Deterministic PSNR probe: center 512 crop, fixed fading draws."""
    out: dict[str, float] = {}
    for snr_db in (-5.0, 5.0, 20.0):
        rng = np.random.default_rng(int(1000 + snr_db))
        gen = torch.Generator(device=device)
        gen.manual_seed(int(2000 + snr_db))
        vals = []
        for p in probe_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            c = 512
            if min(h, w) < c:
                continue
            y0, x0 = (h - c) // 2, (w - c) // 2
            tile = img[y0:y0 + c, x0:x0 + c]
            x = torch.from_numpy(tile[:, :, ::-1].copy()).float().permute(2, 0, 1)[None].to(device) / 255.0
            g = float(sample_power_gain("rician", 6.0, 1, rng)[0])
            eff = torch.tensor([snr_db + 10.0 * math.log10(max(g, 1e-12))], device=device)
            xh = model(x, eff, generator=gen)
            mse = float(torch.mean((xh - x) ** 2))
            vals.append(10.0 * math.log10(1.0 / max(mse, 1e-12)))
        out[f"{snr_db:g}dB"] = round(float(np.mean(vals)), 3) if vals else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    paths = build_cache()
    probe_paths = paths[-N_PROBE:]
    train_paths = paths[:-N_PROBE]

    model = DJSCC().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M train_imgs={len(train_paths)}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.min_lr)

    step0 = 0
    ckpt_last = CKPT_DIR / "djscc_last.pt"
    ckpt_best = CKPT_DIR / "djscc_best.pt"
    best_psnr = -1.0
    if args.resume and ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step0 = int(state["step"])
        best_psnr = float(state.get("best_psnr", -1.0))
        print(f"resumed at step {step0} (best probe {best_psnr:.2f} dB)", flush=True)
    if step0 >= args.steps:
        print("already trained to target steps", flush=True)
        return 0

    ds = CropDataset(train_paths)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)
    it = iter(dl)
    rng = np.random.default_rng(step0 + 1)
    t0 = time.time()
    run_loss = 0.0
    for step in range(step0 + 1, args.steps + 1):
        x = next(it).to(device, non_blocking=True)
        eff = sample_eff_snr(x.shape[0], rng).to(device)
        xhat = model(x, eff)
        loss = torch.mean((xhat - x) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += float(loss)
        if step % 200 == 0:
            rate = 200 / max(1e-9, time.time() - t0)
            print(f"step {step}/{args.steps} loss={run_loss/200:.5f} lr={sched.get_last_lr()[0]:.2e} ({rate:.1f} it/s)", flush=True)
            run_loss, t0 = 0.0, time.time()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            pr = probe_psnr(model, probe_paths, device)
            model.train()
            mean20 = pr.get("20dB", float("nan"))
            print(f"probe@step{step}: {pr}", flush=True)
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            state = {"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "step": step,
                     "probe": pr, "best_psnr": best_psnr, "params": n_params}
            torch.save(state, ckpt_last)
            if not math.isnan(mean20) and mean20 > best_psnr:
                best_psnr = mean20
                state["best_psnr"] = best_psnr
                torch.save(state, ckpt_best)
                print(f"new best probe@20dB: {best_psnr:.2f} dB", flush=True)
            t0 = time.time()
    print(f"training done. best probe@20dB = {best_psnr:.2f} dB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
