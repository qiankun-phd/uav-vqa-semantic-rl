"""Compact SNR-conditioned DeepJSCC image codec (learned baseline M6, P1-2).

Design goals (paper-1 P1-2): a *small* (~1.7M param) convolutional
joint-source-channel autoencoder that is bandwidth-fair against the M2
analog baseline and the M1 digital path:

* Latent: /8 spatial downsampling with C=12 channels
  -> reals per image = H*W*12/64, complex channel uses = H*W*12/128
  -> bandwidth ratio = (H*W*12/128) / (H*W*3) = 1/32 complex uses per
     source real, which puts the largest VisDrone frames (2000x1500 ->
     ~282k uses) just inside the shared slot budget B*tau = 3e5 complex
     uses used by M2 (analog_link.py). Larger frames are pre-downscaled
     to fit, mirroring M2's source scaling.
* Channel: the same post-ZF effective-SNR abstraction as analog_link:
  one Rician block-fading power gain per image slot, receiver knows CSI,
  so the residual impairment is AWGN with variance 1/eff_snr_lin per
  unit-power real symbol. The codec is conditioned on eff_snr_db (FiLM),
  i.e. an SNR-adaptive DeepJSCC in the style of Xu et al. (ADJSCC).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

LATENT_CH = 12
DOWN = 8  # spatial downsampling factor
SLOT_COMPLEX_USES = 300_000.0  # B*tau, matches LinkConfig defaults / M2


def snr_norm(eff_snr_db: torch.Tensor) -> torch.Tensor:
    """Normalize effective SNR (dB) to roughly [-1, 1] for conditioning."""
    return (eff_snr_db - 7.5) / 15.0


class FiLM(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 2 * ch))
        self.ch = ch

    def forward(self, x: torch.Tensor, eff_snr_db: torch.Tensor) -> torch.Tensor:
        s = snr_norm(eff_snr_db).view(-1, 1)
        gb = self.net(s)
        gamma, beta = gb[:, : self.ch], gb[:, self.ch:]
        return x * (1.0 + gamma[..., None, None]) + beta[..., None, None]


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.a1 = nn.PReLU()
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.a2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a2(x + self.c2(self.a1(self.c1(x))))


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.a1 = nn.PReLU()
        self.c2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.a2 = nn.PReLU()
        self.c3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.a3 = nn.PReLU()
        self.film = FiLM(128)
        self.r1 = ResBlock(128)
        self.r2 = ResBlock(128)
        self.proj = nn.Conv2d(128, LATENT_CH, 3, padding=1)

    def forward(self, x: torch.Tensor, eff_snr_db: torch.Tensor) -> torch.Tensor:
        h = self.a1(self.c1(x))
        h = self.a2(self.c2(h))
        h = self.a3(self.c3(h))
        h = self.film(h, eff_snr_db)
        h = self.r2(self.r1(h))
        return self.proj(h)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(LATENT_CH, 128, 3, padding=1)
        self.film = FiLM(128)
        self.r1 = ResBlock(128)
        self.r2 = ResBlock(128)
        self.u1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.a1 = nn.PReLU()
        self.u2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.a2 = nn.PReLU()
        self.u3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.a3 = nn.PReLU()
        self.out = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, y: torch.Tensor, eff_snr_db: torch.Tensor) -> torch.Tensor:
        h = self.proj(y)
        h = self.film(h, eff_snr_db)
        h = self.r2(self.r1(h))
        h = self.a1(self.u1(h))
        h = self.a2(self.u2(h))
        h = self.a3(self.u3(h))
        return torch.sigmoid(self.out(h))


def power_normalize(z: torch.Tensor) -> torch.Tensor:
    """Per-sample unit average power over the real latent symbols."""
    n = z[0].numel()
    p = z.pow(2).flatten(1).mean(dim=1, keepdim=True).clamp_min(1e-12)
    return z / p.sqrt().view(-1, 1, 1, 1)


def awgn_for_eff_snr(z: torch.Tensor, eff_snr_db: torch.Tensor,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """y = x + n with n ~ N(0, 1/eff_snr_lin) per real dim (unit-power x)."""
    sigma = (10.0 ** (-eff_snr_db / 20.0)).view(-1, 1, 1, 1)
    noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
    return z + sigma * noise


class DJSCC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x: torch.Tensor, eff_snr_db: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        z = power_normalize(self.enc(x, eff_snr_db))
        y = awgn_for_eff_snr(z, eff_snr_db, generator=generator)
        return self.dec(y, eff_snr_db)


def complex_uses_for(h: int, w: int) -> float:
    ph = math.ceil(h / DOWN) * DOWN
    pw = math.ceil(w / DOWN) * DOWN
    return (ph // DOWN) * (pw // DOWN) * LATENT_CH / 2.0


@torch.inference_mode()
def transmit_image_djscc(model: DJSCC, image_bgr: np.ndarray, snr_db: float,
                         fading_kind: str, k_factor_db: float,
                         rng: np.random.Generator, device: str = "cuda",
                         budget_uses: float = SLOT_COMPLEX_USES) -> tuple[np.ndarray, dict]:
    """Full-image transmission with block fading (one |h|^2 per slot, CSI at RX).

    Mirrors analog_link.transmit_image_analog: if the frame's channel uses
    exceed the slot budget, the source is downscaled (sqrt scaling) first and
    upscaled back at the receiver; noise is drawn deterministically from rng.
    """
    import cv2
    from vqa_semcom.degradation.digital_link import sample_power_gain

    h, w = image_bgr.shape[:2]
    scale = 1.0
    uses = complex_uses_for(h, w)
    if uses > budget_uses:
        scale = math.sqrt(budget_uses / uses)
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        src = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        src = image_bgr
    sh0, sw0 = src.shape[:2]
    ph = math.ceil(sh0 / DOWN) * DOWN
    pw = math.ceil(sw0 / DOWN) * DOWN
    padded = cv2.copyMakeBorder(src, 0, ph - sh0, 0, pw - sw0, cv2.BORDER_REFLECT)

    g = float(sample_power_gain(fading_kind, k_factor_db, 1, rng)[0])
    eff_snr_db = float(snr_db) + 10.0 * math.log10(max(g, 1e-12))

    x = torch.from_numpy(padded[:, :, ::-1].copy()).float().permute(2, 0, 1)[None] / 255.0
    x = x.to(device)
    eff = torch.tensor([eff_snr_db], device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(0, 2**62)))
    xhat = model(x, eff, generator=gen)
    rec = (xhat[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)[:, :, ::-1]
    rec = rec[:sh0, :sw0]
    mse = float(np.mean((rec.astype(np.float64) - src.astype(np.float64)) ** 2))
    psnr = 10.0 * math.log10(255.0 ** 2 / max(mse, 1e-9))
    if scale < 1.0:
        rec = cv2.resize(rec, (w, h), interpolation=cv2.INTER_LINEAR)
    meta = {
        "method": "djscc",
        "snr_db": float(snr_db),
        "eff_snr_db": round(eff_snr_db, 3),
        "fading_gain": round(g, 5),
        "scale": round(scale, 4),
        "channel_uses": int(min(uses, budget_uses)),
        "psnr_db": round(psnr, 3),
    }
    return rec, meta
