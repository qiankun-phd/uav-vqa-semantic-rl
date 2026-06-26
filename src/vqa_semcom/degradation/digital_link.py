"""Digital semantic-transmission link: LDPC-coded BPSK over Rayleigh/Rician fading.

Models the physical data-transmission process used to expose semantic evidence
(image bytes or detector tokens) to channel impairments, matching the digital
chain in goal-oriented VQA semantic communication (LDPC + BPSK + fading):

    source bits -> LDPC encode -> BPSK -> fading channel (+AWGN) -> demod (LLR)
                -> LDPC decode -> recovered bits -> codec decode

Because a real LDPC belief-propagation decode per block is expensive, we use a
two-stage design:

1.  ``calibrate_link`` runs the *real* pyldpc encode/decode loop under block
    fading to measure post-decode bit-error-rate (BER) and frame-error-rate
    (FER) at each SNR bin.  With LDPC, delivered blocks are essentially
    error-free and failed blocks are lost, so FER is the dominant impairment
    (a coded waterfall -> block-erasure model).
2.  ``transmit_image`` / ``corrupt_detections`` apply the calibrated FER to the
    *actual* transmitted bytes/records (fast), then decode normally.

All randomness is seeded for reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FadingConfig:
    kind: str = "rician"          # "rayleigh" | "rician" | "awgn"
    k_factor_db: float = 6.0       # Rician LoS/scatter ratio (ignored if rayleigh/awgn)


@dataclass(frozen=True)
class LDPCConfig:
    n: int = 96                    # codeword length
    d_v: int = 3                   # variable-node degree
    d_c: int = 6                   # check-node degree (rate ~ 1 - d_v/d_c)
    maxiter: int = 50


@dataclass(frozen=True)
class LinkConfig:
    fading: FadingConfig = FadingConfig()
    ldpc: LDPCConfig = LDPCConfig()
    modulation: str = "bpsk"
    packet_payload_bits: int = 1024   # one transmission unit ("frame") mapped to FER
    calib_blocks: int = 1500
    image_grid: int = 8               # image sent as image_grid x image_grid independently-coded tiles


# ---------------------------------------------------------------------------
# Fading
# ---------------------------------------------------------------------------

def sample_power_gain(kind: str, k_factor_db: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return |h|^2 samples with E[|h|^2] = 1 (unit average power)."""
    kind = kind.lower()
    if kind == "awgn":
        return np.ones(size)
    if kind == "rayleigh":
        return rng.exponential(1.0, size)
    if kind == "rician":
        k = 10.0 ** (k_factor_db / 10.0)
        los = math.sqrt(k / (k + 1.0))
        sig = math.sqrt(1.0 / (2.0 * (k + 1.0)))
        hr = los + sig * rng.standard_normal(size)
        hi = sig * rng.standard_normal(size)
        return hr * hr + hi * hi
    raise ValueError(f"unknown fading kind: {kind}")


# ---------------------------------------------------------------------------
# Stage 1: real LDPC Monte-Carlo calibration
# ---------------------------------------------------------------------------

_LDPC_CACHE: dict[tuple[int, int, int], tuple[Any, Any, int]] = {}


def _build_ldpc(cfg: LDPCConfig):
    key = (cfg.n, cfg.d_v, cfg.d_c)
    if key not in _LDPC_CACHE:
        from pyldpc import make_ldpc
        H, G = make_ldpc(cfg.n, cfg.d_v, cfg.d_c, systematic=True, sparse=True)
        _LDPC_CACHE[key] = (H, G, G.shape[1])
    return _LDPC_CACHE[key]


def calibrate_link(snr_db_list, link_cfg: LinkConfig, seed: int = 0) -> dict[str, dict[str, float]]:
    """Measure post-decode BER and FER per SNR with the real LDPC codec.

    Block fading: one channel power gain |h|^2 per codeword, so the per-block
    effective SNR is ``snr_db + 10 log10|h|^2``.  A block is a "frame error" if
    any information bit is wrong after decoding.
    """
    from pyldpc import encode, decode, get_message

    H, G, k = _build_ldpc(link_cfg.ldpc)
    n = link_cfg.ldpc.n
    rate = k / n
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}

    for snr_db in snr_db_list:
        gains = sample_power_gain(link_cfg.fading.kind, link_cfg.fading.k_factor_db, link_cfg.calib_blocks, rng)
        bit_err = 0
        frame_err = 0
        total_bits = 0
        for g in gains:
            snr_eff = float(snr_db) + 10.0 * math.log10(max(g, 1e-12))
            msg = rng.integers(0, 2, k)
            y = encode(G, msg, snr=snr_eff, seed=int(rng.integers(0, 2**31 - 1)))
            dec = decode(H, y, snr=snr_eff, maxiter=link_cfg.ldpc.maxiter)
            rec = get_message(G, dec)
            errs = int(np.sum(rec != msg))
            bit_err += errs
            total_bits += k
            if errs > 0:
                frame_err += 1
        out[_snr_key(snr_db)] = {
            "snr_db": float(snr_db),
            "ber": bit_err / max(1, total_bits),
            "fer": frame_err / max(1, link_cfg.calib_blocks),
            "code_rate": rate,
            "k": k,
            "n": n,
        }
    return out


def _snr_key(snr_db) -> str:
    v = float(snr_db)
    return f"{int(v)}dB" if v.is_integer() else f"{v:g}dB"


# ---------------------------------------------------------------------------
# Stage 2: apply calibrated FER to real payloads
# ---------------------------------------------------------------------------

def _fer_for_snr(calib: dict[str, dict[str, float]], snr_db: float) -> float:
    key = _snr_key(snr_db)
    if key in calib:
        return float(calib[key]["fer"])
    # nearest SNR fallback
    best = min(calib.values(), key=lambda r: abs(r["snr_db"] - float(snr_db)))
    return float(best["fer"])


def transmit_image(image_bgr: np.ndarray, snr_db: float, calib, link_cfg: LinkConfig,
                   rng: np.random.Generator, jpeg_quality: int = 90,
                   conceal: str = "gray") -> tuple[np.ndarray, dict]:
    """Transmit the image as independently-coded tiles over the calibrated link.

    A raw JPEG bitstream has no resynchronisation, so a single lost packet
    destroys everything after it.  Real packetised image transmission avoids
    this with slicing / restart markers, so we model the image as an
    ``image_grid x image_grid`` array of tiles, each JPEG-encoded and carried in
    its own LDPC frame.  A tile whose frame is lost (probability = calibrated
    FER) is concealed; surviving tiles are JPEG-decoded normally and the image
    is reassembled.  This localises corruption and makes it scale with FER.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    g = max(1, int(link_cfg.image_grid))
    ys = np.linspace(0, h, g + 1).astype(int)
    xs = np.linspace(0, w, g + 1).astype(int)
    fer = _fer_for_snr(calib, snr_db)

    out = image_bgr.copy()
    n_tiles = 0
    lost = 0
    enc_bytes = 0
    for i in range(g):
        for j in range(g):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            tile = image_bgr[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            n_tiles += 1
            ok, buf = cv2.imencode(".jpg", tile, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                continue
            enc_bytes += len(buf)
            if rng.random() < fer:
                lost += 1
                if conceal == "neighbor" and j > 0:
                    out[y0:y1, x0:x1] = out[y0:y1, xs[j - 1]:x0][:, : (x1 - x0)]
                else:
                    out[y0:y1, x0:x1] = 128  # gray concealment
            else:
                dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # real per-tile codec round-trip
                if dec is not None:
                    out[y0:y1, x0:x1] = dec
    meta = {
        "snr_db": float(snr_db), "fer": fer, "tiles": n_tiles, "lost": lost,
        "loss_frac": lost / max(1, n_tiles), "decode_ok": lost < n_tiles,
        "jpeg_bytes": enc_bytes,
    }
    return out, meta


def corrupt_detections(records: list[dict], snr_db: float, calib, link_cfg: LinkConfig,
                       rng: np.random.Generator, num_classes: int = 11) -> tuple[list[dict], dict]:
    """Transmit detector tokens over the link; lost frames -> dropped/garbled.

    Each detection is serialised as one small frame (class id + box + score).  A
    frame lost at the calibrated FER is either dropped (header lost) or has its
    class label / coordinates garbled (payload lost).
    """
    fer = _fer_for_snr(calib, snr_db)
    out = []
    dropped = 0
    garbled = 0
    for rec in records:
        if rng.random() < fer:
            # split lost frames: half drop the detection, half garble fields
            if rng.random() < 0.5:
                dropped += 1
                continue
            garbled += 1
            r = dict(rec)
            if "class_id" in r:
                r["class_id"] = int(rng.integers(0, num_classes))
            if "label" in r:
                r["label"] = f"class_{int(rng.integers(0, num_classes))}"
            for c in ("x1", "y1", "x2", "y2", "cx", "cy", "w", "h"):
                if c in r:
                    try:
                        r[c] = float(r[c]) * float(rng.uniform(0.7, 1.3))
                    except (TypeError, ValueError):
                        pass
            out.append(r)
        else:
            out.append(dict(rec))
    meta = {"snr_db": float(snr_db), "fer": fer, "n_in": len(records),
            "n_out": len(out), "dropped": dropped, "garbled": garbled}
    return out, meta


def link_config_from_dict(d: dict[str, Any]) -> LinkConfig:
    fad = d.get("fading", {})
    ldpc = d.get("ldpc", {})
    return LinkConfig(
        fading=FadingConfig(kind=fad.get("kind", "rician"), k_factor_db=float(fad.get("k_factor_db", 6.0))),
        ldpc=LDPCConfig(n=int(ldpc.get("n", 96)), d_v=int(ldpc.get("d_v", 3)),
                        d_c=int(ldpc.get("d_c", 6)), maxiter=int(ldpc.get("maxiter", 50))),
        modulation=d.get("modulation", "bpsk"),
        packet_payload_bits=int(d.get("packet_payload_bits", 1024)),
        calib_blocks=int(d.get("calib_blocks", 1500)),
        image_grid=int(d.get("image_grid", 8)),
    )


def link_config_to_dict(cfg: LinkConfig) -> dict[str, Any]:
    return {
        "fading": asdict(cfg.fading), "ldpc": asdict(cfg.ldpc),
        "modulation": cfg.modulation, "packet_payload_bits": cfg.packet_payload_bits,
        "calib_blocks": cfg.calib_blocks, "image_grid": cfg.image_grid,
    }


# ---------------------------------------------------------------------------
# Pipeline integration: config-driven calibration cache + routing helpers
# ---------------------------------------------------------------------------

import hashlib
import json
from pathlib import Path

_CALIB_MEM: dict[str, dict] = {}
DEFAULT_SNR_BINS = [-5, 0, 5, 10, 15, 20]


def build_link_config(cfg: dict) -> LinkConfig:
    fl = (cfg.get("vlm", {}) or {}).get("fading_link", {}) or {}
    return link_config_from_dict(fl)


def _calib_key(link_cfg: LinkConfig, snrs) -> str:
    raw = json.dumps({"cfg": link_config_to_dict(link_cfg), "snrs": list(snrs)}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_or_calibrate(cfg: dict, snr_db_list=None, seed: int = 0):
    """Return (LinkConfig, calibration) computing the real LDPC FER/BER once.

    Cached in memory and on disk (keyed by the channel config + SNR bins) so the
    expensive Monte-Carlo runs only once per measurement campaign.
    """
    link_cfg = build_link_config(cfg)
    if snr_db_list is None:
        snr_db_list = (cfg.get("bins", {}) or {}).get("snr_db") or DEFAULT_SNR_BINS
    key = _calib_key(link_cfg, snr_db_list)
    if key in _CALIB_MEM:
        return link_cfg, _CALIB_MEM[key]
    from vqa_semcom.config import resolve_path
    cache_path = (cfg.get("paths", {}) or {}).get("link_calibration_json", "outputs/lut/link_calibration.json")
    p = Path(resolve_path(cache_path))
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if data.get("key") == key:
                _CALIB_MEM[key] = data["calib"]
                return link_cfg, data["calib"]
        except (ValueError, KeyError):
            pass
    calib = calibrate_link(snr_db_list, link_cfg, seed=seed)
    _CALIB_MEM[key] = calib
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"key": key, "link": link_config_to_dict(link_cfg), "calib": calib}, indent=2))
    return link_cfg, calib


def _seed_from(*parts) -> int:
    return int(hashlib.sha1("|".join(str(x) for x in parts).encode()).hexdigest()[:8], 16)


def fer_for(cfg: dict, channel_bin) -> float:
    from vqa_semcom.snr import snr_db_from_label
    _, calib = load_or_calibrate(cfg)
    return _fer_for_snr(calib, snr_db_from_label(channel_bin))


def transmit_image_to_path(image_path, out_dir, channel_bin, cfg: dict):
    """Channel-route an image file through the LDPC link and write the result."""
    import cv2
    from vqa_semcom.config import ensure_parent
    from vqa_semcom.snr import snr_db_from_label

    link_cfg, calib = load_or_calibrate(cfg)
    snr_db = snr_db_from_label(channel_bin)
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")
    rng = np.random.default_rng(_seed_from(Path(image_path).stem, channel_bin))
    quality = int((cfg.get("vlm", {}).get("fading_link", {}) or {}).get("jpeg_quality", 90))
    dec, _meta = transmit_image(img, snr_db, calib, link_cfg, rng, jpeg_quality=quality)
    out_path = Path(out_dir) / str(channel_bin) / f"{Path(image_path).stem}_{channel_bin}.jpg"
    ensure_parent(out_path)
    cv2.imwrite(str(out_path), dec, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return out_path
