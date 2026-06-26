import json
from pathlib import Path
cfg = json.loads(Path("configs/v1_9_snr_lut.yaml").read_text())
def redir(o):
    if isinstance(o, dict): return {k: redir(v) for k,v in o.items()}
    if isinstance(o, list): return [redir(v) for v in o]
    if isinstance(o, str): return o.replace("v1_9","v2_0")
    return o
cfg = redir(cfg)
cfg.setdefault("vlm", {})
cfg["vlm"]["channel_model"] = "ldpc_fading"   # routing flag (enables digital_link path)
cfg["vlm"]["fading_link"] = {
    "channel_mode": "rate_adaptive",
    "fading": {"kind": "rician", "k_factor_db": 6.0},
    "bandwidth_hz": 1000000.0,
    "tx_time_budget_s": 0.3,
    "min_payload_bytes": 1500,
    "ldpc": {"n": 96, "d_v": 3, "d_c": 6, "maxiter": 50},
    "image_grid": 8, "packet_payload_bits": 1024, "calib_blocks": 3000, "jpeg_quality": 90,
}
cfg.setdefault("paths", {})["link_calibration_json"] = "outputs/lut/link_calibration_v2_0.json"
Path("configs/v2_0_ldpc_channel.yaml").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
print("v2_0 config: channel_mode=", cfg["vlm"]["fading_link"]["channel_mode"], "tau=", cfg["vlm"]["fading_link"]["tx_time_budget_s"])
