#!/usr/bin/env python3
"""P1-4 (cheap path): run-to-run reproducibility of the s1 token pipeline.

The three Rician campaigns (v3_0 with Qwen2-VL-2B, v25 with Qwen2.5-VL-3B,
v26 with SmolVLM) re-ran the *entire* s1 chain independently -- YOLO detector
inference, token channel corruption, calibration, symbolic decode. Since only
the s2 receiver differs between campaigns, agreement of the s1 verdicts across
runs measures the end-to-end replay determinism of the token path (detector
nondeterminism included).

Output: outputs/reports/p1_s1_replay_check.json
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def is_test(iid: str) -> bool:
    return (zlib.crc32(str(iid).encode()) % 100) < 20


def load_s1(prefix_files: list[str], run: str) -> pd.DataFrame:
    frames = []
    for f in prefix_files:
        p = REPO / "outputs/vlm" / f
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype={"image_id": str},
                         usecols=["image_id", "question", "question_type", "snr_bin",
                                  "service_level", "correct", "normalized_prediction"])
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big = big[(big.service_level == 1) & big.image_id.map(is_test)]
    big = big.drop_duplicates(["image_id", "question", "snr_bin"])
    big["ok"] = big["correct"].map(lambda v: 1 if str(v).strip().lower() == "true" else 0)
    return big.rename(columns={"ok": f"ok_{run}", "normalized_prediction": f"pred_{run}"})[
        ["image_id", "question", "question_type", "snr_bin", f"ok_{run}", f"pred_{run}"]]


def main() -> int:
    runs = {
        "v3_0": ["v3_0_rician_predictions.csv"],
        "v25": ["v25_rician_main_predictions.csv", "v25_rician_cmp_predictions.csv",
                 "v25_rician_extra_predictions.csv"],
        "v26": ["v26_rician_main_predictions.csv", "v26_rician_cmp_predictions.csv",
                 "v26_rician_extra_predictions.csv"],
    }
    dfs = {r: load_s1(files, r) for r, files in runs.items()}
    base = dfs["v3_0"]
    report = {"note": "s1 symbolic-decoder verdicts on the Rician test split, per unique (image,question,snr)"}
    report["v3_0"] = {"n": len(base), "acc": round(base.ok_v3_0.mean(), 4)}
    for other in ("v25", "v26"):
        m = base.merge(dfs[other], on=["image_id", "question", "question_type", "snr_bin"],
                       how="inner", validate="1:1")
        same_verdict = (m["ok_v3_0"] == m[f"ok_{other}"]).mean()
        same_pred = (m["pred_v3_0"].astype(str) == m[f"pred_{other}"].astype(str)).mean()
        report[other] = {
            "n_common": len(m),
            "acc": round(m[f"ok_{other}"].mean(), 4),
            "frac_same_verdict": round(float(same_verdict), 6),
            "frac_same_normalized_prediction": round(float(same_pred), 6),
            "n_verdict_diff": int((m["ok_v3_0"] != m[f"ok_{other}"]).sum()),
        }
    out = REPO / "outputs/reports/p1_s1_replay_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
