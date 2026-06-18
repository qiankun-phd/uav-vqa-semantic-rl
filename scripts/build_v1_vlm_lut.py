#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.vlm.lut import build_lut_from_predictions, read_predictions, write_lut_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    predictions = read_predictions(resolve_path(cfg["paths"]["vlm_predictions_csv"]))
    if not predictions:
        raise RuntimeError("No V1 predictions found. Run scripts/run_v1_vlm_eval.py first.")
    rows = build_lut_from_predictions(predictions)
    out_path = resolve_path(cfg["paths"]["vlm_lut_csv"])
    write_lut_csv(rows, out_path)
    print(f"prediction_rows={len(predictions)} lut_rows={len(rows)}")
    print(f"vlm_lut_csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

