#!/usr/bin/env bash
# O2 free-phrasing probe chain (paper1 TGCN generality check).
# Run inside tmux session `o2probe` on 160.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/.conda/envs/uav_semcom/bin/python

echo "=== [1/5] paraphrase generation ==="
$PY scripts/o2_make_paraphrases.py

echo "=== [2/5] symbolic decoder (strict + fallback parsers) ==="
$PY scripts/o2_symbolic_eval.py

echo "=== [3/5] VLM reads tokens (GPU) ==="
$PY scripts/o2_vlm_runner.py --mode tokens \
    --out outputs/vlm/o2_vlm_tokens_predictions.csv --resume

echo "=== [4/5] VLM reads degraded image (GPU) ==="
$PY scripts/o2_vlm_runner.py --mode image \
    --out outputs/vlm/o2_vlm_image_predictions.csv --resume

echo "=== [5/5] analysis ==="
$PY scripts/o2_analyze.py

echo "=== O2 CHAIN DONE ==="
