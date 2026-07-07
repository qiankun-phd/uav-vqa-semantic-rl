#!/bin/bash
# DroneVehicle second-dataset evaluation chain (Qwen2-VL-2B, rician).
# Gated on chain_v26.sh (SmolVLM eval) releasing the GPU, then runs
# main -> cmp -> extra with --resume, merges predictions and builds the
# M1/M3/M4/M5 comparison + evidence-question complementarity (F6) reports.
set -u
cd /home/qiankun/phd_research/vqa_semcom || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
mkdir -p "$LOG"
SNR="-5,0,5,10,15,20"

echo "[dvchain] $(date) waiting for chain_v26 to release the GPU"
while pgrep -f "[c]hain_v26" >/dev/null 2>&1; do sleep 60; done
echo "[dvchain] $(date) chain_v26 finished -> letting GPU settle 60s"
sleep 60

for tag in main cmp extra; do
  CFG=configs/dv_rician_${tag}.json
  echo "[dvchain] $(date) START dv $tag ($CFG)"
  $PY scripts/run_v1_detector_eval.py --config "$CFG" \
      --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} --resume \
      > ${LOG}/dv_${tag}.log 2>&1
  echo "[dvchain] $(date) END dv $tag exit=$?"
done

echo "[dvchain] $(date) merging predictions"
$PY scripts/merge_dv_predictions.py > ${LOG}/dv_merge.log 2>&1
echo "[dvchain] $(date) merge exit=$?"

echo "[dvchain] $(date) building comparison_dv.csv (M1/M3/M4/M5)"
$PY scripts/build_comparison_v2.py --pred-dir outputs/vlm --prefix dv \
    --out outputs/reports/comparison_dv.csv > ${LOG}/dv_comparison.log 2>&1
echo "[dvchain] $(date) comparison exit=$?"

echo "[dvchain] $(date) building evidence-question complementarity (F6)"
$PY scripts/build_evidence_complementarity_dv.py > outputs/reports/complementarity_dv.txt 2>&1
echo "[dvchain] $(date) complementarity exit=$?"

echo "[dvchain] $(date) DV CHAIN ALL DONE"
