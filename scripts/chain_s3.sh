#!/bin/bash
# Q1 (s3 ROI, W1) + Q2 (presence t-scan, W2) chain -- run under tmux.
set -u
cd /home/qiankun/phd_research/vqa_semcom || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
mkdir -p ${LOG}
SNR="-5,0,5,10,15,20"
stamp(){ echo "[s3chain] $(date '+%F %T') $*"; }

for part in "main --max-tasks 800" cmp extra; do
  set -- $part; tag=$1; shift; extra_args="$*"
  stamp "START s3 $tag"
  $PY scripts/run_v1_detector_eval.py --config configs/s3_rician_${tag}.json \
      --evaluator qwen --service-levels 3 --snr-bins=${SNR} ${extra_args} --resume \
      > ${LOG}/s3_${tag}.log 2>&1
  stamp "END s3 $tag exit=$?"
done

stamp "W1 runs done -> comparison_s3 + presence breakdown"
$PY scripts/build_comparison_s3.py > ${LOG}/s3_analysis.log 2>&1
stamp "s3 analysis exit=$?"

stamp "START W2 presence token budget (t-scan)"
$PY scripts/run_presence_token_budget.py --resume > ${LOG}/w2_presence_tscan.log 2>&1
stamp "END W2 t-scan exit=$?"
$PY scripts/merge_token_budget_full.py > ${LOG}/w2_merge.log 2>&1
stamp "W2 merge exit=$?"
stamp "S3+W2 CHAIN ALL DONE"
