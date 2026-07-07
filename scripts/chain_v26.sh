#!/bin/bash
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
SNR="-5,0,5,10,15,20"
stamp(){ echo "[v26] $(date '+%F %T') $*"; }
for part in "main --max-tasks 800" cmp extra; do
  set -- $part; tag=$1; shift; extra_args="$*"
  stamp "START v26 $tag"
  $PY scripts/run_v1_detector_eval.py --config configs/v26_rician_${tag}.json \
      --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} ${extra_args} --resume \
      > ${LOG}/v26_${tag}.log 2>&1
  stamp "END v26 $tag exit=$?"
done
stamp "V26 ALL DONE"
