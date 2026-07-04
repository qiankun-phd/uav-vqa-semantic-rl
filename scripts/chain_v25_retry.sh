#!/bin/bash
set -u
cd /home/qiankun/phd_research/uav-vqa-semantic-rl || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
SNR="-5,0,5,10,15,20"
stamp(){ echo "[v25retry] $(date "+%F %T") $*"; }

stamp "download attempt: huggingface.co"
$PY -c "
from huggingface_hub import snapshot_download
print(\"DONE\", snapshot_download(\"Qwen/Qwen2.5-VL-3B-Instruct\"))" > ${LOG}/dl_qwen25vl3b.log 2>&1
if ! grep -q DONE ${LOG}/dl_qwen25vl3b.log; then
  stamp "direct failed -> hf-mirror.com"
  HF_ENDPOINT=https://hf-mirror.com $PY -c "
from huggingface_hub import snapshot_download
print(\"DONE\", snapshot_download(\"Qwen/Qwen2.5-VL-3B-Instruct\"))" >> ${LOG}/dl_qwen25vl3b.log 2>&1
fi
grep -q DONE ${LOG}/dl_qwen25vl3b.log || { stamp "DOWNLOAD FAILED on both endpoints"; exit 1; }
stamp "download OK"

for part in "main --max-tasks 800" cmp extra; do
  set -- $part; tag=$1; shift; extra_args="$*"
  stamp "START v25 $tag"
  $PY scripts/run_v1_detector_eval.py --config configs/v25_rician_${tag}.json \
      --evaluator qwen --service-levels 1,2 --snr-bins=${SNR} ${extra_args} --resume \
      > ${LOG}/v25_${tag}.log 2>&1
  stamp "END v25 $tag exit=$?"
done
stamp "V25 ALL DONE"
