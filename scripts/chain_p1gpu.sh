#!/bin/bash
# P1 GPU chain (paper 1): DJSCC training -> M6 eval -> seed-variance answering.
# Strictly serial on the 4060 (8 GB); every stage is resume-safe.
set -u
cd /home/qiankun/phd_research/vqa_semcom || exit 1
PY=/home/qiankun/.conda/envs/uav_semcom/bin/python
LOG=outputs/logs
mkdir -p ${LOG}
stamp(){ echo "[p1gpu] $(date '+%F %T') $*"; }

stamp "STAGE2 djscc train START"
$PY scripts/p1_train_djscc.py --steps 30000 --resume > ${LOG}/p1_djscc_train.log 2>&1
stamp "STAGE2 exit=$?"

CKPT=outputs/djscc/djscc_best.pt
[ -f "$CKPT" ] || CKPT=outputs/djscc/djscc_last.pt
stamp "STAGE3 m6 eval START ckpt=$CKPT"
$PY scripts/run_m6_djscc_eval.py --ckpt "$CKPT" --test-only --resume > ${LOG}/p1_m6_eval.log 2>&1
stamp "STAGE3 exit=$?"

# stage 4 needs the CPU transmit phase to have finished
until [ -f outputs/vlm/p1_seedvar/.transmit_done ]; do
  stamp "STAGE4 waiting for seedvar transmit marker"; sleep 60
done
stamp "STAGE4 seedvar answer START"
$PY scripts/p1_seed_variance.py --phase answer --seeds 10 --resume > ${LOG}/p1_seedvar_answer.log 2>&1
stamp "STAGE4 exit=$?"
stamp "ALL DONE"
