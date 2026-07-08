#!/usr/bin/env bash
# E4: quality-backend A/B -- calibrated LUT cells vs per-sample predictor.
#
# Arms: lut / persample x seeds {0,1,2}, 500 train episodes on
# configs/v1_9_bubbles.yaml (proposed two-timescale constrained PPO, same
# flags as matrix_v1 RL_COMMON), each checkpoint evaluated under BOTH
# conditions (scenario=utm_conflict train-time eval + nominal re-eval).
# Success gate: persample arm average_accuracy >= lut arm with no
# constraint-metric degradation.
#
# NOTE: filename intentionally avoids the "chain_ab" substring (leftover
# cleanup loops on this host used to pkill that pattern).
set -u

ENVPY="$HOME/.conda/envs/uav_semcom/bin/python"
ROOT="$HOME/phd_research/vqa_main"
cd "$ROOT" || { echo "cannot cd $ROOT"; exit 2; }

SCENARIO="utm_conflict"
NOM_SCENARIO="nominal"
TRAIN_EP="${TRAIN_EP:-500}"
EVAL_EP="${EVAL_EP:-50}"
TPE="${TPE:-20}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-outputs/rl/e4_quality_backend_ab}"
CFG="configs/v1_9_bubbles.yaml"
mkdir -p "$OUT"
STATUS="$OUT/train_status.tsv"
[ -f "$STATUS" ] || printf "arm\tseed\texit\tstart\tend\tdir\n" > "$STATUS"

RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  --scenario "$SCENARIO" --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

run_one() {
  local arm="$1" seed="$2"; shift 2
  local dir="$OUT/${arm}_${seed}${DIR_SUFFIX:-}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[e4] SKIP  arm=$arm seed=$seed${DIR_SUFFIX:-} (already complete)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[e4] START arm=$arm seed=$seed${DIR_SUFFIX:-} $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$CFG" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "$ec" "$start" "$(date +%Y-%m-%dT%H:%M:%S)" "$dir" >> "$STATUS"
  echo "[e4] END   arm=$arm seed=$seed${DIR_SUFFIX:-} exit=$ec $(date)"
  return "$ec"
}

run_nom_eval() {
  local arm="$1" seed="$2"; shift 2
  local model="$OUT/${arm}_${seed}/ppo_two_timescale_policy.pt"
  if [ ! -f "$model" ]; then
    echo "[e4] MISS  arm=$arm seed=$seed nominal-eval (no checkpoint)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}_nom" "$seed" "no-ckpt" "-" "-" "-" >> "$STATUS"
    return 1
  fi
  DIR_SUFFIX="_nom" run_one "$arm" "$seed" \
    --policy ppo --load-ppo-model "$model" --two-timescale-ppo \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" "$@"
}

echo "[e4] ==== quality-backend A/B start $(date) | train_ep=$TRAIN_EP eval_ep=$EVAL_EP tpe=$TPE seeds=$SEEDS ===="
for seed in $SEEDS; do
  run_one lut "$seed" "${RL_COMMON[@]}" --quality-backend lut
  run_nom_eval lut "$seed" --quality-backend lut
  run_one persample "$seed" "${RL_COMMON[@]}" --quality-backend persample
  run_nom_eval persample "$seed" --quality-backend persample
done
echo "[e4] ==== quality-backend A/B done $(date) ===="
