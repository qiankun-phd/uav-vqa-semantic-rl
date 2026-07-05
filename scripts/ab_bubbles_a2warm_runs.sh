#!/usr/bin/env bash
# BUBBLES A2warm runs -- lambda warm-start probe (2026-07, after v3).
#
# Hypothesis under test ("the dual channel arrives late"): in v3 the conflict
# lambda needs hundreds of episodes to climb 0 -> ~4, but the policy is shaped
# by the BC warm start plus the high-entropy phase long before that, which is
# why B2 (lambda_conflict disabled) was behaviorally identical to A2 even with
# the dual channel as the sole conflict feedback path.  A2warm initializes
# lambda_conflict at its observed v3 equilibrium (4.0) from episode 0, so the
# constraint price is present during policy formation.  Expected: A2warm peak
# conflict rate drops well below the v3 A2 value 0.3027 (toward the 0.08
# limit) and separates from the lambda==0 behavior.
#
# Configuration is IDENTICAL to the v3 A2 arm (see ab_bubbles_v3_chain.sh)
# plus a single extra flag: --lambda-init-conflict 4.0.  Outputs land in the
# same v3 root as A2warm_<seed>[/_nom] so summarize_ab_bubbles_v3.py picks
# them up alongside the v3 arms (it evaluates the WARM H1/H2/guard criteria).
#
# NOTE: filename intentionally avoids the "chain_ab" substring (a leftover
# cleanup loop on the 160 host used to pkill that pattern).
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
LAMBDA_INIT="${LAMBDA_INIT:-4.0}"
OUT="outputs/rl/ab_bubbles_v3"
BUBBLES_CFG="configs/v1_9_bubbles.yaml"
mkdir -p "$OUT"
STATUS="$OUT/chain_status.tsv"
if [ ! -f "$STATUS" ]; then
  printf "arm\tseed\tconfig\texit\tstart\tend\tdir\n" > "$STATUS"
fi

# Same knobs as the v3 A2 arm; the only delta is --lambda-init-conflict.
RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  --lambda-init-conflict "$LAMBDA_INIT"
  --scenario "$SCENARIO" --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

run_one() {
  local arm="$1" cfg="$2" seed="$3"; shift 3
  local dir="$OUT/${arm}_${seed}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[warm] SKIP  arm=$arm seed=$seed (already complete)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$seed" "$cfg" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[warm] START arm=$arm seed=$seed cfg=$cfg $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$cfg" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  local end; end="$(date +%Y-%m-%dT%H:%M:%S)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$seed" "$cfg" "$ec" "$start" "$end" "$dir" >> "$STATUS"
  echo "[warm] END   arm=$arm seed=$seed exit=$ec $(date)"
  return "$ec"
}

run_nom_eval() {
  local arm="$1" cfg="$2" seed="$3"
  local train_dir="$OUT/${arm}_${seed}"
  local dir="$OUT/${arm}_${seed}_nom"
  local model="$train_dir/ppo_two_timescale_policy.pt"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[warm] SKIP  arm=$arm seed=$seed nominal-eval (already complete)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}nom" "$seed" "$cfg" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  if [ ! -f "$model" ]; then
    echo "[warm] MISS  arm=$arm seed=$seed nominal-eval (no checkpoint $model)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}nom" "$seed" "$cfg" "no-ckpt" "-" "-" "$dir" >> "$STATUS"
    return 1
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[warm] START arm=$arm seed=$seed nominal-eval $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$cfg" \
    --policy ppo --two-timescale-ppo --load-ppo-model "$model" \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  local end; end="$(date +%Y-%m-%dT%H:%M:%S)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}nom" "$seed" "$cfg" "$ec" "$start" "$end" "$dir" >> "$STATUS"
  echo "[warm] END   arm=$arm seed=$seed nominal-eval exit=$ec $(date)"
  return "$ec"
}

echo "[warm] ==== A2warm lambda warm-start runs start $(date) | train_ep=$TRAIN_EP eval_ep=$EVAL_EP tpe=$TPE seeds=$SEEDS lambda_init=$LAMBDA_INIT ===="

for seed in $SEEDS; do
  run_one A2warm "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}"
  run_nom_eval A2warm "$BUBBLES_CFG" "$seed"
done

echo "[warm] ==== A2warm runs done $(date) ===="
"$ENVPY" scripts/summarize_ab_bubbles_v3.py --root "$OUT" || echo "[warm] summary script failed"
echo "[warm] status table:"
tail -20 "$STATUS"
