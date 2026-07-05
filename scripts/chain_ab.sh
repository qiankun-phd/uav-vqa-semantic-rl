#!/usr/bin/env bash
# A/B training-level validation chain for the BUBBLES-conformant scenario profile.
#
# Arm A = default profile (multi_uav_env.scenario_profile = null) via configs/v1_9_snr_lut.yaml
# Arm B = BUBBLES profile (scenario_profile = "bubbles")        via configs/v1_9_bubbles.yaml
#
# Both arms run the recommended paper controller (proposed two-timescale semantic RL:
# risk-aware CMDP duals + semantic_utility reward + Lyapunov queues + BC warm-start +
# state_v2_fixed + 128x128 + deadline guard + payload-delay projection + T2 token-fast
# projection) under the utm_conflict traffic scenario so the CPA tactical-conflict
# constraint is actually exercised (background operational intents are enabled there).
#
# Serial single-GPU chain. Order puts the headline B-vs-A comparison (seed 0) first.
set -u

ENVPY="$HOME/.conda/envs/uav_semcom/bin/python"
ROOT="$HOME/phd_research/vqa_main"
cd "$ROOT" || { echo "cannot cd $ROOT"; exit 2; }

SCENARIO="utm_conflict"
TRAIN_EP="${TRAIN_EP:-500}"
EVAL_EP="${EVAL_EP:-50}"
TPE="${TPE:-20}"
OUT="outputs/rl/ab_bubbles_utmconf"
mkdir -p "$OUT"
STATUS="$OUT/chain_status.tsv"
printf "arm\tprofile\tseed\tconfig\texit\tstart\tend\tdir\n" > "$STATUS"

COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --scenario "$SCENARIO" --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

run_one() {
  local arm="$1" cfg="$2" seed="$3"
  local profile; [ "$arm" = "B" ] && profile="bubbles" || profile="default"
  local dir="$OUT/${arm}_seed${seed}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[chain] SKIP  arm=$arm seed=$seed (already complete: $dir/run_config.json)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$profile" "$seed" "$cfg" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[chain] START arm=$arm profile=$profile seed=$seed cfg=$cfg $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$cfg" "${COMMON[@]}" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  local end; end="$(date +%Y-%m-%dT%H:%M:%S)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$profile" "$seed" "$cfg" "$ec" "$start" "$end" "$dir" >> "$STATUS"
  echo "[chain] END   arm=$arm seed=$seed exit=$ec $(date)"
  return "$ec"
}

echo "[chain] ==== BUBBLES A/B chain start $(date) | train_ep=$TRAIN_EP eval_ep=$EVAL_EP tpe=$TPE scenario=$SCENARIO ===="
run_one B configs/v1_9_bubbles.yaml 0
run_one A configs/v1_9_snr_lut.yaml 0
run_one B configs/v1_9_bubbles.yaml 1
run_one A configs/v1_9_snr_lut.yaml 1
echo "[chain] ==== BUBBLES A/B chain done $(date) ===="
echo "[chain] status table:"
cat "$STATUS"
