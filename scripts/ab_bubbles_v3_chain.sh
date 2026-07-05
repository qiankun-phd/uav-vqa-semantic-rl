#!/usr/bin/env bash
# BUBBLES A/B v3 chain -- second calibration round (2026-07).
#
# Why v3: the v2 chain showed B2 (lambda_conflict disabled) bit-identical to A2,
# i.e. the conflict dual channel was decorative -- the shaped conflict penalty
# (conflict_cost_weight=0.5) plus the risk/utm Lyapunov terms did all the work.
# v3 calibration zeroes every shaped/queue conflict path so the Lagrangian term
# -lambda_conflict*violation is the SOLE conflict feedback; B2 must now degrade.
#
# NOTE: filename intentionally avoids the "chain_ab" substring (a leftover
# cleanup loop on the 160 host used to pkill that pattern).
#
# Arms (single scenario, dual evaluation conditions):
#   C   : non-learning baselines (semantic_greedy, always_cache)
#   A2  : bubbles profile, proposed two-timescale PPO, seeds 0/1/2
#   B2  : bubbles, conflict dual channel disabled (--lambda-max-conflict 0)
#   B1  : bubbles, slow mobility head disabled (--no-mobility-actor)
#   A1  : legacy profile (scenario_profile null), regression reference
# Each trained checkpoint is evaluated twice:
#   peak    : scenario=utm_conflict (background intents, conflict pressure)
#   nominal : scenario=nominal (bubbles_daily arrivals, no background intents)
#
# Success criteria evaluated by scripts/summarize_ab_bubbles_v3.py at chain end:
#   1. B2 conflict - A2 conflict >= 0.05   (dual channel is load-bearing)
#   2. A2 conflict <= 0.15                 (pulled by conflict_cost_limit 0.08)
#   3. A2 cache ratio <= 0.35
#   4. A2 average_accuracy >= A1 average_accuracy - 0.02
#   5. B1 conflict - A2 conflict >= 0.05   (slow-head effect survives recalib)
#   6. nominal A2 semSucc >= 0.92 and task success >= 0.30 (v2 smoke levels)
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
OUT="outputs/rl/ab_bubbles_v3"
BUBBLES_CFG="configs/v1_9_bubbles.yaml"
LEGACY_CFG="configs/v1_9_snr_lut.yaml"
mkdir -p "$OUT"
STATUS="$OUT/chain_status.tsv"
if [ ! -f "$STATUS" ]; then
  printf "arm\tseed\tconfig\texit\tstart\tend\tdir\n" > "$STATUS"
fi

# v3 calibration knobs are the new code defaults; passed explicitly so the
# run_config.json provenance is self-describing.
RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  --scenario "$SCENARIO" --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

run_one() {
  local arm="$1" cfg="$2" seed="$3"; shift 3
  local dir="$OUT/${arm}_${seed}${DIR_SUFFIX:-}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[chain] SKIP  arm=$arm seed=$seed (already complete)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$seed" "$cfg" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[chain] START arm=$arm seed=$seed cfg=$cfg $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$cfg" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  local end; end="$(date +%Y-%m-%dT%H:%M:%S)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$seed" "$cfg" "$ec" "$start" "$end" "$dir" >> "$STATUS"
  echo "[chain] END   arm=$arm seed=$seed exit=$ec $(date)"
  return "$ec"
}

# Nominal-condition evaluation of an already-trained checkpoint (no retrain).
run_nom_eval() {
  local arm="$1" cfg="$2" seed="$3"
  local train_dir="$OUT/${arm}_${seed}"
  local dir="$OUT/${arm}_${seed}_nom"
  local model="$train_dir/ppo_two_timescale_policy.pt"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[chain] SKIP  arm=$arm seed=$seed nominal-eval (already complete)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}nom" "$seed" "$cfg" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  if [ ! -f "$model" ]; then
    echo "[chain] MISS  arm=$arm seed=$seed nominal-eval (no checkpoint $model)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}nom" "$seed" "$cfg" "no-ckpt" "-" "-" "$dir" >> "$STATUS"
    return 1
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[chain] START arm=$arm seed=$seed nominal-eval $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$cfg" \
    --policy ppo --two-timescale-ppo --load-ppo-model "$model" \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  local end; end="$(date +%Y-%m-%dT%H:%M:%S)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}nom" "$seed" "$cfg" "$ec" "$start" "$end" "$dir" >> "$STATUS"
  echo "[chain] END   arm=$arm seed=$seed nominal-eval exit=$ec $(date)"
  return "$ec"
}

echo "[chain] ==== BUBBLES A/B v3 chain start $(date) | train_ep=$TRAIN_EP eval_ep=$EVAL_EP tpe=$TPE seeds=$SEEDS ===="

# --- C: non-learning baselines, both conditions (fast, run first) ---
run_one Cgreedy "$BUBBLES_CFG" 0 --policy semantic_greedy --scenario "$SCENARIO" \
  --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
run_one Ccache "$BUBBLES_CFG" 0 --policy always_cache --scenario "$SCENARIO" \
  --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
DIR_SUFFIX="_nom" run_one Cgreedy "$BUBBLES_CFG" 0 --policy semantic_greedy --scenario "$NOM_SCENARIO" \
  --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
DIR_SUFFIX="_nom" run_one Ccache "$BUBBLES_CFG" 0 --policy always_cache --scenario "$NOM_SCENARIO" \
  --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"

# --- A2: bubbles, proposed two-timescale, dual-only conflict channel ---
for seed in $SEEDS; do
  run_one A2 "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}"
  run_nom_eval A2 "$BUBBLES_CFG" "$seed"
done

# --- B2: bubbles, conflict dual channel OFF (must degrade in v3) ---
for seed in $SEEDS; do
  run_one B2 "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}" --lambda-max-conflict 0.0
  run_nom_eval B2 "$BUBBLES_CFG" "$seed"
done

# --- B1: bubbles, no slow mobility head ---
for seed in $SEEDS; do
  run_one B1 "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}" --no-mobility-actor
  run_nom_eval B1 "$BUBBLES_CFG" "$seed"
done

# --- A1: legacy profile regression reference ---
for seed in $SEEDS; do
  run_one A1 "$LEGACY_CFG" "$seed" "${RL_COMMON[@]}"
  run_nom_eval A1 "$LEGACY_CFG" "$seed"
done

echo "[chain] ==== chain done $(date) ===="
"$ENVPY" scripts/summarize_ab_bubbles_v3.py --root "$OUT" || echo "[chain] summary script failed"
echo "[chain] status table:"
cat "$STATUS"
