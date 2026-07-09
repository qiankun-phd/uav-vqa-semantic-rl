#!/usr/bin/env bash
# Task #33/#34: v6 scale-consistency matrix -- comm-window deadlines + escalation.
#
#   * attainability_v5 two-key (risk x qtype) LUT anchor {(critical,counting):0.464,
#     (critical,presence):0.696, normal:0.529}
#   * lut_v5 unified count-bucket quality backend (change 2)
#   * cache_quality=entry_v2 s0 quality from the real cached-answer LCB (change 3)
#   * escalation layer: critical reject/expired tasks that are spec-UNattainable are
#     escalated (no quality cost) instead of pinning lambda_quality (change 5)
#   * lambda_decay=0 pure projection on the quality_*/escalation channels (change 6)
#   * critical_cache_compliance=forbidden (v3 ban, now gated to spec_attainable)
#
# delta_esc from the v6 comm-window calibration (scripts/calibrate_epsilon_v6.py,
# outputs/rl/eps_v6_calib.json, tau_conf=2sigma): peak 0.155 (spec-unattainable
# 0.105 quality floor + 0.05), nominal 0.05 (spec-unattainable 0.0 + 0.05).
#
# Same constraint-sensitive arm matrix as v4 (proposed / no_lagrangian /
# fixed_penalty x3 seeds, e4lut x2 backend ablation, non-learning baselines,
# nominal proposed x3 trained-on-nominal).  Output under outputs/rl/eps_recal_v5.
# Filename avoids the "chain_ab" substring (leftover pkill loops on this host).
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
OUT="${OUT:-outputs/rl/eps_recal_v6}"
CFG="configs/v1_9_bubbles.yaml"
CAL="attainability_v5"
CCC="forbidden"
CACHEQ="entry_v2"
ESC="spec_attainable"
QB="lut_v5"
DELTA_PEAK="${DELTA_PEAK:-0.155}"
DELTA_NOM="${DELTA_NOM:-0.05}"
mkdir -p "$OUT"
STATUS="$OUT/train_status.tsv"
[ -f "$STATUS" ] || printf "arm\tseed\texit\tstart\tend\tdir\n" > "$STATUS"

# v5 gate set shared by every arm/condition.
DS="comm_window"   # task #33-A: flight excluded from the deadline clock
V5_GATES=(--epsilon-calibration "$CAL" --critical-cache-compliance "$CCC"
  --cache-quality "$CACHEQ" --escalation-mode "$ESC" --quality-backend "$QB"
  --deadline-semantics "$DS")

RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  "${V5_GATES[@]}"
  --scenario "$SCENARIO" --escalation-cost-limit "$DELTA_PEAK"
  --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

# Static-penalty ablation: frozen v3 A2 convergence-band terminals; escalation
# dual left at 0 (the point is that fixed duals do not adapt).
FIXED_PENALTY=(--freeze-lambda --lambda-init-conflict 4.0
  --lambda-init-quality-critical 9.74 --lambda-init-deadline-critical 6.84
  --lambda-init-battery 0.92)

run_one() {
  local arm="$1" seed="$2"; shift 2
  local dir="$OUT/${arm}_${seed}${DIR_SUFFIX:-}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[recal6] SKIP  arm=$arm seed=$seed${DIR_SUFFIX:-} (done)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[recal6] START arm=$arm seed=$seed${DIR_SUFFIX:-} $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$CFG" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "$ec" "$start" "$(date +%Y-%m-%dT%H:%M:%S)" "$dir" >> "$STATUS"
  echo "[recal6] END   arm=$arm seed=$seed${DIR_SUFFIX:-} exit=$ec $(date)"
  return "$ec"
}

# Evaluate a peak-trained model on the nominal condition (nominal escalation budget).
run_nom_eval() {
  local arm="$1" seed="$2" model_file="$3"; shift 3
  local model="$OUT/${arm}_${seed}/$model_file"
  if [ ! -f "$model" ]; then
    echo "[recal6] MISS  arm=$arm seed=$seed nom-eval (no ckpt $model)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}_nom" "$seed" "no-ckpt" "-" "-" "-" >> "$STATUS"
    return 1
  fi
  DIR_SUFFIX="_nom" run_one "$arm" "$seed" \
    --policy ppo --load-ppo-model "$model" --two-timescale-ppo \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    "${V5_GATES[@]}" --escalation-cost-limit "$DELTA_NOM" \
    --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" "$@"
}

echo "[recal6] ==== epsilon recalibration v5 start $(date) | cal=$CAL ccc=$CCC cacheq=$CACHEQ esc=$ESC qb=$QB delta_peak=$DELTA_PEAK delta_nom=$DELTA_NOM train_ep=$TRAIN_EP seeds=$SEEDS ===="

# --- Non-learning baselines, both conditions ---
for pol in oracle_best_feasible_evidence oracle_escalation_aware semantic_greedy always_cache; do
  run_one "bl_${pol}" 0 --policy "$pol" --scenario "$SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    "${V5_GATES[@]}" --escalation-cost-limit "$DELTA_PEAK"
  DIR_SUFFIX="_nom" run_one "bl_${pol}" 0 --policy "$pol" --scenario "$NOM_SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    "${V5_GATES[@]}" --escalation-cost-limit "$DELTA_NOM"
done
for seed in $SEEDS; do
  run_one bl_random "$seed" --policy random --scenario "$SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    "${V5_GATES[@]}" --escalation-cost-limit "$DELTA_PEAK"
done

# --- Constraint-sensitive learning arms x 3 seeds (peak train + nominal eval) ---
for seed in $SEEDS; do
  run_one proposed "$seed" "${RL_COMMON[@]}"
  run_nom_eval proposed "$seed" ppo_two_timescale_policy.pt
done
for seed in $SEEDS; do
  run_one no_lagrangian "$seed" "${RL_COMMON[@]}" --no-constrained-ppo
  run_nom_eval no_lagrangian "$seed" ppo_two_timescale_policy.pt
done
for seed in $SEEDS; do
  run_one fixed_penalty "$seed" "${RL_COMMON[@]}" "${FIXED_PENALTY[@]}"
  run_nom_eval fixed_penalty "$seed" ppo_two_timescale_policy.pt
done

# --- Backend ablation: legacy v1_9 LUT quality instead of lut_v5 x 2 seeds ---
for seed in 0 1; do
  run_one e4lut "$seed" --policy ppo --proposed-semantic-rl --two-timescale-ppo \
    --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0 \
    --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08 \
    --epsilon-calibration "$CAL" --critical-cache-compliance "$CCC" \
    --cache-quality "$CACHEQ" --escalation-mode "$ESC" --quality-backend lut \
    --scenario "$SCENARIO" --escalation-cost-limit "$DELTA_PEAK" \
    --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
  run_nom_eval e4lut "$seed" ppo_two_timescale_policy.pt --quality-backend lut
done

# --- Nominal-trained proposed x 3 seeds (train ON nominal, nominal budget) ---
for seed in $SEEDS; do
  DIR_SUFFIX="_nomtrain" run_one proposed "$seed" \
    --policy ppo --proposed-semantic-rl --two-timescale-ppo \
    --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0 \
    --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08 \
    "${V5_GATES[@]:0:8}" --quality-backend "$QB" --deadline-semantics "$DS" \
    --scenario "$NOM_SCENARIO" --escalation-cost-limit "$DELTA_NOM" \
    --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
done

echo "[recal6] ==== v5 run done $(date) ===="
cat "$STATUS"
touch "$OUT/RUN_FINISHED"
