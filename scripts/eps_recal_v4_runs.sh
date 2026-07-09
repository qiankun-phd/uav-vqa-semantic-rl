#!/usr/bin/env bash
# Task #28: epsilon recalibration iteration 4 -- attainability_v4 (transmission-only anchor,
# cache-compliance ban, method (c)).  eps_critical=0.504 (pure P10 attainability
# anchor, v2 cache guardrail DROPPED), eps_normal=0.166 (restored v1 anchor);
# the cache shortcut is closed at the compliance-judgment layer instead of via
# epsilon, gated by --critical-cache-compliance forbidden (default allowed keeps
# legacy/v1/v2 reproducible).
#
# Same constraint-sensitive arm matrix as v1/v2 (proposed / no_lagrangian /
# fixed_penalty x3 seeds, e4lut x2, non-learning baselines re-eval; peak=
# utm_conflict + nominal).  All output under outputs/rl/eps_recal_v4.  Legacy /
# v1 / v2 runs are the comparison baseline.
#
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
OUT="${OUT:-outputs/rl/eps_recal_v4}"
CFG="configs/v1_9_bubbles.yaml"
CAL="attainability_v4"
CCC="forbidden"          # critical_cache_compliance: structural ban (method c)
mkdir -p "$OUT"
STATUS="$OUT/train_status.tsv"
[ -f "$STATUS" ] || printf "arm\tseed\texit\tstart\tend\tdir\n" > "$STATUS"

RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  --epsilon-calibration "$CAL" --critical-cache-compliance "$CCC"
  --scenario "$SCENARIO" --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

# Frozen dual terminals from the legacy v3 A2 convergence band (unchanged: this
# is the *static-penalty* ablation -- its point is that fixed duals do not adapt
# to the recalibrated constraint).
FIXED_PENALTY=(--freeze-lambda --lambda-init-conflict 4.0
  --lambda-init-quality-critical 9.74 --lambda-init-deadline-critical 6.84
  --lambda-init-battery 0.92)

run_one() {
  local arm="$1" seed="$2"; shift 2
  local dir="$OUT/${arm}_${seed}${DIR_SUFFIX:-}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[recal] SKIP  arm=$arm seed=$seed${DIR_SUFFIX:-} (done)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[recal] START arm=$arm seed=$seed${DIR_SUFFIX:-} $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$CFG" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "$ec" "$start" "$(date +%Y-%m-%dT%H:%M:%S)" "$dir" >> "$STATUS"
  echo "[recal] END   arm=$arm seed=$seed${DIR_SUFFIX:-} exit=$ec $(date)"
  return "$ec"
}

run_nom_eval() {
  local arm="$1" seed="$2" model_file="$3"; shift 3
  local model="$OUT/${arm}_${seed}/$model_file"
  if [ ! -f "$model" ]; then
    echo "[recal] MISS  arm=$arm seed=$seed nom-eval (no ckpt $model)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}_nom" "$seed" "no-ckpt" "-" "-" "-" >> "$STATUS"
    return 1
  fi
  DIR_SUFFIX="_nom" run_one "$arm" "$seed" \
    --policy ppo --load-ppo-model "$model" --two-timescale-ppo \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    --epsilon-calibration "$CAL" --critical-cache-compliance "$CCC" \
    --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" "$@"
}

echo "[recal] ==== epsilon recalibration v4 rerun start $(date) | cal=$CAL ccc=$CCC train_ep=$TRAIN_EP seeds=$SEEDS ===="

# --- Non-learning baselines, both conditions (fast) ---
for pol in oracle_best_feasible_evidence semantic_greedy always_cache; do
  run_one "bl_${pol}" 0 --policy "$pol" --scenario "$SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    --epsilon-calibration "$CAL" --critical-cache-compliance "$CCC"
  DIR_SUFFIX="_nom" run_one "bl_${pol}" 0 --policy "$pol" --scenario "$NOM_SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    --epsilon-calibration "$CAL" --critical-cache-compliance "$CCC"
done
for seed in $SEEDS; do
  run_one bl_random "$seed" --policy random --scenario "$SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    --epsilon-calibration "$CAL" --critical-cache-compliance "$CCC"
  DIR_SUFFIX="_nom" run_one bl_random "$seed" --policy random --scenario "$NOM_SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    --epsilon-calibration "$CAL" --critical-cache-compliance "$CCC"
done

# --- Constraint-sensitive learning arms x 3 seeds, dual condition ---
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

# --- E4 lut-backend peak check x 2 seeds ---
for seed in 0 1; do
  run_one e4lut "$seed" "${RL_COMMON[@]}" --quality-backend lut
  run_nom_eval e4lut "$seed" ppo_two_timescale_policy.pt --quality-backend lut
done

echo "[recal] ==== v4 run done $(date) ===="
cat "$STATUS"
touch "$OUT/RUN_FINISHED"
