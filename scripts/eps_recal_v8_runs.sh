#!/usr/bin/env bash
# Task #37 v8: lambda_QC UNPIN attack -- two levers on top of the FULL v7b gate
# set (which already carries the reward-wiring fix: mission_aligned reaches the
# two-timescale controller gradient, no_lagrangian peak benefited cacheR
# 0.436->0.298 / task 0.242->0.384 / acc->0.213).  v7b still hung 6/13 because
# the quality-critical dual pins at ~18.4: quality_cost_critical ~0.40 sits an
# order of magnitude above its 0.02 limit at every step, so the shared global
# lambda_max=20 lets lambda_QC ratchet up until its dual penalty (lambda*cost
# ~7.4/step) dwarfs the positive semantic-utility terms (~+23) and drowns the
# learning signal (0/343 blocked events at convergence).
#
# The two v8 levers (both config-gated, default legacy, legacy逐位 regression
# guarded in tests/test_lambda_qc_v8.py):
#
#   * lever 1 --lambda-max-quality 8: per-channel dual ceiling for the
#     quality_normal/quality_critical channels, mirroring the conflict-channel
#     precedent (--lambda-max-conflict 8, v19_ppo.py "so it cannot dwarf the
#     positive semantic utility terms").  Bounds the quality dual penalty at
#     "a conflict step is net-negative" instead of "the whole return is swamped".
#   * lever 2 --dual-warmup-episodes 150: freeze ALL dual variables for the
#     first 150 episodes (costs still logged), then resume normal dual ascent.
#     150 is chosen after the steepest BC/service-prior shaping window
#     (service_prior_decay_episodes=240; at ep150 the BC aux weight is at
#     ~37.5% of its initial value) so the policy forms under the fixed-init
#     reward before the constraint price starts moving (without it lambda_QC is
#     already ~5.5 by the time BC aux has decayed).
#
# Everything else is IDENTICAL to eps_recal_v7b_runs.sh (same v6 scale gates,
# same v7 link/reward flags, same delta_esc peak 0.155 / nominal 0.05, same arm
# matrix: proposed/no_lagrangian/fixed_penalty x3 seeds x2 conditions,
# e4lut x2 backend ablation, non-learning baselines both conditions, nominal-
# trained proposed x3).  Output outputs/rl/eps_recal_v8, tmux epsv8.
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
OUT="${OUT:-outputs/rl/eps_recal_v8}"
CFG="configs/v1_9_bubbles.yaml"
CAL="attainability_v5"
CCC="forbidden"
CACHEQ="entry_v2"
ESC="spec_attainable"
QB="lut_v5"
DELTA_PEAK="${DELTA_PEAK:-0.155}"
DELTA_NOM="${DELTA_NOM:-0.05}"
# v8 levers (default legacy is 20 / 0; v8 overrides both).
LAMBDA_MAX_Q="${LAMBDA_MAX_Q:-8.0}"
DUAL_WARMUP="${DUAL_WARMUP:-150}"
mkdir -p "$OUT"
STATUS="$OUT/train_status.tsv"
[ -f "$STATUS" ] || printf "arm\tseed\texit\tstart\tend\tdir\n" > "$STATUS"

# v7 link-model + reward flags shared by every arm/condition.
RSS="mission_aligned"    # #36 mission-aligned reward success attribution
REFBW="fair_share"       # #35-2 fair-share reference bandwidth
LUTG="outage"            # #35-1 LUT support-set outage guard
V7_LINK=(--reward-success-semantics "$RSS" --reference-bandwidth "$REFBW" --lut-support-guard "$LUTG")

# v6 gate set shared by every arm/condition.
DS="comm_window"   # task #33-A: flight excluded from the deadline clock
V6_GATES=(--epsilon-calibration "$CAL" --critical-cache-compliance "$CCC"
  --cache-quality "$CACHEQ" --escalation-mode "$ESC" --quality-backend "$QB"
  --deadline-semantics "$DS")
# Full v7 gate set = v6 gates + the three v7 link/reward flags.
V7_GATES=("${V6_GATES[@]}" "${V7_LINK[@]}")

# v8 dual levers -- applied to the LEARNING arms only (baselines have no dual).
V8_DUAL=(--lambda-max-quality "$LAMBDA_MAX_Q" --dual-warmup-episodes "$DUAL_WARMUP")

RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  "${V8_DUAL[@]}"
  "${V7_GATES[@]}"
  --scenario "$SCENARIO" --escalation-cost-limit "$DELTA_PEAK"
  --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

# Static-penalty ablation: frozen v3 A2 convergence-band terminals; escalation
# dual left at 0 (the point is that fixed duals do not adapt).  With
# --freeze-lambda the v8 warm-up/cap are inert (no dual ascent happens), but the
# lambda_max_quality cap still clamps the quality_critical INIT 9.74 -> 8 at
# _init_dual_state, so fixed_penalty's static quality price is bounded to the
# same ceiling the adaptive arm now respects (documented; consistent framing).
FIXED_PENALTY=(--freeze-lambda --lambda-init-conflict 4.0
  --lambda-init-quality-critical 9.74 --lambda-init-deadline-critical 6.84
  --lambda-init-battery 0.92)

run_one() {
  local arm="$1" seed="$2"; shift 2
  local dir="$OUT/${arm}_${seed}${DIR_SUFFIX:-}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[recal8] SKIP  arm=$arm seed=$seed${DIR_SUFFIX:-} (done)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[recal8] START arm=$arm seed=$seed${DIR_SUFFIX:-} $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$CFG" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "$ec" "$start" "$(date +%Y-%m-%dT%H:%M:%S)" "$dir" >> "$STATUS"
  echo "[recal8] END   arm=$arm seed=$seed${DIR_SUFFIX:-} exit=$ec $(date)"
  return "$ec"
}

# Evaluate a peak-trained model on the nominal condition (nominal escalation budget).
run_nom_eval() {
  local arm="$1" seed="$2" model_file="$3"; shift 3
  local model="$OUT/${arm}_${seed}/$model_file"
  if [ ! -f "$model" ]; then
    echo "[recal8] MISS  arm=$arm seed=$seed nom-eval (no ckpt $model)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}_nom" "$seed" "no-ckpt" "-" "-" "-" >> "$STATUS"
    return 1
  fi
  DIR_SUFFIX="_nom" run_one "$arm" "$seed" \
    --policy ppo --load-ppo-model "$model" --two-timescale-ppo \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    "${V7_GATES[@]}" --escalation-cost-limit "$DELTA_NOM" \
    --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" "$@"
}

echo "[recal8] ==== eps recalibration v8 (lambda_QC unpin) start $(date) | cal=$CAL ccc=$CCC cacheq=$CACHEQ esc=$ESC qb=$QB rss=$RSS refbw=$REFBW lutg=$LUTG delta_peak=$DELTA_PEAK delta_nom=$DELTA_NOM lambda_max_q=$LAMBDA_MAX_Q dual_warmup=$DUAL_WARMUP train_ep=$TRAIN_EP seeds=$SEEDS ===="

# --- Non-learning baselines, both conditions (no dual; v8 levers not applied) ---
for pol in oracle_best_feasible_evidence oracle_escalation_aware semantic_greedy always_cache; do
  run_one "bl_${pol}" 0 --policy "$pol" --scenario "$SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    "${V7_GATES[@]}" --escalation-cost-limit "$DELTA_PEAK"
  DIR_SUFFIX="_nom" run_one "bl_${pol}" 0 --policy "$pol" --scenario "$NOM_SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    "${V7_GATES[@]}" --escalation-cost-limit "$DELTA_NOM"
done
for seed in $SEEDS; do
  run_one bl_random "$seed" --policy random --scenario "$SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    "${V7_GATES[@]}" --escalation-cost-limit "$DELTA_PEAK"
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
    "${V8_DUAL[@]}" \
    --epsilon-calibration "$CAL" --critical-cache-compliance "$CCC" \
    --cache-quality "$CACHEQ" --escalation-mode "$ESC" --quality-backend lut \
    --deadline-semantics "$DS" "${V7_LINK[@]}" \
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
    "${V8_DUAL[@]}" \
    "${V6_GATES[@]:0:8}" --quality-backend "$QB" --deadline-semantics "$DS" "${V7_LINK[@]}" \
    --scenario "$NOM_SCENARIO" --escalation-cost-limit "$DELTA_NOM" \
    --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
done

echo "[recal8] ==== v8 run done $(date) ===="
cat "$STATUS"
touch "$OUT/RUN_FINISHED"
