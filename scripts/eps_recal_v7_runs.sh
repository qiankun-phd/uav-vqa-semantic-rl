#!/usr/bin/env bash
# Task #35/#36: v7 matrix -- mission-aligned reward + link-model guards on top of
# the v6 scale-consistency gates (comm-window deadlines, 2sigma tau, escalation,
# forbidden cache).  The three v7 additions (all config-gated, default legacy):
#
#   * reward_success_semantics=mission_aligned (#36): a quality+deadline-
#     COMPLIANT service that is only UTM-blocked earns a discounted success bonus
#     and is not charged the airspace penalty (a UTM block is an airspace event,
#     not a service failure) -- flips the compliant-blocked vs banned-cache
#     reward ordering (-1.598 -> +0.778 vs cache -1.08) so cache stops flooding
#     and the quality-critical dual can price;
#   * reference_bandwidth=fair_share (#35-2): the obs SNR bin / spec-attainability
#     certificate anchor on pool / N_uav instead of the 50 kHz s0 default;
#   * lut_support_guard=outage (#35-1): a service whose effective SINR is below
#     the lowest LUT bin (> 2.5 dB) is a quality outage (LCB 0), top clamp above.
#
# Re-calibration (scripts/calibrate_epsilon_v7.py, --link-model corrected) shows
# the link guards do NOT move the utm_conflict peak delta_esc (candidate SINRs
# 26-36 dB, all in-support), so v7 keeps the v6 budget: delta_esc peak 0.155
# (spec-unattainable 0.105 + 0.05), nominal 0.05.  epsilon anchors unchanged
# (attainability_v5: cc 0.464 / cp 0.696 / normal 0.529, tau_conf 2sigma).
#
# Same arm matrix as v6 (proposed / no_lagrangian / fixed_penalty x3 seeds,
# e4lut x2 backend ablation, non-learning baselines, nominal proposed x3).
# Output under outputs/rl/eps_recal_v7.  tmux epsv7.
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
OUT="${OUT:-outputs/rl/eps_recal_v7}"
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

# v7 link-model + reward flags shared by every arm/condition (the new v7 gates).
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

RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  "${V7_GATES[@]}"
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
    echo "[recal7] SKIP  arm=$arm seed=$seed${DIR_SUFFIX:-} (done)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[recal7] START arm=$arm seed=$seed${DIR_SUFFIX:-} $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$CFG" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "$ec" "$start" "$(date +%Y-%m-%dT%H:%M:%S)" "$dir" >> "$STATUS"
  echo "[recal7] END   arm=$arm seed=$seed${DIR_SUFFIX:-} exit=$ec $(date)"
  return "$ec"
}

# Evaluate a peak-trained model on the nominal condition (nominal escalation budget).
run_nom_eval() {
  local arm="$1" seed="$2" model_file="$3"; shift 3
  local model="$OUT/${arm}_${seed}/$model_file"
  if [ ! -f "$model" ]; then
    echo "[recal7] MISS  arm=$arm seed=$seed nom-eval (no ckpt $model)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}_nom" "$seed" "no-ckpt" "-" "-" "-" >> "$STATUS"
    return 1
  fi
  DIR_SUFFIX="_nom" run_one "$arm" "$seed" \
    --policy ppo --load-ppo-model "$model" --two-timescale-ppo \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    "${V7_GATES[@]}" --escalation-cost-limit "$DELTA_NOM" \
    --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" "$@"
}

echo "[recal7] ==== epsilon recalibration v7 start $(date) | cal=$CAL ccc=$CCC cacheq=$CACHEQ esc=$ESC qb=$QB rss=$RSS refbw=$REFBW lutg=$LUTG delta_peak=$DELTA_PEAK delta_nom=$DELTA_NOM train_ep=$TRAIN_EP seeds=$SEEDS ===="

# --- Non-learning baselines, both conditions ---
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
    "${V6_GATES[@]:0:8}" --quality-backend "$QB" --deadline-semantics "$DS" "${V7_LINK[@]}" \
    --scenario "$NOM_SCENARIO" --escalation-cost-limit "$DELTA_NOM" \
    --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
done

echo "[recal7] ==== v7 run done $(date) ===="
cat "$STATUS"
touch "$OUT/RUN_FINISHED"
