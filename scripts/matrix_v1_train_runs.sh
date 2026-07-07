#!/usr/bin/env bash
# Paper-2 experiment matrix -- batch 2: training runs (matrix_v1).
#
# Spec: docs_spec/RL_Experiment_Standards_Survey.md (section 3 final matrix).
# Protocol: 1000 train episodes x 3 seeds per learning arm; every checkpoint
# evaluated under BOTH conditions (peak scenario=utm_conflict / nominal),
# 50 eval episodes (>=32 rollouts) x 20 tasks-per-episode.
#
# Learning arms (all on configs/v1_9_bubbles.yaml, scenario=utm_conflict):
#   proposed      : v3 A2 config (two-timescale semantic-Lyapunov constrained PPO)
#   no_lagrangian : proposed + --no-constrained-ppo (unconstrained ablation of
#                   the Khairy comparison triple; all dual channels off)
#   fixed_penalty : proposed + --freeze-lambda with duals pinned at the v3 A2
#                   500-ep convergence-band terminals (quality_critical 9.74,
#                   deadline_critical 6.84, conflict 4.0, battery 0.92) --
#                   the static-penalty method of the Khairy triple
#   flat_ppo      : conventional/flat PPO (Lya-HiPPO-style baseline): bare
#                   single-timescale hybrid multi-head PPO -- no duals, no
#                   safety layer, no semantic/resource projection, no Lyapunov
#                   queues, no BC warm start, raw env reward
#   service_only  : flat_ppo + --service-only-ppo (discrete head only,
#                   continuous resources at defaults -- the "combinatorial"
#                   hybrid-action fairness baseline)
# Fig.2 material : proposed at --num-uavs {2,6}, seed 0 (default 4 = proposed_0)
# Fig.7 retrain references (1 seed each): legacy profile / x1.5 load (tpe=30)
#   / low SNR (--snr-bins=-5,0)
# Non-learning baselines, both conditions: semantic_greedy / always_cache /
#   oracle_best_feasible_evidence (seed 0) and random (seeds 0 1 2).
#
# NOTE: filename intentionally avoids the "chain_ab" substring (a leftover
# cleanup loop on the 160 host used to pkill that pattern).
set -u

ENVPY="$HOME/.conda/envs/uav_semcom/bin/python"
ROOT="$HOME/phd_research/vqa_main"
cd "$ROOT" || { echo "cannot cd $ROOT"; exit 2; }

SCENARIO="utm_conflict"
NOM_SCENARIO="nominal"
TRAIN_EP="${TRAIN_EP:-1000}"
EVAL_EP="${EVAL_EP:-50}"
TPE="${TPE:-20}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-outputs/rl/matrix_v1}"
BUBBLES_CFG="configs/v1_9_bubbles.yaml"
LEGACY_CFG="configs/v1_9_snr_lut.yaml"
mkdir -p "$OUT"
STATUS="$OUT/train_status.tsv"
if [ ! -f "$STATUS" ]; then
  printf "arm\tseed\texit\tstart\tend\tdir\n" > "$STATUS"
fi

RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  --scenario "$SCENARIO" --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

# Frozen dual values: v3 A2 seeds 0/1/2 lambda-trace terminals (ep 499 means).
FIXED_PENALTY=(--freeze-lambda --lambda-init-conflict 4.0
  --lambda-init-quality-critical 9.74 --lambda-init-deadline-critical 6.84
  --lambda-init-battery 0.92)

FLAT_COMMON=(--policy ppo --train-ppo
  --no-constrained-ppo --no-safety-layer --no-semantic-projection
  --no-resource-projection --no-lyapunov-queues --semantic-reward-mode env
  --state-version v2 --hidden-size 128 --device cuda:0
  --scenario "$SCENARIO" --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

run_one() {
  local arm="$1" cfg="$2" seed="$3"; shift 3
  local dir="$OUT/${arm}_${seed}${DIR_SUFFIX:-}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[matrix] SKIP  arm=$arm seed=$seed (already complete)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[matrix] START arm=$arm seed=$seed $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$cfg" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  local end; end="$(date +%Y-%m-%dT%H:%M:%S)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm${DIR_SUFFIX:-}" "$seed" "$ec" "$start" "$end" "$dir" >> "$STATUS"
  echo "[matrix] END   arm=$arm seed=$seed exit=$ec $(date)"
  return "$ec"
}

# Nominal-condition evaluation of an already-trained checkpoint (no retrain).
# args: arm cfg seed model_file extra-load-flags...
run_nom_eval() {
  local arm="$1" cfg="$2" seed="$3" model_file="$4"; shift 4
  local train_dir="$OUT/${arm}_${seed}"
  local dir="$OUT/${arm}_${seed}_nom"
  local model="$train_dir/$model_file"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[matrix] SKIP  arm=$arm seed=$seed nominal-eval (already complete)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}_nom" "$seed" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  if [ ! -f "$model" ]; then
    echo "[matrix] MISS  arm=$arm seed=$seed nominal-eval (no checkpoint $model)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}_nom" "$seed" "no-ckpt" "-" "-" "$dir" >> "$STATUS"
    return 1
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[matrix] START arm=$arm seed=$seed nominal-eval $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$cfg" \
    --policy ppo --load-ppo-model "$model" "$@" \
    --state-version v2 --hidden-size 128 --device cuda:0 \
    --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  local end; end="$(date +%Y-%m-%dT%H:%M:%S)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${arm}_nom" "$seed" "$ec" "$start" "$end" "$dir" >> "$STATUS"
  echo "[matrix] END   arm=$arm seed=$seed nominal-eval exit=$ec $(date)"
  return "$ec"
}

echo "[matrix] ==== matrix_v1 training batch start $(date) | train_ep=$TRAIN_EP eval_ep=$EVAL_EP tpe=$TPE seeds=$SEEDS ===="

# --- Non-learning baselines, both conditions (fast, run first) ---
for pol in semantic_greedy always_cache oracle_best_feasible_evidence; do
  run_one "bl_${pol}" "$BUBBLES_CFG" 0 --policy "$pol" --scenario "$SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
  DIR_SUFFIX="_nom" run_one "bl_${pol}" "$BUBBLES_CFG" 0 --policy "$pol" --scenario "$NOM_SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
done
for seed in $SEEDS; do
  run_one bl_random "$BUBBLES_CFG" "$seed" --policy random --scenario "$SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
  DIR_SUFFIX="_nom" run_one bl_random "$BUBBLES_CFG" "$seed" --policy random --scenario "$NOM_SCENARIO" \
    --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
done

# --- Learning arms x 3 seeds, dual-condition evaluation ---
for seed in $SEEDS; do
  run_one proposed "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}"
  run_nom_eval proposed "$BUBBLES_CFG" "$seed" ppo_two_timescale_policy.pt --two-timescale-ppo
done
for seed in $SEEDS; do
  run_one no_lagrangian "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}" --no-constrained-ppo
  run_nom_eval no_lagrangian "$BUBBLES_CFG" "$seed" ppo_two_timescale_policy.pt --two-timescale-ppo
done
for seed in $SEEDS; do
  run_one fixed_penalty "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}" "${FIXED_PENALTY[@]}"
  run_nom_eval fixed_penalty "$BUBBLES_CFG" "$seed" ppo_two_timescale_policy.pt --two-timescale-ppo
done
for seed in $SEEDS; do
  run_one flat_ppo "$BUBBLES_CFG" "$seed" "${FLAT_COMMON[@]}"
  run_nom_eval flat_ppo "$BUBBLES_CFG" "$seed" ppo_hybrid_policy.pt
done
for seed in $SEEDS; do
  run_one service_only "$BUBBLES_CFG" "$seed" "${FLAT_COMMON[@]}" --service-only-ppo
  run_nom_eval service_only "$BUBBLES_CFG" "$seed" ppo_service_policy.pt --service-only-ppo
done

# --- Fig.2 material: proposed at UAV-count {2, 6}, seed 0 (default 4 = proposed_0) ---
run_one mscale_uav2 "$BUBBLES_CFG" 0 "${RL_COMMON[@]}" --num-uavs 2
run_one mscale_uav6 "$BUBBLES_CFG" 0 "${RL_COMMON[@]}" --num-uavs 6

# --- Fig.7 retrain references (1 seed, same protocol as proposed) ---
run_one retrain_legacy "$LEGACY_CFG" 0 "${RL_COMMON[@]}"
# argparse last-wins: the trailing --tasks-per-episode 30 overrides RL_COMMON's 20.
run_one retrain_load15 "$BUBBLES_CFG" 0 "${RL_COMMON[@]}" --tasks-per-episode 30
run_one retrain_lowsnr "$BUBBLES_CFG" 0 "${RL_COMMON[@]}" --snr-bins=-5,0

echo "[matrix] ==== training batch done $(date) ===="
echo "[matrix] status table:"
cat "$STATUS"
