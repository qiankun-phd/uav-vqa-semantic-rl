#!/usr/bin/env bash
# Paper-2 experiment matrix -- batch 3: zero-shot generalization sweeps.
#
# Spec: docs_spec/RL_Experiment_Standards_Survey.md section 3 (Fig.3-5, Fig.7).
# Protocol: NO retraining -- every learning-arm checkpoint from batch 2
# (outputs/rl/matrix_v1/{arm}_{seed}) is evaluated zero-shot at every sweep
# point, together with the non-learning baselines, 32 eval episodes per point
# (>=32 rollouts).  Literature anchor: Khairy JSAC'21 zero-shot protocol;
# nobody retrains baselines per generalization point.
#
# Axes (default point = utm_conflict, tpe=20, num_uavs=4 preset, full SNR bins):
#   uav  : num_uavs in {2,3,4,6}
#   load : tasks_per_episode in {10,20,30,40} (x0.5/x1/x1.5/x2 of nominal 20;
#          episode_steps stays 10, so tpe scales the arrival load)
#   snr  : sensed SNR bins shifted {-5,0 | 0,5 | 5,10 | 10,15 | 15,20} dB
# Fig.7 zero-shot specials:
#   zs_legacy : unseen profile (legacy scenario_profile=null config) --
#               proposed zero-shot vs retrain_legacy reference vs baselines
#   x1.5 load  : covered by load30 + retrain_load15 reference evaluated there
#   low SNR    : covered by snr_m5_0 + retrain_lowsnr reference evaluated there
set -u

ENVPY="$HOME/.conda/envs/uav_semcom/bin/python"
ROOT="$HOME/phd_research/vqa_main"
cd "$ROOT" || { echo "cannot cd $ROOT"; exit 2; }

SCENARIO="utm_conflict"
EVAL_EP="${EVAL_EP:-32}"
TPE="${TPE:-20}"
SEEDS="${SEEDS:-0 1 2}"
TRAIN_OUT="${TRAIN_OUT:-outputs/rl/matrix_v1}"
OUT="${OUT:-outputs/rl/matrix_v1/sweep}"
BUBBLES_CFG="configs/v1_9_bubbles.yaml"
LEGACY_CFG="configs/v1_9_snr_lut.yaml"
mkdir -p "$OUT"
STATUS="$OUT/sweep_status.tsv"
if [ ! -f "$STATUS" ]; then
  printf "point\tarm\tseed\texit\tstart\tend\tdir\n" > "$STATUS"
fi

# arm -> "model_file load_flags..."
model_file_for() {
  case "$1" in
    proposed|no_lagrangian|fixed_penalty|retrain_legacy|retrain_load15|retrain_lowsnr)
      echo "ppo_two_timescale_policy.pt" ;;
    flat_ppo) echo "ppo_hybrid_policy.pt" ;;
    service_only) echo "ppo_service_policy.pt" ;;
    *) echo ""; return 1 ;;
  esac
}
load_flags_for() {
  case "$1" in
    proposed|no_lagrangian|fixed_penalty|retrain_legacy|retrain_load15|retrain_lowsnr)
      echo "--two-timescale-ppo" ;;
    flat_ppo) echo "" ;;
    service_only) echo "--service-only-ppo" ;;
  esac
}

# eval_run point arm seed cfg [extra args...]
eval_run() {
  local point="$1" arm="$2" seed="$3" cfg="$4"; shift 4
  local dir="$OUT/$point/${arm}_${seed}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$point" "$arm" "$seed" "skip" "-" "-" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[sweep] START point=$point arm=$arm seed=$seed $(date +%H:%M:%S)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$cfg" \
    --state-version v2 --device cuda:0 --episodes "$EVAL_EP" \
    --seed "$seed" --output-dir "$dir" "$@" > "$dir/run.log" 2>&1
  local ec=$?
  local end; end="$(date +%Y-%m-%dT%H:%M:%S)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$point" "$arm" "$seed" "$ec" "$start" "$end" "$dir" >> "$STATUS"
  [ "$ec" -ne 0 ] && echo "[sweep] FAIL  point=$point arm=$arm seed=$seed exit=$ec"
  return "$ec"
}

# eval_ckpt point arm seed cfg [extra args...] -- zero-shot checkpoint eval
eval_ckpt() {
  local point="$1" arm="$2" seed="$3" cfg="$4"; shift 4
  local model_dir_arm="$arm"
  local model="$TRAIN_OUT/${model_dir_arm}_${seed}/$(model_file_for "$arm")"
  if [ ! -f "$model" ]; then
    echo "[sweep] MISS  point=$point arm=$arm seed=$seed (no checkpoint $model)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$point" "$arm" "$seed" "no-ckpt" "-" "-" "-" >> "$STATUS"
    return 1
  fi
  # shellcheck disable=SC2046
  eval_run "$point" "$arm" "$seed" "$cfg" \
    --policy ppo --load-ppo-model "$model" $(load_flags_for "$arm") \
    --hidden-size 128 "$@"
}

LEARN_ARMS="proposed no_lagrangian fixed_penalty flat_ppo service_only"

# run_point point cfg tpe [extra env args...] -- all arms at one sweep point
run_point() {
  local point="$1" cfg="$2" tpe="$3"; shift 3
  for arm in $LEARN_ARMS; do
    for seed in $SEEDS; do
      eval_ckpt "$point" "$arm" "$seed" "$cfg" --scenario "$SCENARIO" --tasks-per-episode "$tpe" "$@"
    done
  done
  for pol in semantic_greedy always_cache oracle_best_feasible_evidence; do
    eval_run "$point" "bl_${pol}" 0 "$cfg" --policy "$pol" --scenario "$SCENARIO" --tasks-per-episode "$tpe" "$@"
  done
  for seed in $SEEDS; do
    eval_run "$point" bl_random "$seed" "$cfg" --policy random --scenario "$SCENARIO" --tasks-per-episode "$tpe" "$@"
  done
}

echo "[sweep] ==== matrix_v1 sweep start $(date) | eval_ep=$EVAL_EP seeds=$SEEDS ===="

# --- Axis 1: UAV count (Fig.3) ---
for n in 2 3 4 6; do
  run_point "uav/uav${n}" "$BUBBLES_CFG" "$TPE" --num-uavs "$n"
done

# --- Axis 2: task arrival load (Fig.4) ---
for t in 10 20 30 40; do
  run_point "load/load${t}" "$BUBBLES_CFG" "$t"
done

# --- Axis 3: sensed SNR band (Fig.5) ---
run_point "snr/snr_m5_0"  "$BUBBLES_CFG" "$TPE" --snr-bins=-5,0
run_point "snr/snr_0_5"   "$BUBBLES_CFG" "$TPE" --snr-bins=0,5
run_point "snr/snr_5_10"  "$BUBBLES_CFG" "$TPE" --snr-bins=5,10
run_point "snr/snr_10_15" "$BUBBLES_CFG" "$TPE" --snr-bins=10,15
run_point "snr/snr_15_20" "$BUBBLES_CFG" "$TPE" --snr-bins=15,20

# --- Fig.7 zero-shot specials ---
# (a) unseen profile: legacy config (scenario_profile=null), same utm_conflict scenario
run_point "zeroshot/zs_legacy" "$LEGACY_CFG" "$TPE"
eval_ckpt "zeroshot/zs_legacy" retrain_legacy 0 "$LEGACY_CFG" --scenario "$SCENARIO" --tasks-per-episode "$TPE"
# (b) x1.5 load retrain reference at the load30 point
eval_ckpt "load/load30" retrain_load15 0 "$BUBBLES_CFG" --scenario "$SCENARIO" --tasks-per-episode 30
# (c) low-SNR retrain reference at the snr_m5_0 point
eval_ckpt "snr/snr_m5_0" retrain_lowsnr 0 "$BUBBLES_CFG" --scenario "$SCENARIO" --tasks-per-episode "$TPE" --snr-bins=-5,0

echo "[sweep] ==== sweep done $(date) ===="
awk -F'\t' 'NR>1 && $4!="0" && $4!="skip"' "$STATUS" | tee "$OUT/sweep_failures.tsv" | head
echo "[sweep] failures: $(awk -F"\t" "NR>1 && \$4!=\"0\" && \$4!=\"skip\"" "$STATUS" | wc -l)"
