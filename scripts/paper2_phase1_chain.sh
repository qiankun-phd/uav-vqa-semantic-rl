#!/usr/bin/env bash
# Paper II phase 1 (deep-review P0/P1 experiment tails) on the FROZEN v8 env.
#
#   Block 1  flat_ppo_v8         flat single-head PPO baseline (P1): 3 seeds
#                                peak-train + nominal eval of the peak model.
#   Block 2  m3_ablation_v8      dual-machinery ablations (peak x3 seeds):
#                                  quality_off    lambda_max_quality=0 (channel off)
#                                  warm_start     dual-warmup 0 + lambda_QC init 8.0
#                                                 (start AT the pin; ascent on)
#                                  no_slow_head   --no-mobility-actor
#   Block 3  m4_generalization_v8  eval-only sweeps of the v8 checkpoints
#                                (proposed/no_lagrangian/fixed_penalty x3 seeds
#                                + semantic_greedy + escalation-aware oracle):
#                                  uav     --num-uavs 2/3/4/6/8
#                                  arrival --tasks-per-episode 12/16/20/24/28
#                                  snr     --a2g-excess-loss-db 4/8/12/16/20
#                                  i0      --interference-floor-dbm -120..-116
#                                  zs      utm_conflict_soft | tpe 30 (x1.5) |
#                                          low_snr_blockage
#
# Everything else is IDENTICAL to scripts/eps_recal_v8_runs.sh (same v6 gates,
# v7 link/reward flags, v8 dual levers, delta_esc peak 0.155 / nominal 0.05).
# Training blocks use cuda:0 (light, ~3 min/arm); the M4 sweep is CPU-only so
# it never contends with the paper-1 agent's GPU work.
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
CFG="configs/v1_9_bubbles.yaml"
DELTA_PEAK="0.155"
DELTA_NOM="0.05"
V8ROOT="outputs/rl/eps_recal_v8"

FLAT_OUT="outputs/rl/flat_ppo_v8"
M3_OUT="outputs/rl/m3_ablation_v8"
M4_OUT="outputs/rl/m4_generalization_v8"
mkdir -p "$FLAT_OUT" "$M3_OUT" "$M4_OUT"
STATUS="outputs/rl/paper2_phase1_status.tsv"
[ -f "$STATUS" ] || printf "block\tarm\tseed\texit\tstart\tend\tdir\n" > "$STATUS"

V7_LINK=(--reward-success-semantics mission_aligned --reference-bandwidth fair_share --lut-support-guard outage)
V6_GATES=(--epsilon-calibration attainability_v5 --critical-cache-compliance forbidden
  --cache-quality entry_v2 --escalation-mode spec_attainable --quality-backend lut_v5
  --deadline-semantics comm_window)
V7_GATES=("${V6_GATES[@]}" "${V7_LINK[@]}")
V8_DUAL=(--lambda-max-quality 8.0 --dual-warmup-episodes 150)

# GPU courtesy gate: the paper-1 agent owns GPU priority.  Wait (up to 2 h)
# until used VRAM < 6 GB before starting the light training blocks.
gpu_wait() {
  for _ in $(seq 1 120); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && return 0
    if [ "$used" -lt 6000 ]; then return 0; fi
    echo "[p2] GPU busy (${used} MiB used) -- waiting 60s ($(date))"
    sleep 60
  done
  echo "[p2] GPU wait timed out after 2h -- proceeding (light job)"
}

run_one() {
  local block="$1" out_root="$2" arm="$3" seed="$4"; shift 4
  local dir="$out_root/${arm}_${seed}"
  mkdir -p "$dir"
  if [ -f "$dir/run_config.json" ]; then
    echo "[p2] SKIP  $block/$arm seed=$seed (done)"
    printf "%s\t%s\t%s\tskip\t-\t-\t%s\n" "$block" "$arm" "$seed" "$dir" >> "$STATUS"
    return 0
  fi
  local start; start="$(date +%Y-%m-%dT%H:%M:%S)"
  echo "[p2] START $block/$arm seed=$seed $(date)"
  "$ENVPY" scripts/run_v1_9_resource_alloc.py --config "$CFG" "$@" \
    --seed "$seed" --output-dir "$dir" > "$dir/run.log" 2>&1
  local ec=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$block" "$arm" "$seed" "$ec" "$start" "$(date +%Y-%m-%dT%H:%M:%S)" "$dir" >> "$STATUS"
  echo "[p2] END   $block/$arm seed=$seed exit=$ec $(date)"
  return "$ec"
}

# ---------------------------------------------------------------- block 1
# Flat PPO: RL_COMMON with --flat-ppo instead of --two-timescale-ppo (the
# guards/projection/duals/reward are identical; the actor is ONE categorical).
FLAT_COMMON=(--policy ppo --proposed-semantic-rl --flat-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  "${V8_DUAL[@]}" "${V7_GATES[@]}"
  --scenario "$SCENARIO" --escalation-cost-limit "$DELTA_PEAK"
  --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

gpu_wait
echo "[p2] ==== block 1: flat_ppo_v8 start $(date) ===="
for seed in $SEEDS; do
  run_one flat "$FLAT_OUT" flat_ppo "$seed" "${FLAT_COMMON[@]}"
  ckpt="$FLAT_OUT/flat_ppo_${seed}/ppo_flat_policy.pt"
  if [ -f "$ckpt" ]; then
    run_one flat "$FLAT_OUT" flat_ppo_nom "$seed" \
      --policy ppo --load-ppo-model "$ckpt" --flat-ppo \
      --state-version v2 --hidden-size 128 --device cuda:0 \
      "${V7_GATES[@]}" --escalation-cost-limit "$DELTA_NOM" \
      --scenario "$NOM_SCENARIO" --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
  else
    echo "[p2] MISS  flat_ppo seed=$seed nom-eval (no ckpt)"
    printf "flat\tflat_ppo_nom\t%s\tno-ckpt\t-\t-\t-\n" "$seed" >> "$STATUS"
  fi
done

# ---------------------------------------------------------------- block 2
RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --conflict-cost-weight 0.0 --queue-risk-weight 0.0 --queue-utm-weight 0.0
  --lambda-lr-conflict 0.2 --lambda-max-conflict 8.0 --conflict-cost-limit 0.08
  "${V8_DUAL[@]}" "${V7_GATES[@]}"
  --scenario "$SCENARIO" --escalation-cost-limit "$DELTA_PEAK"
  --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

gpu_wait
echo "[p2] ==== block 2: m3_ablation_v8 start $(date) ===="
for seed in $SEEDS; do
  # (i) quality dual channel OFF (lambda_max_quality=0 pins both quality
  #     channels at 0; argparse takes the LAST occurrence of the flag).
  run_one m3 "$M3_OUT" quality_off "$seed" "${RL_COMMON[@]}" --lambda-max-quality 0.0
done
for seed in $SEEDS; do
  # (ii) warm-START instead of warm-up: dual ascent from episode 0 with
  #      lambda_QC initialized AT its v8 pin value (8.0 = the cap).
  run_one m3 "$M3_OUT" warm_start "$seed" "${RL_COMMON[@]}" \
    --dual-warmup-episodes 0 --lambda-init-quality-critical 8.0
done
for seed in $SEEDS; do
  # (iii) no learned slow mobility head (deterministic mobility defaults).
  run_one m3 "$M3_OUT" no_slow_head "$seed" "${RL_COMMON[@]}" --no-mobility-actor
done

# ---------------------------------------------------------------- block 3
# Eval-only generalization sweep of the v8 checkpoints.  CPU on purpose.
M4_EVAL_BASE=(--policy ppo --two-timescale-ppo
  --state-version v2 --hidden-size 128 --device cpu
  "${V7_GATES[@]}" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

m4_point() {
  # m4_point <axis> <tag> [extra flags...]
  local axis="$1" tag="$2"; shift 2
  for arm in proposed no_lagrangian fixed_penalty; do
    for seed in $SEEDS; do
      local ckpt="$V8ROOT/${arm}_${seed}/ppo_two_timescale_policy.pt"
      if [ ! -f "$ckpt" ]; then
        echo "[p2] MISS m4 $arm seed=$seed (no ckpt)"; continue
      fi
      run_one "m4_${axis}" "$M4_OUT/$axis" "${tag}_${arm}" "$seed" \
        "${M4_EVAL_BASE[@]}" --load-ppo-model "$ckpt" \
        --scenario "$SCENARIO" --escalation-cost-limit "$DELTA_PEAK" "$@"
    done
  done
  for pol in semantic_greedy oracle_escalation_aware; do
    run_one "m4_${axis}" "$M4_OUT/$axis" "${tag}_bl_${pol}" 0 \
      --policy "$pol" --state-version v2 --device cpu \
      "${V7_GATES[@]}" --episodes "$EVAL_EP" --tasks-per-episode "$TPE" \
      --scenario "$SCENARIO" --escalation-cost-limit "$DELTA_PEAK" "$@"
  done
}

echo "[p2] ==== block 3: m4_generalization_v8 start $(date) ===="
for n in 2 3 4 6 8; do m4_point uav "n${n}" --num-uavs "$n"; done
for t in 12 16 20 24 28; do m4_point arrival "t${t}" --tasks-per-episode "$t"; done
# base yaml calibration excess_loss_db = 4.5 (first point = in-distribution anchor)
for x in 4.5 8 12 16 20; do m4_point snr "x${x/./p}" --a2g-excess-loss-db "$x"; done
for f in -120 -119 -118 -117 -116; do m4_point i0 "f${f#-}" --interference-floor-dbm "$f"; done
# Zero-shot triple: unseen profile / x1.5 load / unseen low-SNR blockage profile.
# (tags must be single tokens: the summarizer parses <tag>_<arm>_<seed>.)
m4_point zs soft --scenario utm_conflict_soft
m4_point zs load15 --tasks-per-episode 30
m4_point zs lowsnr --scenario low_snr_blockage

# ---------------------------------------------------------------- block 4+5
echo "[p2] ==== block 4: M2 convergence figures $(date) ===="
"$ENVPY" scripts/plot_m2_convergence_v8.py > outputs/rl/m2_convergence_v8.log 2>&1
echo "[p2] plot exit=$?"

echo "[p2] ==== block 5: paper2 numbers summary $(date) ===="
"$ENVPY" scripts/summarize_paper2_v8.py > outputs/rl/paper2_numbers_v8.log 2>&1
echo "[p2] summary exit=$?"

echo "[p2] ==== phase 1 chain done $(date) ===="
touch outputs/rl/PAPER2_PHASE1_FINISHED
