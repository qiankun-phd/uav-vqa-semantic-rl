#!/usr/bin/env bash
# Revised BUBBLES A/B chain (runs AFTER the 2026-07 P0/P1 RL-correctness fixes).
#
# Arms (docs_spec/V19_Design_Review_2026-07.md section 4):
#   C   : non-learning baselines on bubbles (semantic_greedy, always_cache) -- seconds
#   A2  : bubbles profile, fixed proposed two-timescale PPO, seeds 0/1/2
#   B1  : bubbles, slow mobility head disabled (--no-mobility-actor)
#   B2  : bubbles, conflict dual channel disabled (--lambda-max-conflict 0)
#   A1  : legacy profile (scenario_profile null), regression reference
#
# Success criteria (evaluated by scripts/summarize_ab_bubbles_v2.py at chain end):
#   A2 conflict rate <= 0.10; A2 semantic success drop vs A1 <= 0.03;
#   A2 cache ratio <= 0.30; B1 conflict - A2 conflict >= 0.05.
set -u

ENVPY="$HOME/.conda/envs/uav_semcom/bin/python"
ROOT="$HOME/phd_research/vqa_main"
cd "$ROOT" || { echo "cannot cd $ROOT"; exit 2; }

SCENARIO="utm_conflict"
TRAIN_EP="${TRAIN_EP:-500}"
EVAL_EP="${EVAL_EP:-50}"
TPE="${TPE:-20}"
SEEDS="${SEEDS:-0 1 2}"
OUT="outputs/rl/ab_bubbles_v2"
BUBBLES_CFG="configs/v1_9_bubbles.yaml"
LEGACY_CFG="configs/v1_9_snr_lut.yaml"
mkdir -p "$OUT"
STATUS="$OUT/chain_status.tsv"
if [ ! -f "$STATUS" ]; then
  printf "arm\tseed\tconfig\texit\tstart\tend\tdir\n" > "$STATUS"
fi

RL_COMMON=(--policy ppo --proposed-semantic-rl --two-timescale-ppo
  --deadline-aware-evidence-guard --payload-delay-aware-projection --token-fast-resource-projection
  --state-version v2 --hidden-size 128 --device cuda:0
  --scenario "$SCENARIO" --train-episodes "$TRAIN_EP" --episodes "$EVAL_EP" --tasks-per-episode "$TPE")

run_one() {
  local arm="$1" cfg="$2" seed="$3"; shift 3
  local dir="$OUT/${arm}_${seed}"
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

echo "[chain] ==== BUBBLES A/B v2 chain start $(date) | train_ep=$TRAIN_EP eval_ep=$EVAL_EP tpe=$TPE seeds=$SEEDS ===="

# --- C: non-learning baselines on bubbles (fast, run first) ---
run_one Cgreedy "$BUBBLES_CFG" 0 --policy semantic_greedy --scenario "$SCENARIO" \
  --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"
run_one Ccache "$BUBBLES_CFG" 0 --policy always_cache --scenario "$SCENARIO" \
  --state-version v2 --episodes "$EVAL_EP" --tasks-per-episode "$TPE"

# --- A2: bubbles, fixed proposed two-timescale ---
for seed in $SEEDS; do
  run_one A2 "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}"
done

# --- B1: bubbles, no slow mobility head ---
for seed in $SEEDS; do
  run_one B1 "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}" --no-mobility-actor
done

# --- B2: bubbles, conflict dual channel off ---
for seed in $SEEDS; do
  run_one B2 "$BUBBLES_CFG" "$seed" "${RL_COMMON[@]}" --lambda-max-conflict 0.0
done

# --- A1: legacy profile regression reference ---
for seed in $SEEDS; do
  run_one A1 "$LEGACY_CFG" "$seed" "${RL_COMMON[@]}"
done

echo "[chain] ==== chain done $(date) ===="
"$ENVPY" scripts/summarize_ab_bubbles_v2.py --root "$OUT" || echo "[chain] summary script failed"
echo "[chain] status table:"
cat "$STATUS"
