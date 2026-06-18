# Experiment TODO

Last updated: 2026-06-19 00:15 Asia/Macau
Primary project root: /home/qiankun/phd_research/vqa_semcom

## Completed VQA / SNR-LUT Tasks

Completed 2026-06-18 23:34 Asia/Macau:

- PID 1208659 finished the V1.9 Qwen/SNR run.
- Rebuilt `outputs/lut/v1_9_snr_semantic_quality_lut.csv`.
- Rebuilt `outputs/reports/v1_9_snr_eval_report.md`.
- Rebuilt 200-episode `outputs/sim/v1_9_snr_resource_results.csv` and summary.
- Verified final report contains all configured SNR bins: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB.

```text
/home/qiankun/phd_research/vqa_semcom/outputs/vlm/v1_9_snr_predictions.csv
/home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_snr_semantic_quality_lut.csv
/home/qiankun/phd_research/vqa_semcom/outputs/reports/v1_9_snr_eval_report.md
/home/qiankun/phd_research/vqa_semcom/outputs/sim/v1_9_snr_resource_results.csv
/home/qiankun/phd_research/vqa_semcom/outputs/sim/v1_9_snr_resource_summary.md
```

## Immediate Integration Tasks

1. Done: Reran canonical environment smoke against the refreshed full-bin V1.9 LUT.
2. Done: Reran algorithm smoke and formal resource-allocation baselines/PPO variants against the refreshed full-bin V1.9 LUT.
3. Done: Regenerated merged comparison table that uses both:

```text
outputs/sim/v1_9_snr_resource_results.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/formal_comparison_summary.csv
```

4. Do not overwrite VQA/SNR-LUT artifacts or algorithm outputs owned by another thread.

## Algorithm Thread TODO

Completed 2026-06-18 17:03 Asia/Macau:

- Added a V1.9 LUT-backed Gym-like algorithm adapter in `src/vqa_semcom/rl/v19_resource_env.py`.
- Added centralized service-level PPO in `src/vqa_semcom/rl/v19_ppo.py`.
- Added resource-allocation experiment entrypoint `scripts/run_v1_9_resource_alloc.py`.
- Added interface tests in `tests/test_v1_9_rl_resource_alloc.py`.
- Wrote outputs under `outputs/rl/v1_9_resource_alloc/`.

Completed 2026-06-18 17:29 Asia/Macau:

- Reworked `src/vqa_semcom/rl/v19_resource_env.py` into a thin wrapper over `src/vqa_semcom/sim/multi_uav_env.py`.
- Kept `V19StepRecord`, `candidate_action()`, `candidate_metrics()`, vector observations, and baseline/PPO output format compatible with `scripts/run_v1_9_resource_alloc.py`.
- Added a numpy-free fallback path for `src/vqa_semcom/rl/v19_ppo.py`; if the default Python lacks torch optimizer dependencies such as `mpmath`, PPO smoke records rollout traces without optimizer updates.
- Verified `scripts/run_v1_9_resource_alloc.py --smoke --episodes 1 --train-episodes 2 --output-dir outputs/rl/v1_9_resource_alloc_smoke`.

Completed 2026-06-18 23:43 Asia/Macau:

- Upgraded `src/vqa_semcom/rl/v19_ppo.py` from service-level-only PPO to centralized hybrid PPO.
- Added continuous resource heads for `bandwidth`, `power`, `cpu_share`, and `gpu_share` while preserving the existing script-facing PPO API.
- Added constrained/TCH-style Lagrangian PPO costs for `quality_violation` and `deadline_violation`, with `lambda_quality` and `lambda_deadline` recorded per training episode.
- Updated `scripts/run_v1_9_resource_alloc.py` with hybrid/constrained PPO flags, smoke-safe default output directory `outputs/rl/v1_9_hybrid_tch_ppo_smoke`, `ppo_hybrid_policy.pt`, and `ppo_lambda_trace.csv`.
- Updated V1.9 RL tests to verify hybrid action fields and lambda trace output.
- Verified:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 8 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py --smoke --policy ppo --episodes 1 --train-episodes 2 --tasks-per-episode 4 --output-dir outputs/rl/v1_9_hybrid_tch_ppo_smoke
```

Completed 2026-06-19 00:04 Asia/Macau:

- Added formal multi-seed comparison runner:

```text
scripts/run_v1_9_formal_rl_compare.py
```

- Ran 3 seeds (`0,1,2`) with 200 evaluation episodes per seed and 120 PPO training episodes per seed.
- Compared heuristic baselines, service-only PPO, hybrid PPO without Lagrangian, and hybrid TCH/Lagrangian PPO.
- Preserved per-seed `ppo_training_trace.csv` and `ppo_lambda_trace.csv`, plus aggregated `all_training_trace.csv` and `all_lambda_trace.csv`.
- Generated merged report aligned with `outputs/sim/v1_9_snr_resource_results.csv`.

Command:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_formal_rl_compare.py --output-dir outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618 --seeds 0,1,2 --episodes 200 --train-episodes 120
```

Formal artifacts:

```text
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/all_seed_results.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/formal_comparison_summary.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/all_training_trace.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/all_lambda_trace.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/merged_with_sim_results.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/merged_comparison_report.md
```

Key observation:

- All current PPO variants collapse to service level 0/cache in the calibrated canonical environment.
- The best formal RL success rates are currently still heuristic/oracle-style baselines, not PPO.

Useful commands:

```bash
cd /home/qiankun/phd_research/vqa_semcom

# Fast smoke: hybrid constrained PPO only
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py --smoke --policy ppo --episodes 1 --train-episodes 2 --tasks-per-episode 4 --output-dir outputs/rl/v1_9_hybrid_tch_ppo_smoke

# Fast smoke: baseline + tiny hybrid constrained PPO
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py --smoke --episodes 1 --train-episodes 2 --output-dir outputs/rl/v1_9_resource_alloc_smoke

# Baseline-only comparison
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py --episodes 20 --policy all --output-dir outputs/rl/v1_9_resource_alloc_baselines

# Centralized PPO baseline plus heuristic baselines
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py --episodes 20 --train-ppo --train-episodes 120 --output-dir outputs/rl/v1_9_resource_alloc

# Formal multi-seed comparison
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_formal_rl_compare.py --output-dir outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618 --seeds 0,1,2 --episodes 200 --train-episodes 120

# Regression tests for V1.9 LUT/resource/RL interfaces
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'

# Canonical env + wrapper smoke under default python3
python3 -m unittest tests/test_multi_uav_env.py tests/test_v1_9_snr_lut.py tests/test_v1_9_rl_resource_alloc.py
python3 scripts/run_v1_9_resource_alloc.py --smoke --episodes 1 --train-episodes 2 --output-dir outputs/rl/v1_9_resource_alloc_smoke
```

Expected artifacts:

```text
outputs/rl/v1_9_resource_alloc/v1_9_resource_alloc_results.csv
outputs/rl/v1_9_resource_alloc/v1_9_resource_alloc_rollout.csv
outputs/rl/v1_9_resource_alloc/v1_9_resource_alloc_summary.md
outputs/rl/v1_9_resource_alloc/ppo_training_trace.csv
outputs/rl/v1_9_resource_alloc/ppo_service_policy.pt
outputs/rl/v1_9_resource_alloc/ppo_hybrid_policy.pt
outputs/rl/v1_9_resource_alloc/ppo_lambda_trace.csv
```

Next algorithm steps:

1. Fix PPO collapse-to-cache behavior observed in `outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/formal_comparison_summary.csv`.
2. Tune reward scaling and constraint targets: success bonus, quality/deadline penalty, payload/energy delay weights, lambda learning rate, and cost limits.
3. Increase exploration/regularization: entropy schedule, service-level prior or curriculum, longer training, and/or supervised warm-start from `oracle_best_feasible_evidence`.
4. Improve continuous action handling: resource normalization/projection and service-dependent resource floors so hybrid PPO does not waste resource heads on cache-only actions.
5. Add paper ablations after PPO is competitive: service-only PPO, hybrid PPO without Lagrangian, hybrid TCH-PPO, oracle warm-start, and scenario-specific fixed presets.

Algorithm outputs must stay under one of:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/rl/
/home/qiankun/phd_research/vqa_semcom/outputs/hppo/
/home/qiankun/phd_research/vqa_semcom/runs/
```

Do not overwrite V1.9 VLM/LUT/report files.

## Environment Thread TODO

Completed 2026-06-18 17:29 Asia/Macau:

1. Read docs/current_status.md, docs/interfaces.md, and docs/thread_contract.md before editing.
2. Promoted `src/vqa_semcom/sim/multi_uav_env.py` to the canonical reset/step MDP environment.
3. Integrated V0 modeling features: `Area4D`, airspace overlap/conflict, multi-UAV states, LoS/NLoS A2G, fading, SINR/rate, multi-UAV interference, semantic cache, edge model cache, GPU memory, delay and energy decomposition.
4. Preserved required observation fields: task_type, risk_level, view_quality_bin, freshness_bin, sensed_snr_db, snr_bin, uav_state, edge_load, cache_state.
5. Preserved required info fields: answer_accuracy_est, delay_s, energy_j, payload_kb, quality_violation, deadline_violation, snr_bin, service_level.
6. Added regression tests for Area4D/cache-only conflict policy, link monotonicity/interference, semantic cache/model cache effects, UAV mobility, and service-level 3 disabled-by-default behavior.
7. Wrote environment smoke outputs under:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/env/
```

Latest environment validation:

```bash
cd /home/qiankun/phd_research/vqa_semcom
python3 -m unittest tests/test_multi_uav_env.py tests/test_v1_9_snr_lut.py tests/test_v1_9_rl_resource_alloc.py
python3 scripts/run_multi_uav_env_smoke.py --config configs/v1_9_snr_lut.yaml --steps 12 --seed 21
```

Latest environment smoke artifacts:

```text
outputs/env/env_smoke_20260618_173144/trace.csv
outputs/env/env_smoke_20260618_173144/summary.md
```

Completed 2026-06-18 23:45 Asia/Macau:

1. Added fixed scenario presets in `src/vqa_semcom/sim/multi_uav_env.py`: `conflict-heavy`, `interference-heavy`, `cache-heavy`, and `mobility-stress`.
2. Added centralized calibration in `configs/v1_9_snr_lut.yaml` under `multi_uav_env.calibration` for propulsion energy, hover power, sensing/processing time, A2G path-loss constants, edge queue/model-cache delay, GPU memory, and model memory.
3. Updated `scripts/run_multi_uav_env_smoke.py --scenario {scenario}` so each fixed scenario writes trace/summary under `outputs/env/`.
4. Added scenario/calibration tests in `tests/test_multi_uav_env.py`; service level 3 remains disabled by default.
5. Passed `/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests` with 47 tests OK.

Latest fixed-scenario smoke artifacts:

```text
outputs/env/env_smoke_conflict_heavy_20260618_234452/trace.csv
outputs/env/env_smoke_conflict_heavy_20260618_234452/summary.md
outputs/env/env_smoke_interference_heavy_20260618_234452/trace.csv
outputs/env/env_smoke_interference_heavy_20260618_234452/summary.md
outputs/env/env_smoke_cache_heavy_20260618_234452/trace.csv
outputs/env/env_smoke_cache_heavy_20260618_234452/summary.md
outputs/env/env_smoke_mobility_stress_20260618_234452/trace.csv
outputs/env/env_smoke_mobility_stress_20260618_234452/summary.md
```

Completed 2026-06-19 00:15 Asia/Macau:

1. Promoted the environment-thread scope to a UAV-driven semantic communication network simulator/benchmark while preserving the stable reset/step API.
2. Added architecture and formal problem docs:

```text
docs/semantic_network_architecture.md
docs/formal_problem_definition.md
```

3. Extended `src/vqa_semcom/sim/multi_uav_env.py` with:
   - semantic network layer metadata.
   - semantic service routing API.
   - semantic utility API.
   - graph observation schema/export.
   - formal train/test scenario presets.
   - scalability presets for UAV count, task arrival, and edge load.
4. Added benchmark script:

```text
scripts/run_semantic_network_benchmark.py
```

5. Added tests:

```text
tests/test_semantic_network_env.py
```

6. Generated environment-owned benchmark outputs:

```text
outputs/env/formal_scenario_specs.md
outputs/env/scenario_smoke_20260619_001459.csv
```

7. Passed `/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests` with 53 tests OK.
8. Confirmed service level 3 ROI/crop remains disabled; benchmark service levels are 0/1/2 only.

Next environment steps:

1. Use `outputs/env/scenario_smoke_20260619_001459.csv` to sanity-check formal train/test split behavior before algorithm training.
2. Run larger benchmark sweeps with `scripts/run_semantic_network_benchmark.py --include-scalability --scalability-mode full` when a full scalability table is needed.
3. Add a full Gymnasium/DI-engine graph adapter after the graph schema is consumed by a GNN/HPPO design.
4. Calibrate constants against a concrete UAV platform if platform specs become available; current values are lightweight, explainable defaults.
5. Extend service level 3 ROI only after the V1.9/V2 LUT explicitly contains ROI/crop rows.

## Integration TODO

1. Done: Algorithm thread consumes canonical environment obs/action contract through `rl.v19_resource_env`.
2. Done: Environment thread calls the V1.9 LUT to estimate answer accuracy/payload from `snr_bin`.
3. Done: VQA thread refreshed the full-bin V1.9 LUT/report/resource summary after PID 1208659 finished.
4. Done: Reran algorithm hybrid TCH-PPO smoke after the rebuilt full-bin V1.9 LUT under `outputs/rl/v1_9_hybrid_tch_ppo_smoke`.
5. Done: Reran canonical environment smoke for all fixed scenarios under `outputs/env/env_smoke_*_20260618_234452`.
6. Done: Reran formal multi-seed algorithm comparison under `outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618`.
7. Done: Added semantic-network benchmark specs and smoke output under `outputs/env/`.
8. Pending: Improve PPO/TCH-PPO so it does not collapse to service level 0/cache in the calibrated canonical environment.
9. Ongoing: All threads update docs/current_status.md and docs/experiment_todo.md before ending work.

## Output Naming Convention

Use explicit run names:

```text
outputs/rl/hppo_v1_9_lut_seed{seed}_YYYYMMDD_HHMMSS/
outputs/env/env_smoke_YYYYMMDD_HHMMSS/
outputs/sim/v1_9_snr_resource_results.csv
outputs/reports/v1_9_snr_eval_report.md
```

Never write generic temporary outputs into the root directory.
