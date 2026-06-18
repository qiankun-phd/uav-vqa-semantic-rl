# Current Status

Last updated: 2026-06-19 00:28 Asia/Macau
Remote host: qiankun@172.27.57.160

## Code Locations

Primary VQA/SNR-LUT project:

```text
/home/qiankun/phd_research/vqa_semcom
```

Legacy V0 / algorithm reference project:

```text
/home/qiankun/HPPO-VQA/vqa_semcom_v0
```

Local workspace mirror mentioned by user:

```text
/Users/zhangqiankun/Documents/mpu/HPPO-VQA
```

## Active Processes Observed

V1.9 SNR main experiment completed:

```text
PID 1208659
cwd /home/qiankun/phd_research/vqa_semcom
cmd /home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_detector_eval.py --config configs/v1_9_snr_lut.yaml --limit-images 500 --evaluator qwen --service-levels 0,1,2 --snr-bins=-5,0,5,10,15,20 --max-new-tokens 12 --resume
log /home/qiankun/phd_research/vqa_semcom/logs/v1_9_snr_main_500.log
completion check 2026-06-18 23:32 Asia/Macau: process exited; log reports selected_tasks=2973 and prediction_rows=160542.
postprocess check 2026-06-18 23:34 Asia/Macau: LUT/report/resource simulation rebuilt successfully.
```

Legacy TCH-PPO jobs are also active, likely from /home/qiankun/HPPO-VQA/vqa_semcom_v0:

```text
PID 1222872 python scripts/run_tch_ppo.py --config configs/v0.yaml --scenario literature_demo --max-train-iter 2000 --seed 0 --output-dir outputs/rl/tch_ppo_seed0
PID 1222873 python scripts/run_tch_ppo.py --config configs/v0.yaml --scenario literature_demo --max-train-iter 2000 --seed 1 --output-dir outputs/rl/tch_ppo_seed1
PID 1222874 python scripts/run_tch_ppo.py --config configs/v0.yaml --scenario literature_demo --max-train-iter 2000 --seed 2 --output-dir outputs/rl/tch_ppo_seed2
```

Do not kill these processes unless explicitly requested.

No V1.9 algorithm-thread process is currently active. The V1.9 PPO/resource-allocation run below completed in the foreground.

## V1.9 SNR-LUT State

Important config:

```text
/home/qiankun/phd_research/vqa_semcom/configs/v1_9_snr_lut.yaml
```

Important source:

```text
/home/qiankun/phd_research/vqa_semcom/src/vqa_semcom/snr.py
```

Important outputs:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/vlm/v1_9_snr_predictions.csv
/home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_snr_semantic_quality_lut.csv
/home/qiankun/phd_research/vqa_semcom/outputs/reports/v1_9_snr_eval_report.md
/home/qiankun/phd_research/vqa_semcom/outputs/sim/v1_9_snr_resource_results.csv
/home/qiankun/phd_research/vqa_semcom/outputs/sim/v1_9_snr_resource_summary.md
```

Latest available V1.9 report summary:

- completed report prediction rows: 160542
- prediction CSV line count: 909739
- LUT rows: 648
- SNR bins in report: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB
- config SNR bins: -5, 0, 5, 10, 15, 20
- real VLM present: yes
- cache accuracy spread across SNR: 0.000000 expected for SNR-insensitive cache baseline

Current completed V1.9 artifacts:

```text
outputs/vlm/v1_9_snr_predictions.csv
outputs/lut/v1_9_snr_semantic_quality_lut.csv
outputs/reports/v1_9_snr_eval_report.md
outputs/reports/v1_9_snr_accuracy_by_snr.csv
outputs/reports/v1_9_snr_payload_by_snr.csv
outputs/sim/v1_9_snr_resource_results.csv
outputs/sim/v1_9_snr_resource_summary.md
```

## V1.9 VQA-grounded Semantic Utility Model State

Completed 2026-06-19 00:28 Asia/Macau:

- Added a task-conditioned semantic utility layer on top of the V1.9 measured Qwen/VQA predictions.
- The original prediction CSV was not modified.
- The old raw SNR quality LUT remains available as traceable measurement input.
- The new utility model exposes answer accuracy, conservative LCB, payload, uncertainty, and sample count for each task/service/SNR/view/freshness/risk cell.
- SNR monotonic sanity checking is applied to reduce RL instability from sparse/noisy measured cells.
- Cache service cells are forced to be SNR-invariant because cache answer reuse does not transmit visual evidence.

New source and tests:

```text
src/vqa_semcom/semantic/utility.py
tests/test_semantic_utility.py
```

New outputs:

```text
outputs/lut/v1_9_semantic_utility_with_ci.csv
outputs/reports/semantic_utility_calibration.md
```

Latest semantic utility calibration summary:

- utility cells: 648
- CSV line count including header: 649
- SNR bins: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB
- sparse cells: 108
- SNR monotonic adjusted cells: 48
- cache SNR-invariant cells: 216

New control-facing API:

```python
U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
# -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

For algorithm/environment integration, prefer `accuracy_lcb` for conservative QoS checks and keep `uncertainty`/`sample_count` visible in `info`.

Resource simulation headline:

| policy | success | accuracy | delay | energy | payload KB | payload reduction |
|---|---:|---:|---:|---:|---:|---:|
| always_cache | 0.286 | 0.560 | 0.050 | 0.200 | 0.000 | 1.000 |
| always_light | 0.204 | 0.599 | 0.355 | 1.000 | 0.850 | 0.991 |
| always_image | 0.560 | 0.588 | 1.546 | 2.500 | 93.044 | 0.000 |
| greedy_min_sufficient_evidence | 0.711 | 0.707 | 0.896 | 1.562 | 48.024 | 0.484 |
| no_cache_greedy | 0.567 | 0.583 | 1.305 | 2.195 | 73.879 | 0.206 |
| no_semantic_tokens_greedy | 0.704 | 0.712 | 1.119 | 1.843 | 65.779 | 0.293 |
| oracle_best_feasible_evidence | 0.711 | 0.721 | 0.664 | 1.219 | 29.122 | 0.687 |

## V1.9 Algorithm Thread State

Implemented V1.9 LUT-backed centralized resource-allocation adapter and PPO baseline. As of 2026-06-18 17:29 Asia/Macau, `src/vqa_semcom/rl/v19_resource_env.py` is a thin compatibility wrapper over the canonical multi-UAV simulator instead of a separate dynamics implementation:

```text
/home/qiankun/phd_research/vqa_semcom/src/vqa_semcom/rl/v19_resource_env.py
/home/qiankun/phd_research/vqa_semcom/src/vqa_semcom/rl/v19_ppo.py
/home/qiankun/phd_research/vqa_semcom/scripts/run_v1_9_resource_alloc.py
/home/qiankun/phd_research/vqa_semcom/tests/test_v1_9_rl_resource_alloc.py
```

The environment follows docs/interfaces.md:

- observation includes task_type/question_type, risk_level, view_quality_bin, freshness_bin, sensed_snr_db, snr_bin, uav_state, edge_load, cache_state, deadline_s, task_id, feasible_uavs, and numeric vector.
- action accepts service_level, bandwidth, power, cpu_share, gpu_share, uav_assignment, waypoint.
- info exposes answer_accuracy_est, success, delay_s, energy_j, payload_kb, quality_violation, deadline_violation, snr_bin, service_level.
- answer quality oracle is `/home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_snr_semantic_quality_lut.csv`.
- V1.9 default enabled service levels are 0/1/2; service level 3 remains disabled unless a future config/LUT explicitly enables it.

Completed algorithm update 2026-06-18 23:43 Asia/Macau:

- Upgraded `src/vqa_semcom/rl/v19_ppo.py` from service-level-only PPO to centralized hybrid PPO.
- The actor now uses one discrete service-level head plus four continuous resource heads for bandwidth, power, CPU share, and GPU share.
- Added constrained/TCH-style Lagrangian training with explicit `quality_violation` and `deadline_violation` costs, dual variables `lambda_quality` and `lambda_deadline`, and raw/shaped reward traces.
- Kept the public `train_ppo()`, `PPOTrainConfig`, `PPOServicePolicy`, `save_ppo_model()`, and `load_ppo_policy()` API names for script/test compatibility.
- Updated `scripts/run_v1_9_resource_alloc.py` so smoke runs default to `outputs/rl/v1_9_hybrid_tch_ppo_smoke` and write `ppo_lambda_trace.csv`.
- Updated `tests/test_v1_9_rl_resource_alloc.py` to check hybrid action fields and lambda trace output.

Latest hybrid TCH-PPO smoke:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py --smoke --policy ppo --episodes 1 --train-episodes 2 --tasks-per-episode 4 --output-dir outputs/rl/v1_9_hybrid_tch_ppo_smoke
```

Smoke outputs:

```text
outputs/rl/v1_9_hybrid_tch_ppo_smoke/ppo_hybrid_policy.pt
outputs/rl/v1_9_hybrid_tch_ppo_smoke/v1_9_resource_alloc_results.csv
outputs/rl/v1_9_hybrid_tch_ppo_smoke/v1_9_resource_alloc_rollout.csv
outputs/rl/v1_9_hybrid_tch_ppo_smoke/v1_9_resource_alloc_summary.md
outputs/rl/v1_9_hybrid_tch_ppo_smoke/ppo_training_trace.csv
outputs/rl/v1_9_hybrid_tch_ppo_smoke/ppo_lambda_trace.csv
```

Smoke headline:

| policy | success | accuracy | delay | energy | payload KB | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.534 | 6.838 | 965.790 | 0.788 | 1.000 | 1.000 |

Lambda trace smoke check:

| episode | quality cost | deadline cost | lambda quality | lambda deadline |
|---:|---:|---:|---:|---:|
| 0 | 1.000 | 0.500 | 0.050 | 0.025 |
| 1 | 0.500 | 0.500 | 0.075 | 0.050 |

This was a short smoke only; it should not be used as final policy performance.

Completed formal algorithm comparison 2026-06-19 00:04 Asia/Macau:

- Added formal multi-seed runner:

```text
/home/qiankun/phd_research/vqa_semcom/scripts/run_v1_9_formal_rl_compare.py
```

- Ran 3 seeds (`0,1,2`), 200 evaluation episodes per seed, and 120 PPO training episodes per seed.
- Compared heuristic baselines, service-only PPO, hybrid PPO without Lagrangian, and hybrid TCH/Lagrangian PPO.
- Aligned the merged report with `outputs/sim/v1_9_snr_resource_results.csv`.

Command:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_formal_rl_compare.py --output-dir outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618 --seeds 0,1,2 --episodes 200 --train-episodes 120
```

Formal outputs:

```text
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/all_seed_results.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/formal_comparison_summary.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/all_training_trace.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/all_lambda_trace.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/merged_with_sim_results.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/merged_comparison_report.md
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/{baselines,service_only_ppo,hybrid_ppo,hybrid_tch_ppo}_seed*/
```

Formal RL headline, mean +/- std over seeds:

| method | success | accuracy | delay | energy | payload KB | quality violation | deadline violation | s0 | s1 | s2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.017 +/- 0.000 | 0.411 +/- 0.000 | 0.091 +/- 0.000 | 1.610 +/- 0.000 | 0.000 +/- 0.000 | 0.983 +/- 0.000 | 0.000 +/- 0.000 | 1.000 | 0.000 | 0.000 |
| always_light | 0.019 +/- 0.000 | 0.483 +/- 0.000 | 2.407 +/- 0.009 | 295.427 +/- 1.338 | 1.146 +/- 0.000 | 0.858 +/- 0.001 | 0.258 +/- 0.001 | 0.000 | 1.000 | 0.000 |
| always_image | 0.000 +/- 0.000 | 0.439 +/- 0.000 | 4.474 +/- 0.001 | 359.181 +/- 0.088 | 184.117 +/- 0.001 | 0.553 +/- 0.000 | 1.000 +/- 0.000 | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.050 +/- 0.000 | 0.279 +/- 0.001 | 5.010 +/- 0.007 | 510.263 +/- 1.183 | 132.838 +/- 0.164 | 0.712 +/- 0.001 | 0.920 +/- 0.000 | 0.018 | 0.262 | 0.721 |
| no_cache_greedy | 0.019 +/- 0.000 | 0.149 +/- 0.001 | 4.812 +/- 0.006 | 445.137 +/- 1.182 | 158.115 +/- 0.164 | 0.847 +/- 0.001 | 0.969 +/- 0.000 | 0.000 | 0.142 | 0.858 |
| no_semantic_tokens_greedy | 0.018 +/- 0.000 | 0.723 +/- 0.000 | 4.529 +/- 0.001 | 372.476 +/- 0.088 | 180.670 +/- 0.001 | 0.266 +/- 0.000 | 0.982 +/- 0.000 | 0.017 | 0.000 | 0.983 |
| oracle_best_feasible_evidence | 0.049 +/- 0.000 | 0.819 +/- 0.000 | 2.524 +/- 0.008 | 285.849 +/- 0.947 | 47.192 +/- 0.135 | 0.698 +/- 0.001 | 0.269 +/- 0.001 | 0.692 | 0.051 | 0.257 |
| service_only_ppo | 0.017 +/- 0.000 | 0.411 +/- 0.000 | 0.091 +/- 0.000 | 1.610 +/- 0.000 | 0.000 +/- 0.000 | 0.983 +/- 0.000 | 0.000 +/- 0.000 | 1.000 | 0.000 | 0.000 |
| hybrid_ppo | 0.017 +/- 0.000 | 0.411 +/- 0.000 | 0.369 +/- 0.171 | 1.626 +/- 0.028 | 0.000 +/- 0.000 | 0.983 +/- 0.000 | 0.000 +/- 0.000 | 1.000 | 0.000 | 0.000 |
| hybrid_tch_ppo | 0.017 +/- 0.000 | 0.411 +/- 0.000 | 0.273 +/- 0.226 | 1.619 +/- 0.015 | 0.000 +/- 0.000 | 0.983 +/- 0.000 | 0.000 +/- 0.000 | 1.000 | 0.000 | 0.000 |

Interpretation: the current PPO variants all collapse to service level 0/cache under the calibrated canonical multi-UAV environment. The formal run is useful as a reproducible baseline, but the PPO/TCH design needs reward/constraint and exploration tuning before it is a competitive result.

Completed algorithm run:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py --episodes 20 --train-ppo --train-episodes 120 --output-dir outputs/rl/v1_9_resource_alloc
```

Algorithm outputs:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/rl/v1_9_resource_alloc/v1_9_resource_alloc_results.csv
/home/qiankun/phd_research/vqa_semcom/outputs/rl/v1_9_resource_alloc/v1_9_resource_alloc_rollout.csv
/home/qiankun/phd_research/vqa_semcom/outputs/rl/v1_9_resource_alloc/v1_9_resource_alloc_summary.md
/home/qiankun/phd_research/vqa_semcom/outputs/rl/v1_9_resource_alloc/ppo_training_trace.csv
/home/qiankun/phd_research/vqa_semcom/outputs/rl/v1_9_resource_alloc/ppo_service_policy.pt
```

Latest algorithm headline from the completed run:

| policy | success | accuracy | delay | energy | payload KB | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.244 | 0.551 | 0.050 | 0.200 | 0.000 | 0.756 | 0.000 |
| always_light | 0.115 | 0.519 | 0.355 | 0.843 | 0.823 | 0.885 | 0.000 |
| always_image | 0.469 | 0.601 | 1.691 | 2.558 | 186.426 | 0.531 | 0.000 |
| greedy_min_sufficient_evidence | 0.720 | 0.733 | 1.156 | 1.857 | 106.379 | 0.280 | 0.000 |
| no_cache_greedy | 0.519 | 0.610 | 1.527 | 2.378 | 144.970 | 0.481 | 0.000 |
| no_semantic_tokens_greedy | 0.686 | 0.727 | 1.269 | 1.982 | 134.161 | 0.314 | 0.000 |
| oracle_best_feasible_evidence | 0.720 | 0.761 | 0.935 | 1.577 | 79.720 | 0.280 | 0.000 |
| ppo | 0.469 | 0.601 | 1.691 | 2.558 | 186.426 | 0.531 | 0.000 |

Validation completed:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 8 tests OK
```

Latest wrapper smoke after canonical env integration:

```bash
cd /home/qiankun/phd_research/vqa_semcom
python3 -m unittest tests/test_multi_uav_env.py tests/test_v1_9_snr_lut.py tests/test_v1_9_rl_resource_alloc.py
# Ran 15 tests OK

python3 scripts/run_v1_9_resource_alloc.py --smoke --episodes 1 --train-episodes 2 --output-dir outputs/rl/v1_9_resource_alloc_smoke
```

The default `python3` environment currently lacks `numpy` and `mpmath`; the PPO smoke falls back to rollout trace collection if the torch optimizer dependencies are incomplete. Use `/home/qiankun/.conda/envs/uav_semcom/bin/python` for full PPO training.

## Canonical Multi-UAV Environment State

Canonical environment dynamics now live in:

```text
/home/qiankun/phd_research/vqa_semcom/src/vqa_semcom/sim/multi_uav_env.py
```

The environment integrates the V0 modeling work into V1.9:

- dynamic multi-UAV task queue with task type, risk, view quality, freshness/cache age, deadline, epsilon, priority, generation time, and `Area4D` operational area.
- UAV state with position, altitude, battery, speed, camera state, assigned task, travel distance, and utilization.
- LoS/NLoS A2G link budget, fading, SNR/SINR, rate, and multi-UAV interference approximation.
- delay decomposition: fly, sense, tx, queue, infer, load.
- energy decomposition: flight, hover/sense, tx, compute.
- semantic cache store with freshness/priority-aware replacement.
- edge model cache with GPU memory capacity and model load hit/miss delay.
- airspace conflict penalty only for `service_level > 0` observe/revisit/waypoint operational intents; cache-only reuse does not create conflict.
- V1.9 LUT remains the semantic quality source: `LUT[task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level]`.

Environment outputs stay under:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/env/
```

Latest environment smoke:

```bash
cd /home/qiankun/phd_research/vqa_semcom
python3 scripts/run_multi_uav_env_smoke.py --config configs/v1_9_snr_lut.yaml --steps 12 --seed 21
```

Output:

```text
outputs/env/env_smoke_20260618_173144/trace.csv
outputs/env/env_smoke_20260618_173144/summary.md
```

The trace includes conflict, sensed SNR, SINR, rate, distance/elevation, LoS probability, path loss, interference, fading, cache hit, GPU memory, battery, and delay/energy decomposition fields. A mis-synced root-file backup from this integration was moved to:

```text
outputs/env/codex_misrsync_20260618/
```

Scenario/calibration update completed 2026-06-18 23:45 Asia/Macau:

- Added fixed scenario presets in `src/vqa_semcom/sim/multi_uav_env.py`: `conflict-heavy`, `interference-heavy`, `cache-heavy`, and `mobility-stress`.
- Added centralized calibration in `configs/v1_9_snr_lut.yaml` under `multi_uav_env.calibration` for propulsion energy, hover power, sensing/processing time, A2G path-loss constants, edge queue/model-cache delay, GPU memory, and model memory.
- Updated `scripts/run_multi_uav_env_smoke.py` with `--scenario`; all scenario smoke outputs still live under `outputs/env/`.
- Confirmed service levels remain `[0, 1, 2]`; service level 3 ROI/crop remains disabled until the VQA thread provides ROI/crop LUT rows.
- Passed `/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests` with 47 tests OK.

Latest fixed-scenario environment smokes:

| scenario | output dir | conflict rate | avg SINR | avg delay | avg energy |
|---|---|---:|---:|---:|---:|
| conflict-heavy | `outputs/env/env_smoke_conflict_heavy_20260618_234452` | 1.000 | 12.435 | 2.994 | 260.746 |
| interference-heavy | `outputs/env/env_smoke_interference_heavy_20260618_234452` | 0.000 | 16.071 | 8.621 | 997.963 |
| cache-heavy | `outputs/env/env_smoke_cache_heavy_20260618_234452` | 0.000 | 43.406 | 1.923 | 231.230 |
| mobility-stress | `outputs/env/env_smoke_mobility_stress_20260618_234452` | 0.000 | 21.248 | 50.225 | 6203.722 |

## Current Blockers / Watch Items

- UTM/InterUSS-style realistic flight mapping is being added to `sim.multi_uav_env` as an environment-thread feature; it does not introduce an InterUSS runtime dependency and keeps service level 3 ROI disabled.
- V1.9 full-SNR prediction/LUT/report/resource simulation refresh is complete as of 2026-06-18 23:34 Asia/Macau.
- The final report includes all configured SNR bins: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB.
- Algorithm hybrid TCH-PPO smoke has been rerun against the refreshed full-bin LUT under `outputs/rl/v1_9_hybrid_tch_ppo_smoke`.
- Formal multi-seed resource-allocation comparison has been run under `outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618`.
- Current PPO variants collapse to cache in the calibrated canonical environment; next algorithm work should tune reward shaping, lambda targets, entropy/exploration, and possibly constrained action projection.
- The primary VQA/SNR-LUT directory is not currently a Git worktree. Coordination must rely on docs and path discipline unless a repository is initialized later.
- Algorithm and environment threads must not change the LUT schema without updating docs/interfaces.md.
- `sim.multi_uav_env` is now the single source of truth for resource-allocation environment dynamics; keep `rl.v19_resource_env` as an algorithm-facing wrapper.

## How To Refresh Status

```bash
ssh qiankun@172.27.57.160 'pgrep -af "run_v1_detector_eval|run_tch_ppo|hppo|ppo"'
ssh qiankun@172.27.57.160 'tail -n 80 /home/qiankun/phd_research/vqa_semcom/logs/v1_9_snr_main_500.log'
ssh qiankun@172.27.57.160 'ls -lh /home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_snr_semantic_quality_lut.csv /home/qiankun/phd_research/vqa_semcom/outputs/reports/v1_9_snr_eval_report.md'
```

## Integration Verification 2026-06-18 17:35 Asia/Macau

- Checked VQA/SNR-LUT, algorithm wrapper/PPO, and canonical multi-UAV environment integration.
- Passed: /home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests  # 42 tests OK.
- Passed: py_compile for src/vqa_semcom/sim/multi_uav_env.py, src/vqa_semcom/rl/v19_resource_env.py, src/vqa_semcom/rl/v19_ppo.py, scripts/run_v1_9_resource_alloc.py, scripts/run_multi_uav_env_smoke.py.
- Passed: environment smoke wrote outputs/env/env_smoke_20260618_173423/.
- Passed: algorithm smoke wrote outputs/rl/v1_9_resource_alloc_verify_smoke/.
- Cleaned stale root-level mis-sync files by moving multi_uav_env.py, v19_resource_env.py, and test_multi_uav_env.py to outputs/env/codex_misrsync_20260618/root_stale_20260618_1734/. Canonical files remain under src/ and tests/.
- Completed later: PID 1208659 exited and the full-bin V1.9 LUT/report/resource simulation were refreshed at 2026-06-18 23:34 Asia/Macau.

## UTM Realistic Flight Mapping 2026-06-19 Asia/Shanghai

Environment thread added an InterUSS/ASTM-inspired UTM coordination layer to the canonical simulator:

- New documentation: `docs/interuss_realistic_flight_mapping.md`.
- Added operational intent fields to `src/vqa_semcom/sim/multi_uav_env.py`: `operational_intent_id`, `operational_intent_state`, and `operational_priority`.
- Added UTM states: accepted, activated, nonconforming, and contingent.
- Added buffered strategic conflict detection with spatial, altitude, and temporal buffers.
- Added DSS availability/delay and subscription notification delay abstractions; UTM delays are included in total task delay and reported separately.
- Added realistic scenarios accepted by `env.reset(options={"formal_scenario": ...})`: `test_utm_nominal_planning`, `test_utm_off_nominal_planning`, `test_utm_intent_conflict`, `test_utm_dss_outage`, and `test_utm_notification_delay`.
- Added UTM trace fields: `strategic_conflict_count`, `strategic_conflict_task_ids`, `dss_available`, `dss_delay_s`, `subscription_notification_delay_s`, `conflict_notification_pending`, and `utm_constraint_violation`.
- Cache-only service still does not create operational intent or airspace conflict; observe/revisit/waypoint actions with `service_level > 0` do.

Environment-owned UTM outputs:

```text
outputs/env/interuss_mapping_summary.md
outputs/env/utm_realistic_scenario_smoke.csv
```

Validation:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 67 tests OK
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_semantic_network_benchmark.py \
  --config configs/v1_9_snr_lut.yaml \
  --steps 5 \
  --seed 29 \
  --output-dir outputs/env \
  --formal-scenarios test_utm_nominal_planning,test_utm_off_nominal_planning,test_utm_intent_conflict,test_utm_dss_outage,test_utm_notification_delay
# wrote outputs/env/utm_realistic_scenario_smoke.csv with 25 rows
```
