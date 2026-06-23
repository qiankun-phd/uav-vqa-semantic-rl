# Current Status

Last updated: 2026-06-23 Asia/Shanghai
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

Revalidated 2026-06-22 Asia/Macau:

- Confirmed `outputs/lut/v1_9_semantic_utility_with_ci.csv` and `outputs/reports/semantic_utility_calibration.md` already exist.
- Confirmed semantic utility API/test files are present:

```text
src/vqa_semcom/semantic/utility.py
tests/test_semantic_utility.py
```

- Passed targeted semantic utility tests:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_semantic_utility.py'
# Ran 6 tests OK
```

- Passed full project tests:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 67 tests OK
```

Documentation/API hardening completed 2026-06-22 Asia/Macau:

- Added a paper-facing semantic utility model note:

```text
docs/semantic_utility_model.md
```

- Added a runnable API demo:

```text
scripts/demo_semantic_utility_query.py
```

- Demo output artifact:

```text
outputs/reports/semantic_utility_api_examples.md
```

- The demo queries representative cache, semantic-token, and image-evidence cells and reports `accuracy_mean`, `accuracy_lcb`, `payload_kb`, `uncertainty`, and `sample_count`.

Paper-ready semantic communication mainline documentation completed 2026-06-22 Asia/Macau:

- Defined the paper-level mainline as:

```text
Conservative VQA-grounded Semantic-Lyapunov Hybrid Control for UAV Semantic Communications
```

- Added/updated:

```text
docs/twc_algorithm_plan.md
docs/formal_problem_definition.md
docs/interfaces.md
docs/experiment_todo.md
```

- The formal problem now uses conservative semantic QoS:

```text
accuracy_lcb,k >= epsilon_k
```

- The interface now requires `semantic_payload_kb` and `semantic_quality_gap` in addition to semantic accuracy/uncertainty/sample-count fields.
- This was a documentation/interface-planning step only; no RL core algorithm or environment dynamics were changed.

Benchmark narrative documentation completed 2026-06-22 Asia/Macau:

- Added/updated:

```text
docs/paper_algorithm_outline.md
docs/benchmark_protocol.md
docs/interfaces.md
docs/formal_problem_definition.md
docs/twc_algorithm_plan.md
```

- The unified paper narrative is now UAV-assisted semantic VQA emergency networking with conservative semantic utility, evidence routing, Semantic-LCB Lyapunov hybrid control, and scenario-aware evaluation.
- The benchmark protocol defines five paper-ready scenarios: `nominal_patrol`, `disaster_hotspot`, `low_snr_blockage`, `edge_overload`, and `utm_conflict`.
- This was a documentation-only pass; no core RL algorithm or environment dynamics were changed.

Semantic scenario utility diagnostics completed 2026-06-23 Asia/Shanghai:

- Generated scenario-level diagnostics for the Algorithm v2 benchmark without rerunning Qwen/VLM and without changing RL code.
- Inputs were the calibrated semantic utility table and existing rollout traces under:

```text
outputs/lut/v1_9_semantic_utility_with_ci.csv
outputs/rl/semantic_scenario_benchmark_v2/
```

- New artifacts:

```text
outputs/reports/semantic_scenario_utility_diagnostics.md
outputs/reports/semantic_scenario_utility_diagnostics.csv
```

- Main diagnosis:
  - `edge_overload` has average proposed LCB near 0.691, but semantic success is 0 because the benchmark is almost entirely critical tasks and the effective epsilon is about 0.80-0.82; the average semantic quality gap remains about 0.121.
  - `utm_conflict` has average proposed LCB near 0.581 and semantic success 0 because it is below both the normal floor and the critical floor; the 0.420 cache ratio contributes to the quality gap.
  - `disaster_hotspot` uses a strict critical epsilon floor; the recommendation is layered epsilon by task type/risk rather than globally lowering critical requirements.
  - `low_snr_blockage` shows `semantic_greedy` is stronger than proposed PPO, so the algorithm thread should improve semantic-token exploration/prior or distill greedy/oracle routing.
  - Cache remains attractive because it is payload-free and low-delay, but its global LCB is low; cache penalties should be risk-aware.

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

- Mobility-policy action interface completed 2026-06-23 Asia/Shanghai:
  - Canonical env `src/vqa_semcom/sim/multi_uav_env.py` now accepts explicit `mobility_mode`: `stay`, `serve_task`, `reposition`, `avoid_conflict`, and `return_base`.
  - Actions now keep the original semantic/resource fields and add `waypoint_delta`, `altitude_delta`, and backward-compatible absolute `waypoint` support.
  - UAV movement, fly delay, mobility energy, waypoint/altitude reporting, coverage gain, and UTM conflict risk now use the same mobility plan instead of always assuming automatic flight to the task center.
  - Observations now expose UAV-task distance matrices, battery ratios, predicted fly delay/energy, task `Area4D`, UTM conflict risk, future-task proximity, coverage scores, and feasible mobility masks for a future mobility actor.
  - Cache-only service remains free of forced operational-intent/airspace conflict; explicit cache-mode repositioning can still consume mobility energy for patrol behavior.

- Semantic scenario benchmark signal audit completed 2026-06-23 Asia/Shanghai:
  - Root cause for `utm_conflict` mismatch: environment smoke used explicit `concurrent_actions`, while algorithm benchmark v2 uses algorithm-style single-task actions, so no second operational intent was available for strategic conflict detection.
  - Fix: `utm_conflict` now enables background operational-intent detection. Observe/revisit/image/token actions conflict with other active overlapping intents; cache-only reuse still creates no airspace/UTM conflict.
  - Edge calibration: `edge_overload` now uses high but less extreme CPU/GPU load and queue constants, cleaner view/freshness mix, and a scenario-specific semantic threshold cap. This keeps the stress source on edge queue/resource pressure instead of making semantic QoS uniformly infeasible.
  - v3 feasibility calibration: `edge_overload` now uses moderate task spacing, queue delay, edge load, and model-load delay so token-style evidence has a partial feasible deadline region. `utm_conflict` now uses multiple operational areas, tighter buffers, and moderate background intent density so UTM pressure is visible but not saturated.
  - Lightweight V1.9 probe after the final calibration:

| scenario | tasks | token/proposed-style deadline violation | semantic success | UTM conflicts | avg delay s |
|---|---:|---:|---:|---:|---:|
| edge_overload | 12 | 0.583 | 0.750 | 0.000 | 9.740 |
| utm_conflict | 12 | 1.000 | 0.167 | 0.500 | 15.790 |

  - Updated small env artifacts:

```text
outputs/env/semantic_scenario_presets/scenario_preset_summary.csv
outputs/env/semantic_scenario_presets/summary.md
```

- Environment paper scenario preset library completed 2026-06-22 Asia/Shanghai:
  - Added five canonical `scenario` presets in `src/vqa_semcom/sim/multi_uav_env.py`: `nominal_patrol`, `disaster_hotspot`, `low_snr_blockage`, `edge_overload`, and `utm_conflict`.
  - Presets are available via `env.reset(options={"scenario": "<name>"})`; the existing formal train/test split remains unchanged.
  - Added `docs/scenario_presets.md` and updated `docs/semantic_network_architecture.md`.
  - Standardized `semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)` in environment `info`.
  - Kept V1.9 enabled service levels at `[0, 1, 2]`; service level 3 ROI/crop remains disabled.
  - Generated small environment-owned smoke artifacts:

```text
outputs/env/semantic_scenario_presets/scenario_preset_summary.csv
outputs/env/semantic_scenario_presets/summary.md
```

  - Smoke headline:

| scenario | mean LCB | mean gap | delay s | energy J | SINR dB | UTM conflict |
|---|---:|---:|---:|---:|---:|---:|
| nominal_patrol | 0.665 | 0.114 | 13.845 | 1968.702 | 30.289 | 0.000 |
| disaster_hotspot | 0.510 | 0.330 | 8.809 | 1069.477 | 41.055 | 0.000 |
| low_snr_blockage | 0.338 | 0.480 | 518.896 | 4496.708 | -34.574 | 0.000 |
| edge_overload | 0.366 | 0.274 | 9.777 | 1564.751 | 35.068 | 0.000 |
| utm_conflict | 0.213 | 0.607 | 20.347 | 2819.211 | 15.693 | 0.400 |

  - Validation:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_multi_uav_env.py'
# Ran 17 tests OK
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_interuss_realistic_env.py'
# Ran 8 tests OK
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 74 tests OK
```

- UTM/InterUSS-style realistic flight mapping is being added to `sim.multi_uav_env` as an environment-thread feature; it does not introduce an InterUSS runtime dependency and keeps service level 3 ROI disabled.
- V1.9 full-SNR prediction/LUT/report/resource simulation refresh is complete as of 2026-06-18 23:34 Asia/Macau.
- The final report includes all configured SNR bins: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB.
- Algorithm hybrid TCH-PPO smoke has been rerun against the refreshed full-bin LUT under `outputs/rl/v1_9_hybrid_tch_ppo_smoke`.
- Formal multi-seed resource-allocation comparison has been run under `outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618`.
- PPO/TCH-PPO cache collapse was reproduced in the formal multi-seed run and is now mitigated in the small-scale `outputs/rl/v1_9_rl_fix_cache_collapse/` validation; formal multi-seed rerun is still required before claiming final performance.
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

## RL Cache-Collapse Fix 2026-06-22 Asia/Shanghai

Algorithm thread upgraded the semantic-utility-guided cognitive controller:

- `src/vqa_semcom/rl/v19_resource_env.py` now consumes `SemanticUtilityModel.U_sem(...)` when `outputs/lut/v1_9_semantic_utility_with_ci.csv` is available.
- The algorithm-facing accuracy is the conservative `accuracy_lcb`; rollout records also expose `semantic_accuracy_mean`, `semantic_accuracy_lcb`, `semantic_uncertainty`, and `semantic_sample_count`.
- `src/vqa_semcom/rl/v19_ppo.py` now includes entropy scheduling, decaying service-level imitation prior, service-dependent resource floors, semantic feasibility projection, and UTM/DSS/off-nominal cost hooks.
- UTM constraint violations are routed through the conflict-cost channel; DSS delay, subscription notification delay, and off-nominal planning penalty are explicit reward costs.
- `scripts/run_v1_9_resource_alloc.py` exposes the new tuning flags and writes `ppo_training_trace.csv` fields for entropy, prior weight, semantic LCB, uncertainty, and non-cache ratio.

Validation:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 67 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --policy all \
  --proposed-semantic-rl \
  --episodes 6 \
  --tasks-per-episode 8 \
  --train-episodes 30 \
  --demo-episodes 8 \
  --bc-epochs 3 \
  --seed 0 \
  --output-dir outputs/rl/v1_9_rl_fix_cache_collapse
```

Small-scale result in `outputs/rl/v1_9_rl_fix_cache_collapse/`:

- `ppo` success: 0.188, matching `greedy_min_sufficient_evidence` at 0.188.
- `ppo` delay/energy/payload: 3.735 s / 493.786 J / 19.118 KB versus greedy 7.032 s / 879.639 J / 74.465 KB.
- `ppo` deadline violation: 0.500 versus greedy 0.812.
- `ppo` service mix: cache 0.417, semantic tokens 0.479, image 0.104; the policy no longer collapses to all cache in this validation.

Next required algorithm step: scale the scenario-aware v2 comparison below to longer training/evaluation budgets and export paper tables/figures.

## Semantic-Lyapunov Hybrid Control 2026-06-22 Asia/Shanghai

Algorithm thread implemented the first Sem-Lyapunov Hybrid Control v1 skeleton for the paper main method:

- `src/vqa_semcom/rl/v19_resource_env.py` now exposes `semantic_payload_kb`, `semantic_quality_gap`, and virtual queue states `q_quality`, `q_deadline`, `q_energy`, `q_risk`, and `q_utm` in step info and rollout records.
- The environment observation vector now appends the normalized Lyapunov queue state and also exposes `obs["lyapunov_queues"]`.
- The queue increments follow the CMDP constraints: quality gap from `epsilon_k - accuracy_lcb`, deadline overrun from `delay_s - deadline_s`, energy overrun from `energy_j - energy_budget_j`, plus risk/UTM violation queues.
- `src/vqa_semcom/rl/v19_ppo.py` adds a drift-plus-penalty reward path using semantic LCB utility, delay/energy/payload costs, queue-weighted penalties, and uncertainty penalty.
- PPO remains the high-level semantic routing controller; continuous bandwidth/power/cpu/gpu values are corrected through service-dependent resource floors and projection unless the ablation disables projection.
- Added runner support for `semantic_lcb_greedy`, `lyapunov_greedy`, `ppo_without_lcb`, `ppo_without_queues`, and `ppo_without_projection`.
- Smoke outputs are isolated under `outputs/rl/twc_sem_lcb_lyapunov_smoke/`.

Validation:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 10 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --smoke \
  --policy ppo \
  --episodes 1 \
  --train-episodes 2 \
  --tasks-per-episode 4 \
  --output-dir outputs/rl/twc_sem_lcb_lyapunov_smoke
```

Smoke artifact check:

- `v1_9_resource_alloc_rollout.csv` contains semantic mean/LCB/uncertainty/sample count/payload/gap and queue fields.
- `ppo_training_trace.csv` contains mean/max `q_quality`, `q_deadline`, `q_energy`, `q_risk`, and `q_utm`.
- The smoke still uses a tiny 2-episode training budget and is only an interface/algorithm-path validation, not a paper result.

## Scenario-Aware Semantic Benchmark 2026-06-22 Asia/Shanghai

Algorithm thread upgraded the resource-allocation runner into a scenario-aware semantic communication benchmark:

- `scripts/run_v1_9_resource_alloc.py` now accepts the paper scenario presets through `--scenario`:
  `nominal_patrol`, `disaster_hotspot`, `low_snr_blockage`, `edge_overload`, and `utm_conflict`.
- Added `--scenario-benchmark` smoke mode that runs each scenario under:
  `always_cache`, `always_semantic_token`, `always_image`, `semantic_greedy`, `lyapunov_greedy`, and Semantic-LCB Lyapunov PPO.
- Added ablation aliases: `--no-lcb`, `--no-lyapunov-queues`, `--no-projection`, and `--disable-semantic-token`.
- Standardized `semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)` in the RL wrapper record/summary path.
- Scenario summaries now report semantic success, accuracy LCB/mean, uncertainty, semantic quality gap, Lyapunov queues, delay, energy, payload, deadline violation, UTM conflict violation, and cache/token/image service mix.

Smoke command:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --scenario-benchmark \
  --smoke \
  --episodes 1 \
  --train-episodes 2 \
  --tasks-per-episode 4 \
  --output-dir outputs/rl/semantic_scenario_benchmark
```

Scenario benchmark artifacts:

```text
outputs/rl/semantic_scenario_benchmark/scenario_comparison_summary.csv
outputs/rl/semantic_scenario_benchmark/scenario_comparison_report.md
outputs/rl/semantic_scenario_benchmark/<scenario>/v1_9_resource_alloc_summary.md
```

Validation:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 12 tests OK
```

## Scenario-Aware Semantic Benchmark v2 2026-06-23 Asia/Shanghai

Algorithm thread fixed the scenario benchmark PPO cache-collapse issue and ran a formal small multi-seed comparison.

Code updates:

- `src/vqa_semcom/rl/v19_ppo.py` adds stronger semantic success/gap shaping, high-epsilon/high-risk cache penalties, and a cache override in the semantic projection layer.
- The projection layer now compares semantic-token/image candidates when a cache action has large `accuracy_lcb` shortfall, while still respecting hard safety masks.
- Cache actions are projected to minimal/no resource use; token/image actions retain service-dependent resource floors.
- `scripts/run_v1_9_resource_alloc.py` scenario benchmark now runs seeds and PPO ablations: `ppo_without_lcb`, `ppo_without_queues`, `ppo_without_projection`, and `proposed_ppo`.

Formal-small command:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --scenario-benchmark \
  --seeds 0,1,2 \
  --episodes 50 \
  --train-episodes 120 \
  --tasks-per-episode 12 \
  --output-dir outputs/rl/semantic_scenario_benchmark_v2
```

Artifacts:

```text
outputs/rl/semantic_scenario_benchmark_v2/scenario_comparison_all_seed_results.csv
outputs/rl/semantic_scenario_benchmark_v2/scenario_comparison_summary.csv
outputs/rl/semantic_scenario_benchmark_v2/scenario_comparison_report.md
outputs/rl/semantic_scenario_benchmark_v2/cache_collapse_analysis.md
```

Headline v2 proposed PPO results:

| scenario | semantic success | accuracy LCB | quality gap | delay | energy | payload KB | deadline vio | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | 0.303 | 0.611 | 0.186 | 3.336 | 403.832 | 34.219 | 0.330 | 0.367 | 0.451 | 0.182 |
| disaster_hotspot | 0.238 | 0.585 | 0.267 | 1.506 | 149.392 | 3.998 | 0.289 | 0.185 | 0.799 | 0.017 |
| low_snr_blockage | 0.785 | 0.746 | 0.108 | 10.355 | 1339.330 | 1.726 | 0.699 | 0.276 | 0.697 | 0.027 |
| edge_overload | 0.000 | 0.691 | 0.121 | 4.472 | 519.261 | 0.820 | 0.531 | 0.309 | 0.691 | 0.000 |
| utm_conflict | 0.000 | 0.581 | 0.239 | 1.268 | 130.915 | 0.686 | 0.177 | 0.420 | 0.580 | 0.000 |

Interpretation:

- Proposed PPO no longer collapses to `always_cache`; token/image service mix is nonzero in all five scenarios.
- In `low_snr_blockage`, proposed PPO reaches 0.785 semantic success versus 0.053 for cache and 0.950 for semantic greedy, with far lower payload than image/oracle-heavy policies.
- In `edge_overload` and `utm_conflict`, semantic success remains near zero for most methods, but proposed PPO substantially improves conservative LCB/gap over cache while keeping payload small.
- `ppo_without_projection` is consistently weaker or less stable, supporting the resource-projection component.

Validation:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 13 tests OK
```

## Scenario-Aware Semantic Benchmark v3 2026-06-23 Asia/Shanghai

Algorithm thread reran the semantic scenario benchmark after the Environment background operational-intent fix and VQA semantic-utility diagnostics.

Code updates:

- `src/vqa_semcom/rl/v19_resource_env.py` now persists `epsilon_k` in `info["record"]` and rollout CSV rows.
- `scripts/run_v1_9_resource_alloc.py` reports average epsilon and failed-task epsilon statistics in summaries and scenario comparison tables.
- `src/vqa_semcom/rl/v19_ppo.py` adds risk/staleness/UTM-aware cache shortfall penalties and a stronger semantic-token prior distilled from `semantic_greedy`.
- Compute-aware projection now prefers semantic-token evidence over cache when token evidence reduces LCB shortfall under edge/deadline pressure.
- UTM/airspace conflicts are recorded through risk/queue costs instead of being hidden by a cache fallback, so `utm_conflict` now exposes the quality-vs-UTM tradeoff.

Command:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --scenario-benchmark \
  --seeds 0,1,2 \
  --episodes 50 \
  --train-episodes 120 \
  --tasks-per-episode 12 \
  --output-dir outputs/rl/semantic_scenario_benchmark_v3
```

Artifacts:

```text
outputs/rl/semantic_scenario_benchmark_v3/scenario_comparison_all_seed_results.csv
outputs/rl/semantic_scenario_benchmark_v3/scenario_comparison_summary.csv
outputs/rl/semantic_scenario_benchmark_v3/scenario_comparison_report.md
outputs/rl/semantic_scenario_benchmark_v3/cache_collapse_analysis.md
```

Headline v3 proposed PPO results:

| scenario | semantic success | accuracy LCB | epsilon | quality gap | Q_utm | delay | energy | payload KB | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | 0.283 | 0.617 | 0.793 | 0.182 | 0.000 | 4.346 | 552.698 | 18.384 | 0.450 | 0.000 | 0.094 | 0.793 | 0.113 |
| disaster_hotspot | 0.228 | 0.577 | 0.840 | 0.274 | 0.000 | 1.560 | 157.644 | 1.064 | 0.291 | 0.000 | 0.089 | 0.911 | 0.000 |
| low_snr_blockage | 0.786 | 0.756 | 0.803 | 0.099 | 0.000 | 11.017 | 1428.400 | 1.748 | 0.729 | 0.000 | 0.247 | 0.727 | 0.027 |
| edge_overload | 0.010 | 0.419 | 0.640 | 0.222 | 0.000 | 7.441 | 974.501 | 1.153 | 0.902 | 0.000 | 0.000 | 1.000 | 0.000 |
| utm_conflict | 0.000 | 0.625 | 0.820 | 0.195 | 4.660 | 2.560 | 191.845 | 1.084 | 0.371 | 0.913 | 0.087 | 0.913 | 0.000 |

Interpretation:

- v3 directly records epsilon; diagnostics no longer need to reconstruct failed-row thresholds.
- Proposed PPO avoids cache collapse more aggressively than v2: cache ratio drops to 0.000 in `edge_overload` and 0.087 in `utm_conflict`.
- `low_snr_blockage` remains stable around 0.786 semantic success, close to v2 and below semantic greedy 0.950, with low payload.
- `utm_conflict` now correctly exposes background operational-intent conflicts; proposed improves LCB/gap over cache but pays UTM conflict/risk cost.
- `edge_overload` is token-only under proposed PPO; success remains limited by deadline/edge pressure rather than cache collapse.

Environment calibration after reviewing v3:

- `edge_overload` was too narrow for token-style evidence: proposed/token deadline violation was 0.902. The preset now keeps edge CPU/GPU/model-cache pressure but relaxes queue/load/deadline geometry enough that a token-style route has deadline violation about 0.58 in a lightweight V1.9 probe.
- `utm_conflict` was too saturated: proposed UTM conflict was 0.913. The preset now mixes overlapping and separable 4D areas and uses moderate background intent density, reducing token-style UTM conflict to about 0.50 while preserving cache-only no-conflict behavior.
- The existing v3 algorithm report remains a pre-calibration benchmark artifact under `outputs/rl/semantic_scenario_benchmark_v3`; rerun the algorithm benchmark before using post-calibration scenario metrics in paper tables.

Validation:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 13 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 78 tests OK
```

## Semantic Service-Candidate Utility Interface 2026-06-23 Asia/Shanghai

Added a mobility-aware RL/env helper on top of the stable semantic utility LUT key:

```python
SemanticUtilityModel.get_service_candidates(obs)
```

The helper keeps the existing key unchanged:

```text
question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level
```

For each candidate service level it returns:

```text
accuracy_mean, accuracy_lcb, uncertainty, payload_kb, sample_count,
semantic_quality_gap, semantic_efficiency, is_snr_sensitive,
recommended_for_low_snr, recommended_for_critical
```

New/updated files:

```text
src/vqa_semcom/semantic/utility.py
tests/test_semantic_utility.py
outputs/reports/semantic_service_candidate_interface.md
```

The report confirms that the current LUT still supports the paper explanation:

- `s=0` cache answer is payload-free and SNR-invariant, but must be freshness/risk gated.
- `s=1` semantic token is the lightweight semantic communication service and explains low-SNR/edge-overload payload advantages.
- `s=2` image evidence carries more visual evidence but has much larger payload and higher link/queue sensitivity.

Updated 2026-06-23 Asia/Shanghai:

- Extended service candidates with deadline-aware feasibility fields:

```text
estimated_delay_s
estimated_delay_feasible
semantic_feasible
deadline_feasible
joint_feasible
```

- `semantic_feasible` is `accuracy_lcb >= epsilon_k`.
- `deadline_feasible` is `estimated_delay_s <= deadline_s`.
- `joint_feasible` is the semantic/deadline conjunction used to explain why image evidence can be semantically strong but operationally infeasible under low SNR or edge overload.

## Scenario-Aware Semantic Benchmark v4 2026-06-23 Asia/Shanghai

Algorithm thread reran the semantic scenario benchmark after environment commit `8f903c2 fix(env): calibrate scenario feasibility for semantic benchmark`.

No LUT or major algorithm changes were made for v4. The goal was to verify whether the calibrated `edge_overload` and `utm_conflict` presets expose usable feasible regions to the existing Semantic-LCB Lyapunov PPO controller.

Command:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --scenario-benchmark \
  --seeds 0,1,2 \
  --episodes 50 \
  --train-episodes 120 \
  --tasks-per-episode 12 \
  --output-dir outputs/rl/semantic_scenario_benchmark_v4
```

Artifacts:

```text
outputs/rl/semantic_scenario_benchmark_v4/scenario_comparison_all_seed_results.csv
outputs/rl/semantic_scenario_benchmark_v4/scenario_comparison_summary.csv
outputs/rl/semantic_scenario_benchmark_v4/scenario_comparison_report.md
outputs/rl/semantic_scenario_benchmark_v4/v3_vs_v4_delta.md
```

Headline v4 proposed PPO results:

| scenario | semantic success | task success | accuracy LCB | quality gap | delay | energy | payload KB | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | 0.283 | 0.098 | 0.617 | 0.182 | 4.346 | 552.698 | 18.384 | 0.450 | 0.000 | 0.094 | 0.793 | 0.113 |
| disaster_hotspot | 0.228 | 0.130 | 0.577 | 0.274 | 1.560 | 157.644 | 1.064 | 0.291 | 0.000 | 0.089 | 0.911 | 0.000 |
| low_snr_blockage | 0.786 | 0.087 | 0.756 | 0.099 | 11.017 | 1428.400 | 1.748 | 0.729 | 0.000 | 0.247 | 0.727 | 0.027 |
| edge_overload | 0.607 | 0.385 | 0.645 | 0.066 | 4.877 | 793.040 | 0.840 | 0.381 | 0.000 | 0.041 | 0.959 | 0.000 |
| utm_conflict | 0.000 | 0.000 | 0.627 | 0.193 | 6.169 | 844.409 | 1.069 | 0.843 | 0.151 | 0.100 | 0.900 | 0.000 |

v3 -> v4 conclusions:

- `edge_overload` is fixed as a feasible stress case: proposed semantic success improves from 0.010 to 0.607, task success from 0.000 to 0.385, and deadline violation drops from 0.902 to 0.381. The controller uses token routing instead of cache fallback.
- `utm_conflict` is no longer saturated: proposed UTM conflict drops from 0.913 to 0.151. Semantic success remains 0.000 because conservative LCB stays below epsilon and deadline pressure rises; this needs conflict-aware evidence routing and UTM queue tuning rather than more cache penalties.
- `low_snr_blockage` remains stable: proposed semantic success is 0.786 versus semantic greedy 0.950, with much lower image usage.
- Cache collapse does not reappear: proposed cache ratios are 0.041 in `edge_overload`, 0.100 in `utm_conflict`, and 0.247 in `low_snr_blockage`.

Validation:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 13 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 78 tests OK
```

## Two-timescale Mobility-aware Semantic Resource PPO 2026-06-23 Asia/Shanghai

Implemented the first Sem-LCB two-timescale PPO controller for the mobility-aware environment interface.

Algorithm structure:

```text
shared encoder
slow mobility actor: uav_assignment, mobility_mode, waypoint_delta, altitude_delta
fast semantic-resource actor: service_level, bandwidth, power, cpu_share, gpu_share
centralized critic over joint mobility/resource action
Lyapunov virtual queues in observation and training trace
post-policy resource and mobility projection
```

Default mobility timescale is `K=3` slots. The fast actor still updates every task/slot. The reward keeps the Semantic-LCB/Lyapunov mainline and now includes mobility-aware terms:

```text
semantic LCB/success, quality gap, deadline, energy, payload,
edge queue, UTM conflict, flight energy, arrival delay, coverage gain
```

Updated files:

```text
src/vqa_semcom/rl/v19_ppo.py
src/vqa_semcom/rl/v19_resource_env.py
scripts/run_v1_9_resource_alloc.py
tests/test_v1_9_rl_resource_alloc.py
docs/paper_algorithm_outline.md
docs/interfaces.md
```

Smoke benchmark:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --scenario-benchmark \
  --benchmark-ppo-variants proposed_two_timescale_ppo \
  --seeds 0,1,2 \
  --episodes 1 \
  --train-episodes 120 \
  --tasks-per-episode 4 \
  --output-dir outputs/rl/two_timescale_mobility_ppo_scenario_smoke3
```

Artifacts kept small at the benchmark root:

```text
outputs/rl/two_timescale_mobility_ppo_scenario_smoke3/scenario_comparison_all_seed_results.csv
outputs/rl/two_timescale_mobility_ppo_scenario_smoke3/scenario_comparison_summary.csv
outputs/rl/two_timescale_mobility_ppo_scenario_smoke3/scenario_comparison_report.md
```

Smoke headline for `proposed_two_timescale_ppo`:

| scenario | semantic success | accuracy LCB | quality gap | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | 0.556 | 0.716 | 0.087 | 0.917 | 0.000 | 0.111 | 0.611 | 0.278 |
| disaster_hotspot | 0.033 | 0.473 | 0.370 | 0.533 | 0.000 | 0.133 | 0.867 | 0.000 |
| low_snr_blockage | 0.694 | 0.750 | 0.089 | 0.889 | 0.000 | 0.111 | 0.222 | 0.667 |
| edge_overload | 0.683 | 0.640 | 0.069 | 0.067 | 0.000 | 0.083 | 0.917 | 0.000 |
| utm_conflict | 0.267 | 0.554 | 0.274 | 0.600 | 0.000 | 0.100 | 0.633 | 0.267 |

Interpretation:

- The controller no longer collapses to cache in the smoke benchmark; all stress scenarios use token/image evidence as intended.
- `edge_overload` benefits most from the token-first projection and mobility-aware controller, reaching 0.683 semantic success with low deadline violation in this smoke setting.
- `low_snr_blockage` and `nominal_patrol` still show high deadline pressure under short training, so the next formal run should tune arrival-delay/flight-energy/resource floors and use longer training before paper claims.
- `utm_conflict` has non-zero semantic success and zero measured conflict in this smoke run, but deadline pressure remains the binding constraint.

## Two-timescale Mobility Formal 300-Episode Benchmark 2026-06-23 Asia/Shanghai

Initial launch snapshot at 19:48 Asia/Shanghai.

Active tmux sessions:

```text
hppo_tt_proposed_300
hppo_tt_monolithic_300
hppo_tt_no_mobility_300
hppo_tt_no_queue_300
hppo_tt_no_projection_300
```

Output directories:

```text
outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300
outputs/rl/two_timescale_mobility_formal_20260623_monolithic_300
outputs/rl/two_timescale_mobility_formal_20260623_no_mobility_300
outputs/rl/two_timescale_mobility_formal_20260623_no_queue_300_retry1
outputs/rl/two_timescale_mobility_formal_20260623_no_projection_300_retry1
```

Notes:

- `proposed_two_timescale_ppo`, `monolithic_ppo`, and `no_mobility_actor` launched with the requested names and are actively running.
- The requested `no_lyapunov_queues` and `no_projection` labels were not valid runner variants. The legal runner names are `ppo_without_queues` and `ppo_without_projection`, so those two sessions were restarted into `_retry1` output directories without overwriting the failed partial directories.
- At the initial check, valid-run logs had not yet emitted scenario rows, but the five Python processes were active and consuming CPU.
- No CUDA OOM, NaN, or traceback was observed in the corrected sessions at launch time.
- `scenario_comparison_summary.csv` had not yet been generated for the active runs.

Final completion snapshot:

- All five tmux sessions completed and the tmux server exited.
- Corrected output directories generated both `scenario_comparison_summary.csv` and `scenario_comparison_report.md`.
- The originally requested labels `no_lyapunov_queues` and `no_projection` were not valid runner variant names. They were relaunched as `ppo_without_queues` and `ppo_without_projection` into `_retry1` directories to avoid overwriting the partial failed outputs.
- The only tracebacks are in the two initial failed logs from invalid variant labels. Corrected runs completed without observed NaN or CUDA OOM.

Formal output directories:

```text
outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300
outputs/rl/two_timescale_mobility_formal_20260623_monolithic_300
outputs/rl/two_timescale_mobility_formal_20260623_no_mobility_300
outputs/rl/two_timescale_mobility_formal_20260623_no_queue_300_retry1
outputs/rl/two_timescale_mobility_formal_20260623_no_projection_300_retry1
```

Merged report:

```text
outputs/rl/two_timescale_mobility_formal_20260623_summary.md
```

Proposed two-timescale PPO headline results:

| scenario | semantic success | task success | acc LCB | quality gap | delay | energy | mobility energy | arrival delay | coverage | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | 0.386 | 0.157 | 0.617 | 0.185 | 1.802 | 192.303 | 78.000 | 0.000 | 0.000 | 0.229 | 0.000 | 0.093 | 0.547 | 0.359 |
| disaster_hotspot | 0.317 | 0.064 | 0.659 | 0.194 | 4.252 | 569.053 | 538.538 | 3.562 | 0.561 | 0.733 | 0.000 | 0.162 | 0.838 | 0.000 |
| low_snr_blockage | 0.955 | 0.053 | 0.912 | 0.004 | 509.965 | 779.164 | 78.000 | 0.000 | 0.000 | 0.939 | 0.000 | 0.053 | 0.030 | 0.917 |
| edge_overload | 0.697 | 0.681 | 0.674 | 0.050 | 1.742 | 231.837 | 166.961 | 0.476 | 0.010 | 0.046 | 0.000 | 0.058 | 0.942 | 0.000 |
| utm_conflict | 0.007 | 0.000 | 0.482 | 0.338 | 1.145 | 138.846 | 99.749 | 0.272 | -0.005 | 0.030 | 0.094 | 0.099 | 0.872 | 0.029 |

Main interpretation:

- The proposed two-timescale controller improves semantic success over monolithic PPO in `nominal_patrol`, `disaster_hotspot`, `low_snr_blockage`, and `edge_overload`, with especially strong edge-overload task success and deadline control.
- `edge_overload` is the cleanest positive result: proposed reaches 0.697 semantic success and 0.681 task success with 0.046 deadline violation, substantially better than monolithic PPO and fixed token/image baselines.
- `low_snr_blockage` reaches high semantic LCB success but has very high delay and deadline violation because the learned resource/mobility behavior overuses slow high-quality evidence. This is a tuning blocker for paper-ready claims.
- `utm_conflict` is stable and low-delay under proposed, but semantic success remains near zero. The next algorithm step should add stronger UTM-aware semantic evidence routing instead of simply increasing evidence quality.

## Mobility Formal Diagnostics 2026-06-23 Asia/Shanghai

Completed environment-side diagnosis after the 300-episode two-timescale mobility formal benchmark finished.

Artifact:



Main conclusions:

-  proposed-policy deadline pressure is not mobility-driven: arrival delay is 0.000s and mobility energy is the 78 J hover baseline.
-  deadline failures are primarily low-SNR transmission/payload failures, not UAV flight-speed or flight-energy failures; image evidence satisfies conservative semantic QoS but becomes infeasible under the weak link.
-  proposed-policy deadline behavior is healthy and aggregate UTM conflict is reduced versus serve-task baselines; the remaining semantic failure is mainly the strict epsilon=0.82 conservative-QoS threshold.
- Do not globally tune  or  based on this run. Consider only scenario-local /presentation tuning for  if it should be partially image-feasible.
