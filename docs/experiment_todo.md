# Experiment TODO

Last updated: 2026-06-23 Asia/Shanghai
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

Completed 2026-06-19 00:28 Asia/Macau:

- Promoted the raw SNR-LUT result into a VQA-grounded task-conditioned semantic utility model.
- Added `src/vqa_semcom/semantic/utility.py` and `tests/test_semantic_utility.py`.
- Generated:

```text
outputs/lut/v1_9_semantic_utility_with_ci.csv
outputs/reports/semantic_utility_calibration.md
```

- Utility CSV now exposes `sample_count`, `accuracy_mean`, `accuracy_ci_low`, `accuracy_ci_high`, `accuracy_lcb`, `payload_kb`, and `uncertainty`.
- Calibration report confirms all configured SNR bins are present: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB.
- SNR monotonic sanity check adjusted 48 cells; 108 sparse cells are explicitly marked; 216 cache cells are SNR-invariant.
- Verified:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_semantic_utility.py'
```

Revalidated 2026-06-22 Asia/Macau:

- Confirmed semantic utility API, test, calibrated utility CSV, and calibration report are present.
- Passed:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_semantic_utility.py'
# Ran 6 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 67 tests OK
```

- Ready to commit and push the semantic utility interface and generated calibration artifacts.

Documented 2026-06-22 Asia/Macau:

- Added the paper-facing utility model description:

```text
docs/semantic_utility_model.md
```

- Added the query demo script and generated API examples:

```text
scripts/demo_semantic_utility_query.py
outputs/reports/semantic_utility_api_examples.md
```

- The documentation now explains `U_sem`, Wilson CI, SNR monotonic sanity checking, cache SNR-invariant behavior, sparse-cell handling, and why RL should use `accuracy_lcb`.

## Immediate Integration Tasks

1. Done: Reran canonical environment smoke against the refreshed full-bin V1.9 LUT.
2. Done: Reran algorithm smoke and formal resource-allocation baselines/PPO variants against the refreshed full-bin V1.9 LUT.
3. Done: Regenerated merged comparison table that uses both:

```text
outputs/sim/v1_9_snr_resource_results.csv
outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618/formal_comparison_summary.csv
```

4. Done: Built the calibrated semantic utility interface for future RL/environment integration:

```text
outputs/lut/v1_9_semantic_utility_with_ci.csv
src/vqa_semcom/semantic/utility.py
```

5. Done: Algorithm wrapper consumes `U_sem(...).accuracy_lcb`, `payload_kb`, `uncertainty`, and `sample_count` when the calibrated semantic utility CSV is available.
6. Done: Added `semantic_accuracy_mean`, `semantic_accuracy_lcb`, `semantic_uncertainty`, and `semantic_sample_count` to rollout `info` and CSV outputs.
7. Done: Exposed `semantic_payload_kb` and standardized `semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)` for Semantic-Lyapunov queue updates.
8. Do not overwrite VQA/SNR-LUT artifacts or algorithm outputs owned by another thread.

## Paper-ready Mainline TODO

Target paper mainline:

```text
Conservative VQA-grounded Semantic-Lyapunov Hybrid Control for UAV Semantic Communications
```

Algorithm thread responsibilities:

1. Implement Semantic-Lyapunov virtual queues for quality, deadline, energy, and UTM/risk pressure.
2. Build hybrid control around semantic evidence routing plus feasible resource projection.
3. Keep the action contract unchanged: `service_level`, `bandwidth`, `power`, `cpu_share`, `gpu_share`, `uav_assignment`, `waypoint`.
4. Treat RL as a residual/learned policy on top of queue/projection structure, not as the only contribution.
5. Keep algorithm outputs under `outputs/rl`, `outputs/hppo`, or `runs`.

Environment thread responsibilities:

1. Consume the semantic utility interface and expose semantic QoS fields in `info`.
2. Add queue-ready metrics: `semantic_quality_gap`, deadline gap, energy pressure, and UTM/risk pressure.
3. Preserve environment dynamics compatibility; do not change UAV mobility/channel dynamics for this documentation-only planning step.
4. Keep environment outputs under `outputs/env`.

Final ablations required for the paper:

- without LCB: use `accuracy_mean` instead of `accuracy_lcb`,
- without queue: remove Semantic-Lyapunov virtual queues,
- without projection: remove resource/action projection,
- without semantic tokens: disable `s=1`,
- oracle and greedy semantic-utility baselines,
- fixed-service baselines: always cache, always semantic tokens, always image.

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

Completed 2026-06-23 Asia/Shanghai:

1. Audited the five semantic scenario presets against `outputs/rl/semantic_scenario_benchmark_v2/scenario_comparison_report.md`.
2. Confirmed `scenario` was passed into the benchmark runner and `utm_conflict_violation` was present in rollout/summary fields.
3. Identified the UTM mismatch root cause: benchmark actions are single-task actions without explicit `concurrent_actions`, while env smoke injected concurrent operational intents.
4. Added background operational-intent conflict detection for the `utm_conflict` preset only; cache-only service remains conflict-free.
5. Recalibrated `edge_overload` so semantic evidence can be feasible for a subset of tasks while edge queue/deadline pressure remains high.
6. Regenerated environment-owned small summaries under `outputs/env/semantic_scenario_presets/`.
7. Verified lightweight V1.9 probes: `utm_conflict` token/image actions now produce UTM conflict 12/12; `edge_overload` token/image semantic feasibility is 9/12 and 4/12 while deadline pressure remains 12/12.

Completed 2026-06-22 Asia/Shanghai:

1. Added five paper-facing UAV semantic VQA scenario presets in `src/vqa_semcom/sim/multi_uav_env.py`: `nominal_patrol`, `disaster_hotspot`, `low_snr_blockage`, `edge_overload`, and `utm_conflict`.
2. Kept the presets as ordinary `scenario` entries so the existing formal train/test split remains stable.
3. Standardized queue-ready semantic QoS gap as `semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)`.
4. Added `docs/scenario_presets.md` and linked the scenario library from `docs/semantic_network_architecture.md`.
5. Generated small preset smoke summaries under:

```text
outputs/env/semantic_scenario_presets/scenario_preset_summary.csv
outputs/env/semantic_scenario_presets/summary.md
```

6. Updated tests so all five presets reset/evaluate successfully, expose semantic/UTM/risk fields, and keep service level 3 disabled.
7. Verified:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_multi_uav_env.py'
# Ran 17 tests OK
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_interuss_realistic_env.py'
# Ran 8 tests OK
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 74 tests OK
```

Completed 2026-06-19 Asia/Shanghai:

1. Added InterUSS/UTM realistic flight mapping without adding an external InterUSS dependency.
2. Added `docs/interuss_realistic_flight_mapping.md` covering DSS, USS qualifier, ASTM F3548 scenario families, and environment field mapping.
3. Extended `src/vqa_semcom/sim/multi_uav_env.py` with operational intent id/state/priority, buffered strategic conflict detection, DSS availability/delay, subscription notification delay, and UTM-specific info fields.
4. Added realistic UTM formal scenarios: `test_utm_nominal_planning`, `test_utm_off_nominal_planning`, `test_utm_intent_conflict`, `test_utm_dss_outage`, and `test_utm_notification_delay`.
5. Added `tests/test_interuss_realistic_env.py` for nominal activation, buffered conflict detection, cache-only no-conflict behavior, DSS outage/contingent state, notification delay, and spatial/temporal buffer behavior.
6. Prepared environment-owned summary outputs under `outputs/env/`: `interuss_mapping_summary.md` and `utm_realistic_scenario_smoke.csv`.
7. Passed `/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests` with 67 tests OK; UTM smoke covered all five realistic scenarios with 25 rows.

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

Next environment steps:

1. Use the fixed scenarios for algorithm ablations: conflict penalty, interference-aware resource allocation, cache reuse, and mobility/energy stress.
2. Calibrate constants against a concrete UAV platform if platform specs become available; current values are lightweight, explainable defaults.
3. Add a full Gymnasium/DI-engine adapter only after the canonical env/action mask stabilizes.
4. Extend service level 3 ROI only after the V1.9/V2 LUT explicitly contains ROI/crop rows.

## Integration TODO

1. Done: Algorithm thread consumes canonical environment obs/action contract through `rl.v19_resource_env`.
2. Done: Environment thread calls the V1.9 LUT to estimate answer accuracy/payload from `snr_bin`.
3. Done: VQA thread refreshed the full-bin V1.9 LUT/report/resource summary after PID 1208659 finished.
4. Done: Reran algorithm hybrid TCH-PPO smoke after the rebuilt full-bin V1.9 LUT under `outputs/rl/v1_9_hybrid_tch_ppo_smoke`.
5. Done: Reran canonical environment smoke for all fixed scenarios under `outputs/env/env_smoke_*_20260618_234452`.
6. Done: Reran formal multi-seed algorithm comparison under `outputs/rl/v1_9_formal_hybrid_tch_ppo_20260618`.
7. Done: VQA thread added the calibrated semantic utility model with CI, uncertainty, and SNR monotonic sanity checks.
8. Done: Algorithm-facing environment wrapper queries the semantic utility interface when `outputs/lut/v1_9_semantic_utility_with_ci.csv` is available and exposes conservative `accuracy_lcb`, uncertainty, and sample count in `info`/CSV records.
9. Done: Added the first cache-collapse fix for PPO/TCH-PPO: reward scaling through semantic LCB, entropy schedule, decaying service-level imitation prior, oracle warm-start, service-dependent resource floors, semantic feasibility projection, and UTM/DSS/off-nominal cost hooks.
10. Done: Paper-ready mainline documentation defines Conservative VQA-grounded Semantic-Lyapunov Hybrid Control as the paper-level problem/algorithm direction.
11. Done: Algorithm-facing environment wrapper exposes the paper-ready semantic info fields, including `semantic_payload_kb`, `semantic_quality_gap`, and Lyapunov queue state fields.
12. Done: Algorithm thread implemented explicit Semantic-Lyapunov queue states and added without-LCB/without-queue/without-projection ablation switches.
13. Ongoing: All threads update docs/current_status.md and docs/experiment_todo.md before ending work.

## Paper Benchmark Protocol TODO

Completed 2026-06-22 Asia/Macau:

- Added `docs/paper_algorithm_outline.md`.
- Updated `docs/benchmark_protocol.md` around five benchmark scenarios:
  - `nominal_patrol`
  - `disaster_hotspot`
  - `low_snr_blockage`
  - `edge_overload`
  - `utm_conflict`
- Checked consistency across `docs/interfaces.md`, `docs/formal_problem_definition.md`, and `docs/twc_algorithm_plan.md`:
  - semantic QoS: `accuracy_lcb >= epsilon_k`
  - quality gap: `semantic_quality_gap = max(0, epsilon_k - accuracy_lcb)`
  - service levels: `s=0` cache answer, `s=1` semantic token / compact evidence, `s=2` image evidence.

Thread responsibilities:

1. Environment thread owns scenario presets and environment-only smoke artifacts for the five benchmark scenarios.
2. Algorithm thread owns scenario benchmark runs, ablations, and policy summaries under `outputs/rl`, `outputs/hppo`, or `runs`.
3. VQA/semantic utility controller thread owns utility calibration, summary tables, and paper narrative synthesis.

Benchmark ablations to prepare:

- without LCB,
- without Semantic-Lyapunov queue,
- without resource/action projection,
- without semantic tokens,
- greedy and oracle semantic-utility baselines,
- fixed-service baselines.

## RL Cache-Collapse Follow-Up

Completed 2026-06-22 Asia/Shanghai:

1. Updated `src/vqa_semcom/rl/v19_resource_env.py` to consume `SemanticUtilityModel.U_sem(...)` without changing the LUT schema.
2. Updated `src/vqa_semcom/rl/v19_ppo.py` with conservative semantic utility reward, risk-aware constrained costs, entropy/prior schedules, oracle imitation warm-start support, service-dependent resource floors, and semantic safety projection.
3. Preserved realistic UTM extension points for operational intent conflict, DSS/notification delay, and off-nominal planning penalty.
4. Updated `scripts/run_v1_9_resource_alloc.py` with tuning flags and summary columns for accuracy LCB/mean, uncertainty, and UTM violation rate.
5. Added regression checks in `tests/test_v1_9_rl_resource_alloc.py`.
6. Passed `/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests` with 67 tests OK.
7. Ran small validation under `outputs/rl/v1_9_rl_fix_cache_collapse/`; PPO no longer collapses to all cache and achieved success 0.188 with lower delay/energy/payload than `greedy_min_sufficient_evidence` in the pilot.

Next algorithm steps:

1. Rerun formal multi-seed comparison after reviewing the pilot traces: service-only PPO, hybrid PPO no Lagrangian, hybrid TCH/Lagrangian PPO, and proposed semantic cognitive RL.
2. Include fixed unseen scenarios: `conflict-heavy`, `interference-heavy`, `cache-heavy`, and `mobility-stress`.
3. Report `ppo_training_trace.csv` non-cache ratio, entropy schedule, service prior, semantic LCB, uncertainty, and lambda trace alongside success/accuracy/delay/energy/payload.
4. Tune deadline feasibility separately if formal runs still show high deadline violation for evidence-rich actions.

## Semantic-Lyapunov Control Follow-Up

Completed 2026-06-22 Asia/Shanghai:

1. Added virtual queues to `src/vqa_semcom/rl/v19_resource_env.py`: `Q_quality`, `Q_deadline`, `Q_energy`, `Q_risk`, and `Q_utm`.
2. Appended queue state to the numeric observation vector and wrote queue state into rollout records.
3. Added semantic rollout fields: `semantic_payload_kb` and `semantic_quality_gap`.
4. Added drift-plus-penalty reward support in `src/vqa_semcom/rl/v19_ppo.py` with semantic LCB reward, resource costs, queue-weighted penalties, and uncertainty penalty.
5. Kept PPO as high-level service/evidence/UAV routing while retaining resource floors/projection for bandwidth/power/cpu/gpu.
6. Added baselines/ablations in the runner: `semantic_lcb_greedy`, `lyapunov_greedy`, `ppo_without_lcb`, `ppo_without_queues`, and `ppo_without_projection`.
7. Verified:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 10 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --smoke --policy ppo --episodes 1 --train-episodes 2 --tasks-per-episode 4 \
  --output-dir outputs/rl/twc_sem_lcb_lyapunov_smoke
```

New artifacts:

```text
outputs/rl/twc_sem_lcb_lyapunov_smoke/v1_9_resource_alloc_results.csv
outputs/rl/twc_sem_lcb_lyapunov_smoke/v1_9_resource_alloc_summary.md
outputs/rl/twc_sem_lcb_lyapunov_smoke/ppo_training_trace.csv
outputs/rl/twc_sem_lcb_lyapunov_smoke/ppo_lambda_trace.csv
```

Next algorithm steps:

1. Run the formal Sem-Lyapunov comparison under `outputs/rl/twc_sem_lcb_lyapunov_formal/` after reviewing the smoke traces.
2. Use the formal runner to compare heuristic, semantic LCB greedy, Lyapunov greedy, service-only PPO, hybrid PPO, Lagrangian PPO, and Sem-Lyapunov hybrid control.
3. Include ablation tables for no-LCB, no-queue, and no-projection variants.
4. Keep `.pt`, large rollout CSV, and `run.log` out of commits; retain only summary/report/trace CSV artifacts.

## Scenario-Aware Semantic Benchmark Follow-Up

Completed 2026-06-22 Asia/Shanghai:

1. Added paper scenario support to `scripts/run_v1_9_resource_alloc.py` through `--scenario`.
2. Added `--scenario-benchmark` smoke mode for:
   `nominal_patrol`, `disaster_hotspot`, `low_snr_blockage`, `edge_overload`, and `utm_conflict`.
3. Added benchmark policies:
   `always_cache`, `always_semantic_token`, `always_image`, `semantic_greedy`, `lyapunov_greedy`, and Semantic-LCB Lyapunov PPO.
4. Added ablation switches/aliases:
   `--no-lcb`, `--no-lyapunov-queues`, `--no-projection`, and `--disable-semantic-token`.
5. Generated:

```text
outputs/rl/semantic_scenario_benchmark/scenario_comparison_summary.csv
outputs/rl/semantic_scenario_benchmark/scenario_comparison_report.md
outputs/rl/semantic_scenario_benchmark/<scenario>/v1_9_resource_alloc_summary.md
```

6. Verified:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 12 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_9_resource_alloc.py \
  --scenario-benchmark --smoke --episodes 1 --train-episodes 2 --tasks-per-episode 4 \
  --output-dir outputs/rl/semantic_scenario_benchmark
```

Next algorithm steps:

1. Promote the scenario benchmark from smoke to multi-seed formal runs after checking queue/deadline scaling.
2. Add the no-LCB/no-queue/no-projection/no-semantic-token ablations to the scenario table.
3. Use the scenario benchmark report as the paper-facing stress-test table template.

## Scenario Benchmark v2 Cache-Collapse Fix

Completed 2026-06-23 Asia/Shanghai:

1. Diagnosed PPO cache bias in `outputs/rl/semantic_scenario_benchmark/scenario_comparison_report.md`: cache minimized delay/energy/payload while semantic shortfall penalties were too weak.
2. Strengthened the proposed controller in `src/vqa_semcom/rl/v19_ppo.py`:
   - semantic success and LCB gap reward scaling,
   - explicit cache shortfall penalty,
   - high-epsilon/high-risk cache penalty,
   - semantic-token exploration bonus,
   - cache override inside semantic projection when `semantic_quality_gap` is large.
3. Confirmed `semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)` remains the standardized queue gap.
4. Confirmed resource projection behavior:
   - cache actions use minimal/no resource allocation,
   - semantic-token/image evidence keeps service-dependent bandwidth/power/cpu/gpu floors,
   - no-projection ablation remains available for comparison.
5. Ran formal-small scenario benchmark:

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

Generated artifacts:

```text
outputs/rl/semantic_scenario_benchmark_v2/scenario_comparison_all_seed_results.csv
outputs/rl/semantic_scenario_benchmark_v2/scenario_comparison_summary.csv
outputs/rl/semantic_scenario_benchmark_v2/scenario_comparison_report.md
outputs/rl/semantic_scenario_benchmark_v2/cache_collapse_analysis.md
```

Key observation:

- Proposed PPO no longer collapses to `always_cache` in any of the five scenarios.
- `low_snr_blockage`: proposed semantic success 0.785 vs cache 0.053 and semantic greedy 0.950, with 1.726 KB payload.
- `nominal_patrol`: proposed semantic success 0.303 vs semantic greedy 0.087 under this small run, but payload/seed variance should be checked in longer runs.
- `edge_overload` and `utm_conflict`: semantic success remains near zero for most policies, but proposed PPO improves LCB/gap while keeping payload small.

Verified:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 13 tests OK
```

Next algorithm steps:

1. Run a paper-scale version of `semantic_scenario_benchmark_v2` with seeds 0-4, 200+ eval episodes, and 500-1000+ train episodes per PPO variant.
2. Add/commit only summary/report/small CSV artifacts; keep `.pt`, rollout CSV, and logs untracked.
3. Turn `scenario_comparison_summary.csv` into paper tables and plot semantic success/resource tradeoff curves.
4. Investigate `edge_overload` and `utm_conflict` feasibility: success is near zero for almost all methods, so these may need to be framed as LCB/gap robustness rather than success-rate wins.

## Semantic Scenario Utility Diagnostics

Completed 2026-06-23 Asia/Shanghai:

- Diagnosed the five Algorithm v2 scenario benchmark outputs against the calibrated semantic utility model.
- Did not rerun Qwen/VLM and did not change RL/environment code.
- Generated:

```text
outputs/reports/semantic_scenario_utility_diagnostics.md
outputs/reports/semantic_scenario_utility_diagnostics.csv
```

Key findings:

1. `edge_overload` has average proposed LCB around 0.691 but semantic success remains 0 because the rollout is almost entirely critical tasks and the effective epsilon is around 0.80-0.82; average quality gap remains about 0.121.
2. `utm_conflict` has average proposed LCB around 0.581 and semantic success 0 because the controller keeps a high cache ratio while the scenario requires mostly critical-level semantic reliability.
3. `disaster_hotspot` should not simply lower all critical epsilon values; use layered epsilon by task type and risk, especially separating presence from critical counting.
4. `low_snr_blockage` shows `semantic_greedy` outperforming proposed PPO, so the algorithm thread should improve token/image routing via semantic-token exploration prior or greedy/oracle distillation.
5. Cache needs stronger risk-aware shortfall penalties because zero payload and low delay can still attract the policy despite low conservative LCB.

Next coordination items:

1. Environment thread: persist `epsilon_k` directly in rollout records so future diagnostics do not need to reconstruct failed-row epsilon from `accuracy_lcb + semantic_quality_gap`.
2. Algorithm thread: add risk-aware cache penalties and token exploration/distillation for low-SNR and edge-overload cases.
3. Total-control thread: use the diagnostics report to decide whether `edge_overload` and `utm_conflict` should be framed as robustness/gap-reduction scenarios or tuned into feasible success-rate scenarios.

## Scenario Benchmark v3 Risk-Aware Routing

Completed 2026-06-23 Asia/Shanghai:

1. Integrated the VQA semantic-utility diagnostics into the Algorithm runner without changing the LUT schema or environment dynamics.
2. Persisted `epsilon_k` in the V1.9 rollout record path so future diagnostics can directly audit quality thresholds.
3. Strengthened proposed PPO routing in `src/vqa_semcom/rl/v19_ppo.py`:
   - risk/staleness/UTM-aware cache shortfall penalties,
   - higher semantic-token prior and `semantic_greedy` distillation weight,
   - compute-aware projection that prefers semantic-token evidence over cache when token evidence reduces LCB shortfall under deadline or edge pressure,
   - UTM/airspace conflicts are recorded as risk/queue costs rather than silently hidden by a cache fallback.
4. Added epsilon statistics to scenario summaries:
   average epsilon, failed-task epsilon mean, and failed-task epsilon range.
5. Ran the v3 scenario benchmark after the Environment background operational-intent fix:

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

Generated artifacts:

```text
outputs/rl/semantic_scenario_benchmark_v3/scenario_comparison_all_seed_results.csv
outputs/rl/semantic_scenario_benchmark_v3/scenario_comparison_summary.csv
outputs/rl/semantic_scenario_benchmark_v3/scenario_comparison_report.md
outputs/rl/semantic_scenario_benchmark_v3/cache_collapse_analysis.md
```

Key observation:

- Proposed PPO avoids cache collapse more aggressively than v2: cache ratio is 0.000 in `edge_overload`, 0.087 in `utm_conflict`, and below 0.25 in all scenarios except `low_snr_blockage` where cache remains a minority fallback.
- `low_snr_blockage` remains the strongest success case: proposed PPO reaches 0.786 semantic success with 0.756 accuracy LCB and 1.748 KB payload, still below `semantic_greedy` and worth longer training/distillation.
- `edge_overload` becomes token-only under proposed PPO; semantic success is still low because deadline/edge pressure dominates, not because of cache collapse.
- `utm_conflict` now exposes the expected quality-vs-UTM tradeoff after the background operational-intent fix: proposed PPO improves LCB/gap over cache but pays a high UTM conflict rate that should be treated as a constrained-control target in the next paper run.

Verified:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests -p 'test_v1_9*.py'
# Ran 13 tests OK

/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
# Ran 78 tests OK
```

Next algorithm steps:

1. Decide whether `utm_conflict` should prioritize conflict-free evidence routing or semantic-gap reduction under explicit UTM queue constraints.
2. Run a longer paper-scale v3/v4 benchmark with seeds 0-4, 200+ evaluation episodes, and 500-1000+ training episodes after the UTM objective is finalized.
3. Add paper plots for semantic success/resource tradeoff and epsilon-conditioned failure analysis.
4. Keep `.pt`, rollout CSV, and run logs out of commits; retain only summary/report/small CSV artifacts.

## Output Naming Convention

Use explicit run names:

```text
outputs/rl/hppo_v1_9_lut_seed{seed}_YYYYMMDD_HHMMSS/
outputs/env/env_smoke_YYYYMMDD_HHMMSS/
outputs/sim/v1_9_snr_resource_results.csv
outputs/reports/v1_9_snr_eval_report.md
```

Never write generic temporary outputs into the root directory.
