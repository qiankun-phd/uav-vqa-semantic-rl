# Semantic Network Benchmark Protocol

Last updated: 2026-06-22 Asia/Shanghai

This protocol defines the paper-facing benchmark for the UAV-driven VQA semantic communication network simulator. The canonical simulator is `src/vqa_semcom/sim/multi_uav_env.py`; algorithm code should consume this environment rather than duplicate its dynamics.

## Scope

The benchmark evaluates a controller that jointly selects semantic service routing, UAV assignment, communication resources, edge resources, semantic cache usage, and UTM-style operational-intent behavior.

V1.9 service levels are:

| service level | name | evidence | status |
|---:|---|---|---|
| 0 | `cache_answer` | cached semantic answer | enabled |
| 1 | `semantic_tokens` | detector tags/boxes/tokens | enabled |
| 2 | `raw_image_evidence` | full image evidence for VLM/VQA inference | enabled |
| 3 | `roi_crop_image` | ROI/crop evidence | disabled until ROI/crop LUT exists |

VQA answer accuracy and payload are provided by the V1.9 LUT:

```text
A_k = LUT[question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level]
D_k = LUT_payload[question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level]
```

No online VLM/Qwen inference is run inside benchmark episodes.

## Train/Test Scenario Split

Training scenarios:

| scenario | purpose |
|---|---|
| `train_nominal` | calibrated nominal task arrivals and resource states |
| `train_mixed_random` | samples known stressors from nominal/cache/interference/mobility settings |

Held-out semantic-network test scenarios:

| scenario | stressor |
|---|---|
| `test_conflict_heavy` | overlapping Area4D operational volumes and airspace conflict penalties |
| `test_interference_heavy` | concurrent same-band multi-UAV uploads and SINR degradation |
| `test_cache_heavy` | repeated areas, semantic cache reuse, and freshness effects |
| `test_mobility_stress` | sparse disaster areas, longer flight legs, tighter battery reserve |
| `test_unseen_mixed` | larger unseen mixture of conflict, interference, cache, and mobility stressors |

UTM realistic flight-test scenarios:

| scenario | InterUSS/UTM concept |
|---|---|
| `test_utm_nominal_planning` | accepted-to-activated operational intent flow with available DSS |
| `test_utm_off_nominal_planning` | nonconforming operations under mobility/battery pressure |
| `test_utm_intent_conflict` | buffered strategic conflict detection over 4D operational intents |
| `test_utm_dss_outage` | contingent operation when DSS coordination is unavailable |
| `test_utm_notification_delay` | subscription/notification delay for conflict updates |

The UTM scenarios are benchmark/test scenarios, not default PPO training scenarios, unless a specific experiment intentionally trains on UTM stressors.

## Scalability Dimensions

Scalability is evaluated by varying three independent dimensions.

UAV count:

| profile | value |
|---|---:|
| `M2` | 2 UAVs |
| `M4` | 4 UAVs |
| `M6` | 6 UAVs |
| `M8` | 8 UAVs |

Task arrival:

| profile | setting |
|---|---|
| `low` | 12 tasks, 12 slots |
| `medium` | 24 tasks, 12 slots |
| `high` | 40 tasks, 16 slots |

Edge load:

| profile | CPU/GPU load range |
|---|---|
| `light` | low CPU/GPU background load |
| `medium` | moderate CPU/GPU background load |
| `heavy` | high CPU/GPU background load |

Paper experiments should report the default configuration plus either the compact scalability set or the full Cartesian sweep, depending on compute budget.

## Metrics

Primary task metrics:

- `success_rate`: mean of `info["success"]`.
- `answer_accuracy_est`: LUT-estimated VQA correctness.
- `quality_satisfaction`: `1 - mean(quality_violation)`.
- `deadline_satisfaction`: `1 - mean(deadline_violation)`.
- `completion_rate`: fraction of completed tasks in multi-slot rollouts when task-level accounting is used.

Resource and cost metrics:

- `delay_s`: total task delay including fly, sense, tx, queue, infer, load, DSS, and notification components.
- `energy_j`: total energy including flight, hover/sense, tx, and compute components.
- `payload_kb`: semantic evidence payload from the LUT.
- `semantic_utility`: task-oriented utility exposed by the simulator.
- `semantic_efficiency`: semantic gain per payload unit.

Violation metrics:

- `quality_violation`
- `deadline_violation`
- `resource_violation`
- `battery_violation`
- `airspace_conflict`
- `utm_constraint_violation`

Communication metrics:

- `sensed_snr_db`, `sinr_db`, `snr_bin`
- `rate_mbps`
- `distance_3d_m`, `elevation_deg`
- `los_probability`, `path_loss_db`, `interference_dbm`

MEC/cache metrics:

- `queue_delay_s`, `infer_delay_s`, `load_delay_s`
- `gpu_memory_ok`, `gpu_memory_used_mb`, `gpu_memory_capacity_mb`
- `cache_hit_probability`, `semantic_cache_hit`

UAV/UTM metrics:

- `fly_delay_s`, `fly_energy_j`, `battery_remaining_j`
- `operational_intent_state`
- `strategic_conflict_count`
- `dss_available`, `dss_delay_s`
- `subscription_notification_delay_s`, `conflict_notification_pending`

Service-level distribution should be reported from `service_level` counts, especially to show cache/tokens/image usage and to confirm `s=3` is absent unless ROI LUT support is explicitly enabled.

## Benchmark Artifacts

Commit these environment-owned benchmark artifacts when they are intentionally refreshed:

```text
docs/benchmark_protocol.md
docs/semantic_network_architecture.md
docs/formal_problem_definition.md
docs/interuss_realistic_flight_mapping.md
outputs/env/formal_scenario_specs.md
outputs/env/benchmark_protocol_smoke.csv
outputs/env/utm_realistic_scenario_smoke.csv
outputs/env/interuss_mapping_summary.md
```

`outputs/env/formal_scenario_specs.md` is generated from `multi_uav_env.py` and should be committed when scenario definitions change. `outputs/env/benchmark_protocol_smoke.csv` is the fixed-name smoke artifact for this protocol.

Do not commit transient run artifacts:

```text
outputs/env/scenario_smoke_*.csv
outputs/env/env_smoke_*/
outputs/rl/
outputs/hppo/
runs/
__pycache__/
*.pyc
```

Do not overwrite VQA LUT/report artifacts from environment benchmark runs:

```text
outputs/lut/
outputs/reports/
outputs/sim/v1_9_*
```

## Lightweight Protocol Smoke

Use this smoke for protocol validation:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_semantic_network_benchmark.py \
  --config configs/v1_9_snr_lut.yaml \
  --steps 5 \
  --seed 42 \
  --output-dir outputs/env \
  --formal-scenarios train_nominal,test_conflict_heavy,test_interference_heavy,test_cache_heavy,test_mobility_stress,test_utm_intent_conflict,test_utm_dss_outage,test_utm_notification_delay
```

After the script writes a timestamped `outputs/env/scenario_smoke_*.csv`, copy that CSV to:

```text
outputs/env/benchmark_protocol_smoke.csv
```

The timestamped CSV is a temporary run product and should be removed before committing.

## Reproducibility Checklist

Before committing benchmark protocol changes:

1. Regenerate `outputs/env/formal_scenario_specs.md`.
2. Regenerate `outputs/env/benchmark_protocol_smoke.csv`.
3. Remove timestamped `outputs/env/scenario_smoke_*.csv` files.
4. Run `/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests`.
5. Confirm `git status --short` only includes intentional protocol/docs/artifact changes and unrelated thread outputs are left unstaged.
