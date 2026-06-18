# Interfaces

Last updated: 2026-06-19 Asia/Macau
Primary project root: /home/qiankun/phd_research/vqa_semcom

## Stable Environment API

The environment thread should expose a Gym-like API:

```python
obs = env.reset(seed=None, options=None)
obs, reward, done, info = env.step(action)
```

If using Gymnasium-style signatures, wrap them so algorithm code can call one stable adapter:

```python
obs, info = env.reset(seed=None, options=None)
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

## Observation Contract

Each observation must contain these fields or a documented numeric encoding of them:

| field | type | notes |
|---|---|---|
| task_type | str/int | At least presence, counting. Optional future risk tasks. |
| risk_level | str/int | normal or critical. |
| view_quality_bin | str/int | poor, medium, good. |
| freshness_bin | str/int | fresh, stale, expired. |
| sensed_snr_db | float | Continuous sensed SNR used by physical delay/rate model. |
| snr_bin | str/float/int | Nearest LUT SNR bin label/value. Current V1.9 report uses 0dB, 10dB, 20dB; config also lists -5, 0, 5, 10, 15, 20. |
| uav_state | dict/vector | UAV position, velocity/heading, battery, remaining compute or flight budget. |
| edge_load | float/vector | Edge/server compute congestion or queue load. |
| cache_state | dict/vector | Whether cache answer exists, age/freshness, cache confidence if available. |

Recommended additional fields:

- deadline_s
- payload_kb_estimates_by_service
- task_id
- episode_step
- feasible_uavs
- scenario
- formal_scenario
- benchmark_split
- scalability_profile
- graph

## Semantic Network Observation Extension

The canonical simulator is also a semantic communication network benchmark. Observations may include:

| field | type | notes |
|---|---|---|
| network_layers | dict | task, semantic service, semantic utility, network, and cognitive control layers. |
| graph | dict | Graph observation export with UAV/task/edge node sets and link/compute edge sets. |
| formal_scenario | str | One of the formal train/test scenarios, if selected. |
| scalability_profile | dict | Selected UAV count, task arrival, and edge load presets. |

Graph schema:

```python
graph = obs["graph"]
schema = env.graph_observation_schema()
```

Node sets:

- `uav`
- `task`
- `edge`

Edge sets:

- `uav_task_link`
- `task_edge_compute`

## Action Contract

Algorithm thread outputs an action with these fields:

| field | type | notes |
|---|---|---|
| service_level | int | 0 cache answer, 1 detector semantic tokens, 2 raw image evidence. Service 3 ROI is reserved and currently not active in V1.9. |
| bandwidth | float | Hz or normalized share; unit must be written in info. |
| power | float | Watts or normalized share; unit must be written in info. |
| cpu_share | float | 0..1 normalized edge/local CPU share. |
| gpu_share | float | 0..1 normalized GPU share, if used. |
| uav_assignment | int/list | UAV id or task-to-UAV mapping. |
| waypoint | list/array | Optional next UAV waypoint or movement command. |

Minimal V1.9-compatible action:

```python
action = {
    'service_level': 0,
    'bandwidth': 1_000_000.0,
    'power': 0.1,
    'cpu_share': 0.0,
    'gpu_share': 0.0,
    'uav_assignment': 0,
    'waypoint': None,
}
```

## LUT Query Contract

The VQA/SNR-LUT thread provides an empirical answer-quality lookup table:

```text
A_k = LUT[question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level]
```

Current LUT path:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_snr_semantic_quality_lut.csv
```

Current report path:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/reports/v1_9_snr_eval_report.md
```

Nearest-bin helper exists in:

```text
/home/qiankun/phd_research/vqa_semcom/src/vqa_semcom/snr.py
```

Relevant helper functions:

```python
nearest_snr_bin(sensed_snr_db, snr_bins_db)
snr_db_to_bin_label(sensed_snr_db, snr_bins_db)
channel_bin_from_snr(snr_db)
```

## Reward Contract

Algorithm and environment threads should keep reward components inspectable through info:

```python
info = {
    'answer_accuracy_est': A_k,
    'success': bool,
    'delay_s': float,
    'energy_j': float,
    'payload_kb': float,
    'quality_violation': bool,
    'deadline_violation': bool,
    'snr_bin': str,
    'service_level': int,
}
```

Extended semantic-network `info` fields:

- `semantic_service_name`
- `semantic_evidence_type`
- `semantic_utility`
- `semantic_efficiency`
- `formal_scenario`
- `benchmark_split`

The semantic utility API is available through:

```python
env.semantic_service_route(...)
env.semantic_utility(...)
env.semantic_utility_schema()
```

Suggested scalar reward:

```text
reward = w_success * success
       - w_delay * normalized_delay
       - w_energy * normalized_energy
       - w_payload * normalized_payload
       - w_violation * (quality_violation + deadline_violation)
```

Do not hide the components; record them in CSV for later paper tables.

## Current V1.9 Policy Baseline Fields

The resource simulation summary currently reports:

- success
- accuracy
- delay
- energy
- payload KB
- payload reduction
- quality violation
- deadline violation
- service-level selection ratio

Algorithm experiments should output the same metrics so they can be compared directly against:

- always_cache
- always_light
- always_image
- greedy_min_sufficient_evidence
- no_cache_greedy
- no_semantic_tokens_greedy
- oracle_best_feasible_evidence

## Formal Benchmark Scenarios

Train scenarios:

- `train_nominal`
- `train_mixed_random`

Test scenarios:

- `test_conflict_heavy`
- `test_interference_heavy`
- `test_cache_heavy`
- `test_mobility_stress`
- `test_unseen_mixed`

Scalability presets:

- UAV count: `M2`, `M4`, `M6`, `M8`
- task arrival: `low`, `medium`, `high`
- edge load: `light`, `medium`, `heavy`

Environment-owned benchmark outputs:

```text
outputs/env/formal_scenario_specs.md
outputs/env/scenario_smoke_*.csv
```
