# Formal Semantic Network Scenario Specs

This file is generated from `src/vqa_semcom/sim/multi_uav_env.py` and is owned by the environment thread.

## Network Architecture Layers

- `task_layer`: VQA task, risk level, deadline, semantic accuracy requirement
- `semantic_service_layer`: cache answer, semantic tokens, raw image evidence
- `semantic_utility_layer`: LUT answer accuracy, payload-aware utility, constraint-aware reward
- `network_layer`: UAV mobility, A2G SINR/rate, bandwidth, power, edge CPU/GPU
- `cognitive_control_layer`: RL controller, heuristic baseline, hybrid action projection

## Semantic Service Levels

- `s=0` `cache_answer`: cached semantic answer (enabled when present in LUT/config)
- `s=1` `semantic_tokens`: detector tags/boxes/tokens (enabled when present in LUT/config)
- `s=2` `raw_image_evidence`: full image evidence for VQA/VLM (enabled when present in LUT/config)
- `s=3` `roi_crop_image`: reserved ROI/crop image evidence (reserved/disabled by default)

## Formal Train/Test Scenarios

| name | split | base scenario | description |
|---|---|---|---|
| `train_nominal` | train | nominal | Stable training scenario with calibrated nominal task arrivals. |
| `train_mixed_random` | train | nominal+cache-heavy+interference-heavy+mobility-stress | Training scenario that samples a known stressor per episode. |
| `test_conflict_heavy` | test | conflict-heavy | Held-out airspace conflict stress test. |
| `test_interference_heavy` | test | interference-heavy | Held-out multi-UAV same-band interference stress test. |
| `test_cache_heavy` | test | cache-heavy | Held-out semantic cache reuse/freshness stress test. |
| `test_mobility_stress` | test | mobility-stress | Held-out long-range mobility and battery stress test. |
| `test_unseen_mixed` | test | conflict-heavy+interference-heavy+cache-heavy+mobility-stress | Unseen mixture with larger network size and heavier arrivals than training. |

## Scalability Presets

### uav_count

- `M2`: `{'num_uavs': 2}`
- `M4`: `{'num_uavs': 4}`
- `M6`: `{'num_uavs': 6}`
- `M8`: `{'num_uavs': 8}`

### task_arrival

- `low`: `{'tasks_per_episode': 12, 'episode_steps': 12}`
- `medium`: `{'tasks_per_episode': 24, 'episode_steps': 12}`
- `high`: `{'tasks_per_episode': 40, 'episode_steps': 16}`

### edge_load

- `light`: `{'edge_load_range': [0.05, 0.2], 'gpu_load_range': [0.03, 0.16]}`
- `medium`: `{'edge_load_range': [0.2, 0.45], 'gpu_load_range': [0.15, 0.38]}`
- `heavy`: `{'edge_load_range': [0.45, 0.8], 'gpu_load_range': [0.38, 0.72]}`

## Graph Observation Schema

- node sets: `uav`, `task`, `edge`
- edge sets: `uav_task_link`, `task_edge_compute`
- schema is available at runtime via `env.graph_observation_schema()` and observations include `obs['graph']`.

