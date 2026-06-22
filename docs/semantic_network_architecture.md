# UAV-Driven Semantic Communication Network Architecture

Last updated: 2026-06-19 Asia/Macau

This project treats the canonical simulator in `src/vqa_semcom/sim/multi_uav_env.py` as a UAV-driven semantic communication network simulator and benchmark, not only as an RL environment.

## Layered Architecture

### Task Layer

The task layer models emergency VQA service requests:

- `task_id`, `question`, `question_type`
- operational `Area4D`
- `risk_level`
- deadline `tau_k`
- required semantic accuracy `epsilon_k`
- priority `rho_k`
- generation time and cache age
- view quality and freshness bins

### Semantic Service Layer

The service layer routes each task to one of the active semantic services:

| service level | name | evidence |
|---:|---|---|
| 0 | `cache_answer` | cached semantic answer |
| 1 | `semantic_tokens` | detector tags, boxes, and lightweight semantic tokens |
| 2 | `raw_image_evidence` | raw/full image evidence for VLM/VQA inference |
| 3 | `roi_crop_image` | reserved ROI/crop evidence, disabled until ROI LUT exists |

V1.9 enables only `s=0,1,2`.

### Semantic Utility Layer

The semantic utility layer converts a service decision into task-oriented value:

```text
A_k = LUT[question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level]
D_k = LUT payload bytes
```

The simulator exposes:

- `semantic_service_route(...)`
- `semantic_utility(...)`
- `semantic_utility_schema()`

The utility API keeps semantic gain, delay factor, quality shortfall, resource cost, semantic utility, and semantic efficiency inspectable in `info`.

### Network Layer

The network layer models:

- multiple UAVs with position, altitude, speed, battery, utilization, and camera state
- VQA task queues over disaster areas/grids
- A2G LoS/NLoS link budget, fading, SNR/SINR, rate, and interference
- bandwidth, transmit power, edge CPU share, and GPU share
- edge queue load, GPU load, model cache, GPU memory, and model-loading delay
- semantic cache freshness and replacement
- airspace conflict over overlapping `Area4D` operational intents
- delay decomposition: fly, sense, tx, queue, infer, load
- energy decomposition: flight, hover/sense, tx, compute

### Cognitive Control Layer

The cognitive control layer consumes the stable `reset/step` interface and may use:

- heuristic baselines
- PPO/HPPO/TCH-PPO controllers
- hybrid discrete-continuous actions
- future graph policies using `obs["graph"]`

This layer is intentionally separate from simulator dynamics. Algorithm code should consume the simulator, not duplicate it.

## Graph Observation Schema

The simulator exports a graph observation for future GNN policies:

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

The graph is a schema-stable export. Current PPO baselines may ignore it.

## Benchmark Ownership

Environment-owned benchmark outputs live under:

```text
outputs/env/
```

The simulator must not overwrite VQA LUT/report artifacts or algorithm outputs.

## Paper Scenario Library

The canonical simulator now exposes five paper-facing stress presets through `env.reset(options={"scenario": ...})`:

| preset | stress source |
|---|---|
| `nominal_patrol` | routine patrol, medium SNR, low UTM pressure |
| `disaster_hotspot` | clustered critical tasks, high `epsilon_k`, tight deadlines |
| `low_snr_blockage` | weak/blocked A2G link and low SINR |
| `edge_overload` | high CPU/GPU queue load and model-cache pressure |
| `utm_conflict` | overlapping 4D operational intent, DSS delay, and notification delay |

The detailed protocol is in `docs/scenario_presets.md`. These presets are intended for benchmark and ablation reporting; service level 3 ROI/crop remains disabled until ROI semantic utility rows are available.
