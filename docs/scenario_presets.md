# UAV Semantic Scenario Presets

Last updated: 2026-06-23 Asia/Shanghai

This document defines the paper-facing scenario presets for the canonical UAV-driven semantic VQA network simulator:

```text
src/vqa_semcom/sim/multi_uav_env.py
```

Each preset is available through:

```python
obs = env.reset(options={"scenario": "<preset_name>"})
```

The presets are environment-owned benchmark scenarios. They should write only small summaries/traces under `outputs/env/` and must not overwrite VQA LUT/report artifacts or algorithm outputs under `outputs/rl/`.

## Service Levels

V1.9 keeps the active semantic service set fixed to:

| level | service | evidence |
|---:|---|---|
| 0 | cache answer | cached semantic answer reuse |
| 1 | semantic token / compact evidence | detector tags, boxes, and compact tokens |
| 2 | image evidence | full image evidence for VLM/VQA inference |

Service level 3 ROI/crop remains disabled until the VQA thread provides ROI/crop semantic utility rows.

## Preset Summary

| preset | purpose | dominant stressor | expected algorithm pressure |
|---|---|---|---|
| `nominal_patrol` | normal inspection | medium SNR, low UTM conflict, medium deadline/epsilon | balanced semantic routing and resource use |
| `disaster_hotspot` | emergency burst | critical task concentration, stricter `epsilon_k`, tighter deadlines | priority/risk-aware preemption and conservative semantic QoS |
| `low_snr_blockage` | blocked A2G link | high path loss, low SNR/SINR, lower bandwidth | semantic tokens/cache should be more robust than large image payloads |
| `edge_overload` | MEC pressure | high CPU/GPU load, queue delay, model-cache pressure | resource projection and queue-aware service selection |
| `utm_conflict` | airspace coordination | overlapping 4D volumes, DSS delay, notification delay, intent state changes | UTM/risk cost handling and conflict-aware routing |

## Scenario Details

### `nominal_patrol`

Routine multi-UAV patrol with staggered task arrivals across several areas. It uses medium A2G settings, low interference overlap, low UTM buffers, medium semantic thresholds, and moderate edge load.

Use this preset as the main sanity check for nominal task completion, semantic accuracy, delay, energy, and payload.

### `disaster_hotspot`

Clustered burst where most tasks are critical. The preset forces high-risk tasks into the same hotspot area, raises `epsilon_k` for critical tasks, and tightens deadlines.

Use this preset to test risk-aware semantic QoS and whether high-priority VQA tasks are protected under resource pressure.

### `low_snr_blockage`

Large-area patrol with strong excess path loss, higher NLoS loss, slower fading recovery, and lower bandwidth. A small semantic cache seed is included to make cache/token choices meaningful.

Use this preset to validate the claim that task-oriented semantic evidence can remain useful when transmitting full image evidence is slow or unreliable.

### `edge_overload`

High edge CPU/GPU load, larger queue delay constants, single model-cache slot, and reduced GPU memory capacity. The channel is not the primary stressor.

The preset caps semantic thresholds more gently than `disaster_hotspot` and uses moderate task spacing, high-but-not-saturated edge load, and calibrated queue/model-load delay constants. This keeps the benchmark focused on edge queue/resource pressure instead of turning every semantic-token route into a deadline failure. In the latest lightweight V1.9 probe, the token route keeps deadline violation near 0.58 and semantic success near 0.75, so the scenario remains stressful while retaining a meaningful feasible region.

Use this preset to test MEC-aware semantic routing and Lyapunov/resource-projection behavior.

### `utm_conflict`

Burst tasks include a mix of overlapping and separable operational areas. The preset enables UTM-style flight-intent validation with DSS delay, subscription notification delay, spatial/altitude/temporal buffers, and observable intent states:

```text
accepted / activated / nonconforming / contingent
```

Use this preset to test UTM/risk costs and conflict-aware task/UAV matching. Cache-only service does not create an operational intent and should not trigger an airspace conflict by itself.

The preset enables background operational-intent conflict detection. Background intent density and 4D buffers are calibrated to produce moderate conflict pressure rather than all-conflict behavior. In the latest lightweight V1.9 probe, token-style actions produce UTM conflict near 0.50. Cache-only service still creates no operational intent and remains conflict-free.

## Required Info Fields

Every preset smoke must expose the following fields in `info`:

```text
semantic_accuracy_lcb
semantic_quality_gap
epsilon_k
deadline_s
energy_j
utm_delay_s
utm_conflict_violation
risk_violation
airspace_state
```

The semantic quality gap is defined as:

```text
semantic_quality_gap = max(0, epsilon_k - semantic_accuracy_lcb)
```

This non-negative quantity is the recommended input to semantic QoS queues.

## Smoke Artifacts

The preset smoke summary is stored under:

```text
outputs/env/semantic_scenario_presets/
```

Committed benchmark artifacts should be limited to small summary/report/CSV files, for example:

```text
outputs/env/semantic_scenario_presets/scenario_preset_summary.csv
outputs/env/semantic_scenario_presets/summary.md
```

Do not commit `.pt` checkpoints, large rollouts, `run.log`, temporary debugger output, or algorithm-owned `outputs/rl/*` artifacts from this environment task.
