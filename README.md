# UAV-VQA Semantic Communication RL

This repository contains a research prototype for UAV-driven VQA semantic
communication network optimization. The project studies how multiple UAVs can
serve visual-question-answering tasks by choosing semantic communication paths,
UAV mobility actions, edge resources, and admission decisions under semantic
QoS, deadline, energy, and UTM airspace constraints.

The active research line has moved beyond the original V0/V1 LUT-only pipeline.
The current system combines:

- VQA-grounded semantic utility modeling.
- A multi-UAV, edge-computing, A2G-channel, and UTM-aware simulator.
- Semantic path control over cache, semantic token, image evidence,
  cache update, defer, and reject/admission actions.
- Two-timescale PPO with Lyapunov-guided constraint queues, resource
  projection, expert warm-start, and bottleneck-aware path/mobility control.

The stable historical baseline remains on `main`. The current active branch is:

```bash
git checkout codex/semantic-path-cache-defer
```

## Research Objective

The target problem is not ordinary throughput maximization. A UAV serves a VQA
task successfully only when the answer is semantically reliable and the network
service is feasible:

```text
semantic_accuracy_lcb >= epsilon_k
delay <= tau_k
energy <= budget
no UTM / airspace violation
```

Each task carries semantic and network context such as:

```text
task_type, risk_level, epsilon_k, tau_k, view_quality_bin,
freshness_bin, sensed_snr_db, snr_bin, UAV state, edge load,
semantic cache state, and UTM risk.
```

High-risk tasks can require stricter semantic QoS. Cached answers can be fast,
but they are accepted only when freshness and conservative semantic quality are
sufficient.

## System Architecture

The current simulator and controller are organized as a UAV semantic
communication network:

```text
VQA task queue
  -> semantic utility lookup
  -> semantic path decision
  -> UAV mobility / A2G link / edge compute
  -> semantic cache and model cache update
  -> QoS, deadline, energy, and UTM feedback
  -> RL controller update
```

Main layers:

- Task layer: VQA tasks with risk, deadline, semantic accuracy requirement, and
  Area4D location.
- Semantic service layer: cache answer, semantic token, image evidence,
  cache update, defer, and reject/admission control.
- Semantic utility layer: calibrated VQA accuracy, conservative LCB, payload,
  uncertainty, and sparse-cell visibility.
- Network layer: UAV mobility, A2G SINR/rate, bandwidth, power, edge CPU/GPU,
  model cache, and semantic cache.
- UTM layer: operational intent, background conflicts, DSS delay, subscription
  notification delay, spatial/temporal buffers, and avoid-conflict mobility.
- Control layer: two-timescale PPO plus Lyapunov queues and action/resource
  projection.

## Semantic Paths

The controller selects one semantic path per served task:

| Path | Meaning | Typical benefit | Main risk |
|---|---|---|---|
| `cache` | Reuse a cached semantic answer | Zero visual payload and low delay | Stale or low-quality cache |
| `token` | Transmit compact semantic evidence | Low payload | Can still miss deadline under weak links |
| `image` | Transmit high-fidelity image evidence | Rich visual information | High payload and delay |
| `cache_update` | Serve with new evidence and refresh cache | Future reuse value | Extra current service cost |
| `defer` | Delay service for a future opportunity | Avoids bad immediate conditions | Deadline may expire |
| `reject` | Admission reject for infeasible tasks | Avoids unsafe or wasteful service | Does not count as task success |

The `reject` path is used for infeasibility-aware semantic admission control. A
correct reject is preferable to forcing a service action that will violate
deadline, energy, resource, or UTM constraints. A wrong reject is penalized when
a feasible service path exists.

## Semantic Utility Interface

The control-facing semantic utility model is built from measured/calibrated VQA
predictions and exposed through:

```python
U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
# -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

Important conventions:

- RL and feasibility checks prefer `accuracy_lcb` for conservative semantic QoS.
- Cache service (`service_level=0`) is SNR-invariant because it does not transmit
  visual evidence.
- Token and image evidence depend on SNR, view quality, task type, freshness,
  and risk level.
- Sparse cells expose uncertainty and sample count rather than hiding low
  confidence.

Relevant files:

```text
src/vqa_semcom/semantic/utility.py
docs/semantic_utility_model.md
outputs/semantic/semantic_path_utility_diagnosis_20260624/report.md
```

## Multi-UAV Simulation Environment

The canonical environment is:

```text
src/vqa_semcom/sim/multi_uav_env.py
```

It models:

- Multiple UAVs with position, speed, battery, flight energy, and mobility modes.
- A2G links with path loss, LoS/NLoS, fading, interference, SINR, and rate.
- Edge computing with CPU/GPU load, queue delay, inference delay, model load
  delay, and model cache capacity.
- Semantic cache entries with spatial locality, freshness, quality, uncertainty,
  reuse count, and capacity.
- Task queue dynamics with `pending`, `served`, `deferred`, `expired`, and
  `rejected` states.
- UTM constraints with operational intent states, background conflicts, DSS
  delay, subscription notification delay, spatial/temporal buffers, and
  avoid-conflict mobility.

The environment exposes per-path and per-mobility diagnostics through fields
such as:

```text
candidate_path_metrics[path]
candidate_mobility_metrics[path][mobility_mode]
cache_eligible
reject_feasible
bottleneck_type
required_rate_mbps
required_bandwidth_hz
```

Cache eligibility is explicit. A task can use cache only when an exact or nearby
same-type semantic cache entry is available, freshness is acceptable, and
`cache_quality_lcb >= epsilon_k`.

## Scenarios

The project uses scenario-aware evaluation rather than one nominal setting.
Current scenario groups include:

| Scenario | Purpose |
|---|---|
| `normal_patrol` | Routine multi-UAV patrol with moderate load |
| `disaster_hotspot` | Clustered high-risk VQA burst |
| `low_snr_soft` | Calibrated weak-link scenario with feasible service region |
| `low_snr_blockage` | Hard weak-link/blockage stress test |
| `edge_overload_soft` | Calibrated edge-load stress with nonzero feasible service |
| `edge_overload` | Hard edge overload, often near-infeasible |
| `utm_conflict_soft` | Calibrated UTM conflict scenario with safe feasible actions |
| `utm_conflict` | Hard UTM conflict stress test |

Hard scenarios are intentionally retained as infeasibility stress tests. Soft
variants are used to evaluate learnable behavior without silently weakening the
hard scenarios.

## RL Methodology

The current RL implementation is in:

```text
src/vqa_semcom/rl/v19_resource_env.py
src/vqa_semcom/rl/v19_ppo.py
scripts/run_v1_9_resource_alloc.py
```

The controller follows a two-timescale structure:

```text
Slow timescale:
  UAV assignment, mobility mode, waypoint / avoid_conflict / stay

Fast timescale:
  semantic path, bandwidth, power, CPU share, GPU share
```

The PPO policy consumes semantic, network, cache, queue, bottleneck, and UTM
features. It includes:

- Semantic path head over cache/token/image/defer/cache_update/reject.
- Mobility and resource heads for UAV assignment and continuous resource control.
- Lyapunov-style virtual queues for quality, deadline, energy, UTM, defer, and
  cache staleness pressure.
- Resource/action projection after raw PPO action generation.
- Expert path selector and behavior-cloning warm start from candidate metrics.
- Bottleneck-aware fallback using `semantic_quality`, `tx_delay`, `queue_delay`,
  `mobility`, `utm`, `energy`, `resource`, and `expired` reasons.
- Reject/admission control for tasks with no feasible service path.

## Key Outputs

Important current reports:

```text
outputs/env/semantic_path_bottleneck_diagnosis_20260624/report.md
outputs/env/semantic_path_soft_reject_diagnosis_20260624/report.md
outputs/rl/semantic_path_cache_defer_fix4_20260624/fix4_comparison.md
outputs/rl/semantic_path_cache_defer_reject_fix5_20260624/fix5_comparison.md
outputs/semantic/semantic_path_utility_diagnosis_20260624/report.md
```

The latest reject/admission-control run reports the following qualitative state:

- `normal_patrol` and `low_snr_soft` remain stable without reject leakage.
- `edge_overload_soft` and `utm_conflict_soft` recover nonzero service compared
  with their hard counterparts.
- Hard `edge_overload` and hard `utm_conflict` are treated as near-infeasible
  stress tests where correct reject avoids unsafe service.
- Wrong reject is reported separately and should remain low.

See the comparison report for exact numbers and caveats.

## Running Tests

Use the remote conda environment:

```bash
cd /home/qiankun/phd_research/vqa_semcom
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
```

Some PPO tests also verify CUDA device selection when available. The RA_DI
conda environment is used for GPU training runs:

```bash
conda activate RA_DI
```

## Common Commands

### Semantic utility demo

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/demo_semantic_utility_query.py
```

### Environment feasibility diagnosis

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/diagnose_semantic_path_feasibility.py \
  --output-dir outputs/env/semantic_path_soft_reject_diagnosis_20260624
```

### Semantic path utility diagnosis

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/diagnose_semantic_path_utility.py \
  --output-dir outputs/semantic/semantic_path_utility_diagnosis_20260624
```

### Scenario-aware PPO benchmark

```bash
conda activate RA_DI
python scripts/run_v1_9_resource_alloc.py \
  --scenario-benchmark \
  --benchmark-scenarios normal_patrol,disaster_hotspot,low_snr_soft,low_snr_blockage,edge_overload,edge_overload_soft,utm_conflict,utm_conflict_soft \
  --benchmark-ppo-variants semantic_path_two_timescale_ppo \
  --train-ppo \
  --two-timescale-ppo \
  --semantic-path-ppo \
  --policy all \
  --train-episodes 300 \
  --episodes 50 \
  --tasks-per-episode 12 \
  --seeds 0,1,2 \
  --device cuda \
  --output-dir outputs/rl/semantic_path_cache_defer_reject_fix5_20260624
```

Command-line options evolve with the experiment script. Check the script help if
a flag changes:

```bash
python scripts/run_v1_9_resource_alloc.py --help
```

## Project Map

```text
configs/v1_9_snr_lut.yaml                  # Main V1.9 semantic utility config
src/vqa_semcom/semantic/utility.py          # VQA-grounded semantic utility
src/vqa_semcom/sim/multi_uav_env.py         # Multi-UAV semantic network environment
src/vqa_semcom/rl/v19_resource_env.py       # RL wrapper and observation bridge
src/vqa_semcom/rl/v19_ppo.py                # PPO, expert warm-start, queues, projection
scripts/run_v1_9_resource_alloc.py          # Scenario benchmarks and PPO training
scripts/diagnose_semantic_path_feasibility.py
scripts/diagnose_semantic_path_utility.py
docs/interfaces.md
docs/formal_problem_definition.md
docs/semantic_network_architecture.md
docs/benchmark_protocol.md
docs/current_status.md
docs/experiment_todo.md
```

## Legacy V0/V1 Pipeline

The original V0/V1 pipeline is still useful for traceability:

```text
VisDrone annotations -> VQA-style tasks -> semantic quality LUT -> resource simulation
```

Examples:

```bash
python scripts/build_v0_lut.py --config configs/v0.yaml --demo
python scripts/run_v0_sim.py --config configs/v0.yaml --episodes 10
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_vlm_eval.py --config configs/v1_qwen.yaml --limit-images 20 --evaluator qwen --resume
```

The current research branch uses the VQA-grounded V1.9 semantic utility layer and
multi-UAV semantic path control as the main line.

## Known Limitations

- Hard `edge_overload` and hard `utm_conflict` are near-infeasible stress tests;
  they should not be interpreted as ordinary service scenarios.
- Current results are research-progress artifacts, not final paper claims.
- Large per-run rollout directories and model checkpoints are intentionally not
  all tracked in git; summary reports and CSVs are tracked selectively.
- The reject path is an admission-control mechanism. It improves safety and
  infeasibility awareness, but it does not count as task success.
- Future work should separate final benchmark presets, ablation tables, and
  publication-quality figures from exploratory fix runs.
