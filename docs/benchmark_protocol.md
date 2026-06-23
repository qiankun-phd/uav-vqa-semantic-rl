# UAV Semantic Communication Benchmark Protocol

Last updated: 2026-06-22 Asia/Shanghai

This protocol defines the paper-ready benchmark for the UAV-assisted semantic VQA emergency network. The goal is to evaluate semantic communication control under sensing, communication, edge-computing, cache, mobility, and UTM/risk pressure.

The benchmark should support a journal-level wireless and edge intelligence study. It should not be framed as a generic PPO benchmark or as a target for one specific venue.

## System Under Test

The network consists of:

- multiple UAVs that collect or reuse visual semantic evidence,
- emergency VQA tasks generated over disaster/inspection regions,
- wireless UAV-edge links with sensed SNR and payload-dependent delay,
- edge inference and queueing resources,
- semantic cache state and freshness,
- UTM/airspace constraints for realistic UAV operations.

The service levels are:

| service level | semantic name | transmitted/reused evidence |
|---:|---|---|
| 0 | cache answer | cached answer or cached semantic result |
| 1 | semantic token / compact evidence | detector tags, boxes, counts, and compact evidence |
| 2 | image evidence | raw/full visual evidence for VQA inference |

Service level 3, ROI/crop evidence, is reserved until calibrated semantic utility rows exist for that service.

## Semantic Utility Contract

Task quality is defined by VQA-grounded semantic reliability:

```text
U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
  -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

The semantic QoS condition is:

```text
accuracy_lcb >= epsilon_k
```

The benchmark should report both expected accuracy and conservative LCB accuracy, but online conservative control should use `accuracy_lcb`.

The semantic quality gap is:

```text
semantic_quality_gap = max(0, epsilon_k - accuracy_lcb)
```

## Train/Test Scenario Split

Training should use easier or mixed versions of the scenario family:

| split | scenario preset | role |
|---|---|---|
| train | `nominal_patrol` | stable baseline policy learning |
| train | mixed sampled stressors | randomized mild SNR/cache/edge/mobility pressure |

Evaluation should use fixed, named stress scenarios:

```text
nominal_patrol
disaster_hotspot
low_snr_blockage
edge_overload
utm_conflict
```

All paper tables should report at least these five scenarios. Additional scalability sweeps can be layered on top.

## Scenario Matrix

### 1. `nominal_patrol`

Network pressure source:

- moderate task arrivals,
- mostly feasible UAV mobility,
- medium/good SNR,
- balanced edge load,
- normal cache freshness.

Why this is UAV semantic communication:

- UAVs still decide whether to reuse cache, send compact semantic evidence, or send image evidence.
- The objective is answer correctness under payload, delay, and energy limits rather than raw throughput.

Algorithm capability to verify:

- stable service-level routing,
- reasonable payload reduction compared with always-image,
- no unnecessary UTM or energy violations.

Main metrics:

- conservative VQA success rate,
- `semantic_accuracy_lcb`,
- average payload KB,
- delay and energy,
- service-level selection ratio.

Expected phenomenon:

- semantic tokens and cache should be selected often,
- greedy/oracle-style policies should reduce payload while keeping high success,
- learned controllers should not collapse to cache-only behavior.

### 2. `disaster_hotspot`

Network pressure source:

- bursty task arrivals around damaged/blocked regions,
- higher task priority and more critical tasks,
- higher UAV sensing demand,
- more repeated/nearby questions that can benefit from cache.

Why this is UAV semantic communication:

- many questions do not need full images if compact evidence is sufficient,
- emergency tasks require task-aware semantic evidence routing under deadlines.

Algorithm capability to verify:

- prioritization of critical VQA tasks,
- semantic cache exploitation without stale-answer overuse,
- service escalation from cache/tokens to image evidence when LCB is insufficient.

Main metrics:

- critical-task success rate,
- quality violation rate,
- cache hit/service ratio,
- semantic uncertainty,
- deadline violation rate.

Expected phenomenon:

- conservative policies should select more image evidence for critical low-confidence cells,
- semantic tokens should still reduce payload for presence/counting tasks,
- without LCB should overestimate risky or sparse cells.

### 3. `low_snr_blockage`

Network pressure source:

- poor sensed SNR caused by blockage, distance, or unfavorable geometry,
- lower transmission rate and stronger evidence degradation,
- larger payload penalty for image evidence.

Why this is UAV semantic communication:

- channel quality affects answer reliability through semantic utility, not only bit rate.
- the controller must decide whether semantic tokens, cache, or image evidence are worth transmitting at low SNR.

Algorithm capability to verify:

- SNR-aware semantic evidence routing,
- payload-delay tradeoff under poor links,
- robust conservative QoS using `accuracy_lcb`.

Main metrics:

- success vs SNR bin,
- payload reduction vs always-image,
- transmission delay,
- `semantic_quality_gap`,
- uncertainty-weighted failures.

Expected phenomenon:

- always-image becomes expensive and delay-heavy,
- semantic tokens may dominate when compact evidence is reliable,
- conservative LCB avoids over-selecting sparse high-mean cells.

### 4. `edge_overload`

Network pressure source:

- high edge CPU/GPU queue load,
- model/cache load pressure,
- inference delay increase,
- limited compute share for VQA processing.

Why this is UAV semantic communication:

- semantic evidence level changes edge workload: cache is cheap, tokens are moderate, images are heavy.
- resource allocation must jointly account for communication payload and edge inference cost.

Algorithm capability to verify:

- compute-aware semantic routing,
- CPU/GPU resource projection,
- deadline-aware degradation under edge congestion.

Main metrics:

- queue/inference delay,
- deadline violation rate,
- service-level ratio,
- energy and payload,
- conservative success under heavy edge load.

Expected phenomenon:

- image evidence should be used selectively,
- cache/tokens should reduce edge pressure,
- without projection should suffer resource violations or unstable delays.

### 5. `utm_conflict`

Network pressure source:

- overlapping UAV operational intents,
- conflict-heavy Area4D routing,
- DSS/notification delay or airspace coordination pressure,
- off-nominal mobility/risk states.

Why this is UAV semantic communication:

- UAVs must provide semantic VQA service while respecting airspace constraints.
- the semantic objective is coupled with mobility and operational intent risk.

Algorithm capability to verify:

- UTM/risk-aware task assignment and waypoint choices,
- safe service routing under airspace conflicts,
- Semantic-Lyapunov queue response to risk pressure.

Main metrics:

- UTM/risk violation rate,
- airspace conflict count,
- conservative VQA success,
- delay/energy under detours,
- task completion under conflict pressure.

Expected phenomenon:

- naive always-image or conflict-unaware policies incur higher delay/UTM risk,
- queue-aware hybrid control should trade some payload/route cost for lower risk,
- oracle/greedy baselines establish feasibility bounds.

## Scalability Dimensions

Scalability experiments should vary:

| dimension | values |
|---|---|
| UAV count | 2, 4, 6, 8 |
| task arrival | low, medium, high, bursty |
| edge load | light, medium, heavy |
| sensed SNR mix | good-dominant, balanced, blockage-heavy |
| critical-task ratio | low, medium, high |

At minimum, report UAV count, task arrival, and edge load sweeps for `nominal_patrol` and one stress scenario.

## Metrics Definition

Primary semantic metrics:

- `conservative_success_rate`: fraction of tasks with `accuracy_lcb >= epsilon_k`, deadline satisfied, and no hard UTM/resource violation.
- `semantic_accuracy_mean`: expected answer correctness from `U_sem`.
- `semantic_accuracy_lcb`: conservative correctness used for QoS.
- `semantic_quality_gap`: `max(0, epsilon_k - accuracy_lcb)`.
- `semantic_uncertainty`: CI/sparse-cell uncertainty.
- `semantic_sample_count`: measured samples behind each utility cell.

Communication/resource metrics:

- `semantic_payload_kb` / `payload_kb`,
- average delay and deadline violation,
- average energy,
- edge queue/inference/model-load delay,
- CPU/GPU share usage,
- service-level selection ratio.

UAV/UTM metrics:

- flight delay and propulsion energy,
- battery pressure,
- operational intent conflict count,
- UTM/risk violation rate,
- DSS/notification delay when enabled.

Baseline comparison metrics:

- payload reduction vs always-image,
- success gain vs always-cache and always-token,
- gap to oracle best feasible evidence,
- ablation drop for without LCB, without queue, without projection, without semantic tokens.

## Benchmark Artifacts

Commit when intentionally refreshed:

```text
docs/benchmark_protocol.md
docs/paper_algorithm_outline.md
docs/formal_problem_definition.md
docs/interfaces.md
docs/twc_algorithm_plan.md
outputs/env/formal_scenario_specs.md
outputs/env/benchmark_protocol_smoke.csv
```

Do not commit transient or heavy artifacts:

```text
outputs/env/scenario_smoke_*.csv
outputs/env/env_smoke_*/
outputs/rl/
outputs/hppo/
runs/
*.pt
*.pth
run.log
__pycache__/
```

For algorithm experiments, commit only intentionally curated CSV/Markdown summaries, not model checkpoints, rollout dumps, or logs.

## Smoke Command

Use a short smoke only to validate benchmark wiring:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m unittest discover -s tests
```

If a benchmark runner is used, write a fixed summary CSV such as:

```text
outputs/env/benchmark_protocol_smoke.csv
```

and avoid committing timestamped temporary smoke outputs.

