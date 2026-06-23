# Paper Algorithm Outline

Last updated: 2026-06-23 Asia/Shanghai

Working paper narrative:

```text
Conservative VQA-grounded Semantic-Lyapunov Hybrid Control for UAV Semantic Communications
```

This outline fixes the paper-ready story for the UAV semantic communication benchmark and algorithm. The contribution should be positioned as a high-quality wireless and edge intelligence study, not as a generic PPO resource allocation exercise.

## 1. UAV-assisted Semantic VQA Emergency Network

The target system is a multi-UAV emergency inspection network. Rescue centers and users issue visual questions about blocked roads, damaged areas, trapped people, vehicle counts, and other operational states. UAVs support the service by collecting or reusing visual semantic evidence and delivering it to an edge VQA service.

The network couples:

- UAV mobility and sensing,
- air-to-ground semantic evidence transmission,
- edge computing and cache state,
- VQA-grounded answer quality,
- UTM/airspace risk.

The key difference from conventional communication is that the system does not optimize raw rate alone. It optimizes whether the received evidence is sufficient to answer the task.

## 2. Conservative Semantic Utility Model

The semantic utility model is calibrated from measured VQA correctness:

```text
U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
  -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

The conservative semantic QoS condition is:

```text
accuracy_lcb >= epsilon_k
```

The lower confidence bound protects the controller from over-trusting sparse or noisy VQA measurements. `accuracy_mean` is useful for expected performance reporting, while `accuracy_lcb` is the online control quantity.

The utility model also exposes:

- `payload_kb`: communication cost of evidence transmission,
- `uncertainty`: finite-sample and confidence penalty,
- `sample_count`: how much measured evidence supports the cell.

## 3. Semantic Evidence Routing

For each VQA task, the controller chooses a semantic evidence service:

| service level | name | meaning |
|---:|---|---|
| 0 | cache answer | reuse cached answer or semantic result |
| 1 | semantic token / compact evidence | transmit detector tags, boxes, counts, or compact semantic evidence |
| 2 | image evidence | transmit image evidence for VQA inference |

This routing decision is the communication-load reduction point. Instead of always uploading images, the controller selects the smallest evidence type whose conservative semantic utility can satisfy the task.

Example:

```text
presence question + fresh cache -> s=0
counting question + good SNR/view -> s=1
critical or uncertain task -> s=2
```

## 4. Semantic-LCB Lyapunov Hybrid Control

The algorithmic mainline combines conservative utility, virtual queues, and hybrid action control.

Semantic virtual queues track long-term violations:

```text
Q_sem(t+1) = [Q_sem(t) + epsilon_k - accuracy_lcb,k(t)]_+
Q_ddl(t+1) = [Q_ddl(t) + T_k(t) - tau_k]_+
Q_eng(t+1) = [Q_eng(t) + E_k(t) - E_budget]_+
Q_risk(t+1) = [Q_risk(t) + R_utm(t) - R_max]_+
```

The controller action remains:

```text
a_t = (service_level, bandwidth, power, cpu_share, gpu_share, uav_assignment, waypoint)
```

The hybrid control structure has three parts:

1. semantic evidence routing over `s=0/1/2`,
2. projection or alternating optimization for feasible radio/compute resources,
3. learned or heuristic residual control for scenario-specific decisions.

The objective is:

```text
maximize conservative VQA success
minimize delay + energy + payload + uncertainty + UTM risk
```

This makes the method interpretable: queue states explain why the controller escalates evidence, increases resources, or changes UAV assignments.

## 4.1 Two-timescale Mobility-aware Semantic Resource PPO

The current paper method is upgraded from a monolithic service/resource PPO into a two-timescale cognitive controller:

```text
state
  -> shared encoder
  -> slow mobility actor
  -> fast semantic-resource actor
  -> centralized critic V(s, a_mobility, a_resource)
```

The slow actor updates every `K=3` slots by default and controls UAV/network geometry:

```text
a_mobility = (
  uav_assignment,
  mobility_mode,
  waypoint_delta,
  altitude_delta
)
```

where `mobility_mode` is one of `stay`, `serve_task`, `reposition`, `avoid_conflict`, or `return_base`. The fast actor updates every task/slot and controls semantic evidence and radio/edge resources:

```text
a_resource = (
  service_level,
  bandwidth,
  power,
  cpu_share,
  gpu_share
)
```

The critic evaluates the joint action so the policy can trade mobility choices against semantic utility and resource feasibility. Lyapunov queues remain part of the observation and reward shaping:

```text
Q_sem, Q_ddl, Q_eng, Q_risk/utm
```

The drift-plus-penalty reward now includes mobility terms:

```text
semantic LCB / success
- semantic quality gap
- deadline, energy, payload, edge queue, UTM conflict costs
- mobility energy
- arrival delay
+ coverage gain
```

Continuous resource decisions are projected after actor sampling. The projection enforces service-dependent resource floors, prevents cache actions from wasting bandwidth/compute, gives semantic-token actions enough communication/inference resources, and keeps image evidence bounded under edge overload. Mobility actions are similarly clipped to waypoint and altitude envelopes and masked by battery/UTM feasibility where the environment exposes the required state.

Paper-facing ablations:

- proposed two-timescale PPO,
- monolithic PPO,
- no mobility actor,
- no Lyapunov queues,
- no projection,
- semantic greedy,
- always cache / token / image.

This framing makes the controller a UAV semantic communication network controller rather than a generic PPO baseline: the slow actor reshapes sensing/coverage/UTM geometry while the fast actor chooses the cheapest evidence and resource allocation that satisfies conservative semantic QoS.

## 5. Scenario-aware Evaluation

The benchmark uses five named scenarios:

```text
nominal_patrol
disaster_hotspot
low_snr_blockage
edge_overload
utm_conflict
```

Each scenario stresses a different part of the semantic communication network:

- nominal patrol tests stable payload-saving behavior,
- disaster hotspot tests bursty and critical VQA demand,
- low-SNR blockage tests channel-aware semantic reliability,
- edge overload tests compute-aware evidence selection,
- UTM conflict tests airspace/risk-aware UAV semantic service.

The paper should report:

- conservative success rate,
- `semantic_accuracy_lcb` and `semantic_quality_gap`,
- payload, delay, and energy,
- uncertainty and sparse-cell sensitivity,
- service-level selection ratio,
- UTM/risk violation metrics when applicable.

Required ablations:

- without LCB,
- without queue,
- without projection,
- without semantic tokens,
- greedy and oracle semantic-utility baselines,
- fixed-service baselines.

The expected claim is that conservative semantic utility plus Lyapunov-style hybrid control can reduce communication load while maintaining reliable VQA task success under UAV mobility, edge load, and UTM constraints.
