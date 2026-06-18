# Formal Problem Definition

Last updated: 2026-06-19 Asia/Macau

## System

The system is a UAV-assisted goal-oriented VQA semantic communication network.

- UAV set: `M = {1, ..., m}`
- VQA task set at slot `t`: `K(t)`
- edge/BS set: `B = {1, ..., b}`
- semantic service set: `S = {0,1,2}` in V1.9

`s=3` ROI/crop evidence is reserved and disabled until the VQA thread provides ROI/crop utility/LUT rows.

## Task Model

Each task `k` has:

```text
T_k = (area4d_k, question_k, l_k, tau_k, epsilon_k, rho_k, risk_k, t_gen,k)
```

where:

- `area4d_k` is the operational area and time window
- `l_k` is the question type
- `tau_k` is the deadline
- `epsilon_k` is the required expected answer accuracy
- `rho_k` is task priority
- `risk_k` is normal or critical

## Semantic Quality Model

VQA answer accuracy and payload are provided by the empirical V1.9 LUT:

```text
A_k = LUT[l_k, s_k, snr_bin, view_quality_bin, freshness_bin, risk_k]
D_k = LUT_payload[l_k, s_k, snr_bin, view_quality_bin, freshness_bin, risk_k]
```

No online Qwen/VLM call is performed inside the simulator.

## Decision Variables

At each slot:

- `x_mk(t)`: UAV-task assignment
- `s_k(t)`: semantic service level
- `b_k(t)`: bandwidth allocation
- `p_k(t)`: transmit power
- `f_k^cpu(t)`: CPU share
- `f_k^gpu(t)`: GPU share
- `w_m(t)`: optional waypoint/revisit target
- `c_k(t)`: cache/revisit/observe decision

## Delay

The simulator decomposes total delay:

```text
T_k = T_k^fly + T_k^sense + T_k^tx + T_k^queue + T_k^infer + T_k^load
```

where:

- fly delay depends on UAV position, target area, and speed
- sense delay depends on semantic service level
- tx delay depends on LUT payload and A2G SINR/rate
- queue and infer delay depend on edge CPU/GPU load and resource shares
- load delay depends on edge model-cache hit/miss

## Energy

The simulator decomposes energy:

```text
E_k = E_k^fly + E_k^hover/sense + E_k^tx + E_k^compute
```

## Constraints

Quality:

```text
A_k >= epsilon_k
```

Deadline:

```text
T_k <= tau_k
```

Resource:

```text
sum b_k <= B_total
0 <= f_k^cpu, f_k^gpu <= 1
GPU memory used <= GPU memory capacity
UAV energy >= reserve
```

Airspace:

```text
Area4D_i overlaps Area4D_j => conflict penalty
```

Only service levels requiring an operational intent, such as observe/revisit, generate airspace conflict checks. Cache-only reuse does not.

## Objective

The benchmark minimizes semantic-network cost while maximizing task-oriented success:

```text
success_k = 1[A_k >= epsilon_k and T_k <= tau_k and no hard resource/airspace violation]
```

The inspectable scalar reward is:

```text
r_k = rho_k A_k success_k
      - lambda_T T_k
      - lambda_E E_k
      - lambda_D D_k
      - lambda_Q 1[A_k < epsilon_k]
      - lambda_deadline 1[T_k > tau_k]
      - lambda_conflict 1[airspace conflict]
```

The simulator also exposes a semantic utility API:

```text
semantic_utility = semantic_gain * deadline_factor
                   - resource_cost
                   - quality_shortfall
                   + success_bonus
```

## Formal Scenarios

Train:

- `train_nominal`
- `train_mixed_random`

Test:

- `test_conflict_heavy`
- `test_interference_heavy`
- `test_cache_heavy`
- `test_mobility_stress`
- `test_unseen_mixed`

## Scalability Configs

UAV count:

- `M2`, `M4`, `M6`, `M8`

Task arrival:

- `low`, `medium`, `high`

Edge load:

- `light`, `medium`, `heavy`

The generated benchmark spec is written to:

```text
outputs/env/formal_scenario_specs.md
```
