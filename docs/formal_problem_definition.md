# Formal Problem Definition

Last updated: 2026-06-22 Asia/Macau

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

## Semantic Utility and QoS Model

VQA answer accuracy, payload, and confidence are provided by the calibrated semantic utility interface:

```text
U_sem(l_k, s_k, snr_bin, view_quality_bin, freshness_bin, risk_k)
  -> accuracy_mean,k, accuracy_lcb,k, payload_kb,k, uncertainty_k, sample_count_k
```

No online Qwen/VLM call is performed inside the simulator. The measured VQA utility is calibrated offline from answer correctness.

The semantic QoS constraint is conservative:

```text
accuracy_lcb,k(t) >= epsilon_k
```

This replaces traditional rate-only or SINR-only QoS. SINR/SNR still affects transmission rate and the selected SNR bin, but task success is defined by VQA answer reliability.

## Semantic Evidence Routing

The service decision `s_k(t)` selects what semantic evidence is transmitted or reused:

```text
s_k = 0: cache answer
s_k = 1: semantic token / compact evidence
s_k = 2: image evidence
```

Service level 3, ROI/crop evidence, remains reserved and disabled until the utility model includes calibrated ROI rows.

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
- tx delay depends on semantic utility payload and A2G SINR/rate
- queue and infer delay depend on edge CPU/GPU load and resource shares
- load delay depends on edge model-cache hit/miss

## Energy

The simulator decomposes energy:

```text
E_k = E_k^fly + E_k^hover/sense + E_k^tx + E_k^compute
```

## Constraints

Semantic quality:

```text
accuracy_lcb,k >= epsilon_k
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

The benchmark maximizes conservative VQA success while minimizing semantic-network cost:

```text
success_k = 1[accuracy_lcb,k >= epsilon_k
              and T_k <= tau_k
              and no hard resource/airspace violation]
```

The inspectable scalar reward is:

```text
r_k = rho_k accuracy_lcb,k success_k
      - lambda_T T_k
      - lambda_E E_k
      - lambda_D payload_kb,k
      - lambda_U uncertainty_k
      - lambda_Q 1[accuracy_lcb,k < epsilon_k]
      - lambda_deadline 1[T_k > tau_k]
      - lambda_conflict 1[airspace conflict]
```

The paper-level objective can be stated as:

```text
maximize  sum_t sum_k rho_k success_k(t)
          - delay cost
          - energy cost
          - payload cost
          - semantic uncertainty cost
          - UTM/risk cost
```

This makes the communication objective semantic: the controller is rewarded for correct, timely, low-payload VQA service rather than for maximizing raw rate.

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
