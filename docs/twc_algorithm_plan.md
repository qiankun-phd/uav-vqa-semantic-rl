# Paper-ready Semantic-Lyapunov Algorithm Plan

Last updated: 2026-06-22 Asia/Macau

Working title:

```text
Conservative VQA-grounded Semantic-Lyapunov Hybrid Control for UAV Semantic Communications
```

This document fixes the paper-facing algorithmic mainline for a journal-level wireless and edge intelligence study. The contribution should not be framed as generic PPO resource allocation. The core is a conservative semantic control architecture that uses measured VQA utility, virtual queues, and hybrid action projection to manage UAV semantic communication services.

## Main Idea

The system serves VQA tasks in a UAV-assisted semantic communication network. For each task, the controller decides:

- which semantic evidence level to route,
- which radio and computing resources to allocate,
- which UAV/waypoint action should support the sensing service,
- whether the conservative semantic QoS requirement can be satisfied.

The key quality metric is not rate or SINR alone. It is VQA answer reliability:

```text
accuracy_lcb >= epsilon_k
```

where `accuracy_lcb` is the lower confidence bound from the VQA-grounded semantic utility model.

## Layer 1: Conservative Semantic Utility Model

The semantic utility layer is fixed before online control:

```text
U_sem(l_k, s_k, snr_bin, view_quality_bin, freshness_bin, risk_k)
  -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

The controller should use:

- `accuracy_lcb` for conservative semantic QoS checks,
- `accuracy_mean` for expected utility reporting,
- `payload_kb` for communication load and transmission delay,
- `uncertainty` for risk-aware penalties or exploration,
- `sample_count` for confidence diagnostics.

This layer is calibrated from measured VQA correctness and includes Wilson confidence intervals, SNR monotonic sanity checks, cache SNR-invariant handling, and sparse-cell uncertainty.

## Layer 2: Semantic Virtual Queues

The online control problem is written with virtual queues that track long-term violations and service pressure. Recommended queues:

```text
Q_k^sem(t+1) = [Q_k^sem(t) + epsilon_k - accuracy_lcb,k(t)]_+
Q_k^ddl(t+1) = [Q_k^ddl(t) + T_k(t) - tau_k]_+
Q^eng_m(t+1) = [Q^eng_m(t) + E_m(t) - E_budget,m]_+
Q^utm(t+1)   = [Q^utm(t) + R_utm(t) - R_utm,max]_+
```

Where:

- `Q^sem` penalizes conservative semantic QoS shortfall.
- `Q^ddl` penalizes deadline violations.
- `Q^eng` penalizes UAV energy pressure.
- `Q^utm` penalizes airspace/UTM/risk pressure.

The Lyapunov drift-plus-penalty objective gives a principled control signal:

```text
minimize  Delta(Q(t)) + V * cost(t)
```

with cost terms for delay, energy, payload, semantic uncertainty, and UTM risk.

## Layer 3: Hybrid Control

The control action is hybrid:

```text
a_t = (s_k, bandwidth, power, cpu_share, gpu_share, uav_assignment, waypoint)
```

The action has three components:

1. Semantic evidence routing:
   - `s=0`: cache answer
   - `s=1`: semantic token / compact evidence
   - `s=2`: image evidence

2. Resource projection/AO:
   - project bandwidth/power/CPU/GPU shares into feasible bounds,
   - enforce service-dependent minimum resource floors,
   - optionally solve a small alternating optimization subproblem for continuous resources.

3. RL or learned residual policy:
   - learn task-dependent routing and resource residuals,
   - respect the projection layer,
   - optimize conservative semantic success rather than raw throughput.

## Control Objective

The objective is to maximize conservative VQA success while minimizing resource cost:

```text
maximize  sum_k rho_k * 1[accuracy_lcb,k >= epsilon_k, T_k <= tau_k]
          - lambda_T * delay
          - lambda_E * energy
          - lambda_D * payload_kb
          - lambda_U * uncertainty
          - lambda_R * UTM_risk
```

The controller should report all components separately. A single scalar reward is useful for training, but the paper tables must expose success, conservative accuracy, delay, energy, payload, uncertainty, and UTM/risk metrics.

## Required Ablations

Final paper experiments should include:

- without LCB: use `accuracy_mean` instead of `accuracy_lcb`,
- without queue: remove semantic/deadline/energy/UTM virtual queues,
- without projection: train raw continuous resource actions without feasibility projection,
- without semantic tokens: disable `s=1`,
- greedy baseline: minimum sufficient evidence using semantic utility,
- oracle baseline: best feasible evidence under known utility,
- fixed-service baselines: always cache, always semantic token, always image.

## Expected Thread Responsibilities

Algorithm thread:

- implement Lyapunov queue updates,
- implement hybrid routing/resource projection,
- tune RL only after queue/projection baselines are stable,
- write outputs under `outputs/rl`, `outputs/hppo`, or `runs`.

Environment thread:

- expose semantic utility fields in `info`,
- expose semantic/deadline/energy/UTM queue metrics,
- preserve action fields and dynamics compatibility,
- write environment-only outputs under `outputs/env`.

VQA/semantic utility thread:

- maintain `U_sem` artifacts,
- update calibration reports,
- avoid overwriting algorithm/environment outputs.
