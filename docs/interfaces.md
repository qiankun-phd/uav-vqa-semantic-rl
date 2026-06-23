# Interfaces

Last updated: 2026-06-23 Asia/Shanghai
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
| snr_bin | str/float/int | Nearest semantic-utility SNR bin label/value. Current V1.9 utility model uses -5dB, 0dB, 5dB, 10dB, 15dB, 20dB. |
| uav_state | dict/vector | UAV position, velocity/heading, battery, remaining compute or flight budget. |
| edge_load | float/vector | Edge/server compute congestion or queue load. |
| cache_state | dict/vector | Whether cache answer exists, age/freshness, cache confidence if available. |

Recommended additional fields:

- deadline_s
- payload_kb_estimates_by_service
- task_id
- episode_step
- feasible_uavs
- uav_task_distances_m
- uav_battery_ratio
- predicted_fly_delay_s
- predicted_fly_energy_j
- task_area4d
- utm_conflict_risk
- future_task_proximity
- coverage_score_by_uav
- feasible_mobility_mask

## Action Contract

Algorithm thread outputs an action with these fields:

| field | type | notes |
|---|---|---|
| service_level | int | 0 cache answer, 1 semantic token / compact evidence, 2 image evidence. Backward-compatible aliases: 1 detector semantic tokens, 2 raw image evidence. Service 3 ROI is reserved and currently not active in V1.9. |
| bandwidth | float | Hz or normalized share; unit must be written in info. |
| power | float | Watts or normalized share; unit must be written in info. |
| cpu_share | float | 0..1 normalized edge/local CPU share. |
| gpu_share | float | 0..1 normalized GPU share, if used. |
| uav_assignment | int/list | UAV id or task-to-UAV mapping. |
| mobility_mode | str | `stay`, `serve_task`, `reposition`, `avoid_conflict`, or `return_base`. Old actions without this field default to `stay` for cache and `serve_task` for token/image service. |
| waypoint_delta | list/array/dict | Optional relative `dx, dy` movement command for `reposition`. |
| altitude_delta | float | Optional relative altitude command for `reposition`. |
| waypoint | list/array | Backward-compatible absolute waypoint. If present without `mobility_mode`, it is treated as `reposition`. |

Minimal V1.9-compatible action:

```python
action = {
    'service_level': 0,
    'bandwidth': 1_000_000.0,
    'power': 0.1,
    'cpu_share': 0.0,
    'gpu_share': 0.0,
    'uav_assignment': 0,
    'mobility_mode': 'stay',
    'waypoint_delta': [0.0, 0.0],
    'altitude_delta': 0.0,
    'waypoint': None,
}
```

## Semantic Utility Interface

The VQA/SNR thread now provides a VQA-grounded, task-conditioned semantic utility model, not only a raw lookup table:

```text
U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
  -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

Primary calibrated utility path:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_semantic_utility_with_ci.csv
```

Primary calibration report:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/reports/semantic_utility_calibration.md
```

API implementation:

```text
/home/qiankun/phd_research/vqa_semcom/src/vqa_semcom/semantic/utility.py
```

The raw measured V1.9 quality table remains available for traceability:

```text
/home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_snr_semantic_quality_lut.csv
```

The new utility CSV extends raw answer accuracy with:

```text
sample_count
accuracy_mean
accuracy_ci_low
accuracy_ci_high
accuracy_lcb
payload_kb
uncertainty
```

Semantics:

- `accuracy_mean`: calibrated expected answer correctness for the task/service/SNR/view/freshness/risk condition.
- `accuracy_lcb`: conservative lower confidence bound for QoS decisions.
- `payload_kb`: measured communication load for the selected visual/semantic evidence.
- `uncertainty`: finite-sample/statistical uncertainty; sparse cells should be treated cautiously by RL.
- `sample_count`: number of measured VQA prediction samples behind the cell.

Nearest-bin SNR helpers still live in:

```text
/home/qiankun/phd_research/vqa_semcom/src/vqa_semcom/snr.py
```

Relevant helper functions:

```python
nearest_snr_bin(sensed_snr_db, snr_bins_db)
snr_db_to_bin_label(sensed_snr_db, snr_bins_db)
channel_bin_from_snr(snr_db)
```

Recommended control usage:

```python
from vqa_semcom.semantic.utility import SemanticUtilityModel

utility = SemanticUtilityModel.from_csv(Path("outputs/lut/v1_9_semantic_utility_with_ci.csv"))
u = utility.U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)

A_k = u.accuracy_lcb  # conservative control-facing quality estimate
payload_kb = u.payload_kb
semantic_uncertainty = u.uncertainty
```

For paper writing, describe this as a task-conditioned semantic utility model calibrated from measured VQA correctness under sensed SNR, view quality, freshness, and service-level evidence. Do not describe it as an image-quality score.

## Reward Contract

Algorithm and environment threads should keep reward components inspectable through info:

```python
info = {
    'answer_accuracy_est': A_k,
    'semantic_accuracy_mean': float,
    'semantic_accuracy_lcb': float,
    'semantic_uncertainty': float,
    'semantic_sample_count': int,
    'semantic_payload_kb': float,
    'semantic_quality_gap': float,  # epsilon_k - semantic_accuracy_lcb
    'semantic_success': bool,       # semantic_accuracy_lcb >= epsilon_k
    'success': bool,
    'deadline_s': float,
    'epsilon_k': float,
    'risk_level': str,
    'view_quality_bin': str,
    'freshness_bin': str,
    'delay_s': float,
    'energy_j': float,
    'payload_kb': float,
    'quality_violation': bool,
    'deadline_violation': bool,
    'risk_violation': bool,
    'utm_delay_s': float,
    'utm_conflict_violation': bool,
    'utm_constraint_violation': bool,
    'airspace_state': str,
    'dss_delay_s': float,
    'snr_bin': str,
    'service_level': int,
    'mobility_mode': str,
    'waypoint_x': float,
    'waypoint_y': float,
    'altitude_m': float,
    'fly_distance_m': float,
    'coverage_gain': float,
    'mobility_energy_j': float,
    'arrival_delay_s': float,
    'utm_conflict_risk': float,
}
```

QoS semantics:

- `semantic_accuracy_mean` is the calibrated expected correctness.
- `semantic_accuracy_lcb` is the conservative control-facing QoS estimate and should be used for hard semantic QoS checks.
- `semantic_quality_gap > 0` means the task is below its required semantic threshold.
- `semantic_success` is the semantic-only success flag before deadline/resource/UTM constraints.
- `payload_kb` and `semantic_payload_kb` should match for the selected service; both are exposed for backward compatibility.

UTM/risk cost semantics for journal-level constrained semantic control:

- `utm_delay_s = utm_dss_delay_s + utm_notification_delay_s`.
- `utm_conflict_violation` is true when strategic airspace conflict or UTM coordination violation is active.
- `risk_violation` is true when a critical/high-risk task violates semantic QoS, deadline, or UTM coordination.
- `airspace_state` mirrors the operational intent state: accepted, activated, nonconforming, or contingent.

Mobility-control semantics:

- `mobility_mode=stay` keeps the UAV at its current position and altitude and accounts for hover energy.
- `mobility_mode=serve_task` moves toward the selected task's 4D area.
- `mobility_mode=reposition` follows `waypoint_delta` or the backward-compatible absolute `waypoint`.
- `mobility_mode=avoid_conflict` moves away from the local UTM conflict centroid and reports reduced predicted conflict risk.
- `mobility_mode=return_base` moves toward the UAV's initial base/safe point.
- `service_level=0` cache answer reuse does not by itself create an operational intent or airspace conflict; explicit mobility can still consume flight/hover energy.

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
