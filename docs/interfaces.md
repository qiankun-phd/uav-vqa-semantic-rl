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
| candidate_path_metrics | dict | Per-path feasibility and cost estimates for `cache`, `token`, `image`, `defer`, and `cache_update`. |

Recommended additional fields:

- deadline_s
- task_status
- remaining_deadline_s
- defer_count
- expired
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
| semantic_path | str | Preferred V2 semantic service route: `cache`, `token`, `image`, `defer`, or `cache_update`. Backward compatible with `service_level`: cache=0, token=1, image=2. |
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
    'semantic_path': 'cache',
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

Semantic path semantics:

| semantic_path | compatible service_level | behavior |
|---|---:|---|
| cache | 0 | Reuse an eligible semantic cache entry. It is SNR-invariant and only succeeds when exact/nearby cache quality LCB clears `epsilon_k` and freshness is not expired. |
| token | 1 | Use semantic token / detector compact evidence for the current task. |
| image | 2 | Use image evidence for the current task. |
| defer | none | Keep the task in the queue, increment `defer_count`, age cache state, and consume deadline; it must not mark the task completed. |
| cache_update | 1 | V1 maps to token evidence and, when successful, writes or refreshes the semantic cache entry. |

Task queue status:

```text
task_status in {pending, served, deferred, expired}
remaining_deadline_s = max(0, deadline_s - elapsed_slots * slot_duration_s)
expired = remaining_deadline_s <= 0 before service completion
```

Cache eligibility fields exposed in observations and step info:

```text
cache_exact_match
cache_nearby_match
cache_eligible
cache_quality_lcb
cache_age
cache_freshness_bin
cache_hit_probability
```

The environment defines `cache_eligible=True` only when there is an exact or nearby same-type semantic cache entry, the cache freshness is not expired, and `cache_quality_lcb >= epsilon_k`.

Candidate path metrics:

```text
candidate_path_metrics[path] = {
  feasible,
  accuracy_lcb,
  accuracy_mean,
  quality_gap,
  payload_kb,
  delay_s,
  energy_j,
  deadline_slack_s,
  cache_eligible,
  utm_constraint_violation,
}
```

These metrics are diagnostic estimates for action selection and projection. They should not mutate the task queue or cache store.

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

Candidate-service usage for mobility-aware RL/env:

```python
candidates = utility.get_service_candidates(obs)
```

Each candidate keeps the same semantic utility LUT key and returns `semantic_path`, `accuracy_mean`, `accuracy_lcb`, `uncertainty`, `payload_kb`, `semantic_quality_gap`, `semantic_efficiency`, `estimated_delay_s`, `estimated_delay_feasible`, `semantic_feasible`, `deadline_feasible`, `joint_feasible`, `cache_accuracy_mean`, `cache_accuracy_lcb`, `cache_uncertainty`, `cache_quality_gap`, `cache_recommended`, `cache_eligible`, `candidate_path_metrics`, `is_snr_sensitive`, `recommended_for_low_snr`, and `recommended_for_critical`.

The helper does not add a LUT dimension. It evaluates the same task condition across service levels and computes:

```text
semantic_quality_gap = max(0, epsilon_k - accuracy_lcb)
semantic_feasible = accuracy_lcb >= epsilon_k
deadline_feasible = estimated_delay_s <= deadline_s
joint_feasible = semantic_feasible and deadline_feasible
semantic_efficiency = quality-adjusted conservative utility per payload unit
```

The candidate helper reads deadline from `deadline_s`, `tau_k`, or `deadline`. It reads per-service delay estimates from `estimated_delay_by_service`, `delay_by_service`, `service_delay_s`, `service_delay_by_level`, or `estimated_delay_s_by_service`. If no delay estimate is provided, it uses a small conservative fallback based on service level and payload.

Cache/path helper usage for semantic path control:

```python
cache_lcb = utility.cache_quality_lcb(task_type, snr_bin, view_quality_bin, freshness_bin, risk_level)
cache = utility.cache_quality_metrics(task_type, snr_bin, view_quality_bin, freshness_bin, risk_level, epsilon_k)
path = utility.path_utility("token", task_type, snr_bin, view_quality_bin, freshness_bin, risk_level, epsilon_k)
```

Path names are paper-facing control choices:

| semantic_path | service level | meaning |
|---|---:|---|
| `cache` | 0 | Reuse cached answer or cached semantic result. Cache is SNR-invariant; quality depends on freshness and cached answer reliability. |
| `token` | 1 | Send compact semantic tokens such as detector tags, boxes, counts, and confidence summaries. |
| `image` | 2 | Send image evidence for visual reasoning. Payload and deadline pressure are high under weak links. |
| `cache_update` | default 1 | Serve with fresh evidence and refresh the cache for future requests. Defaults to token update, but callers may request image update if needed. |

Cache semantics:

```text
cache_quality_gap = max(0, epsilon_k - cache_accuracy_lcb)
cache_eligible = cache_recommended
```

- `cache_accuracy_mean`, `cache_accuracy_lcb`, and `cache_uncertainty` are always derived from service level 0.
- Cache quality is not driven by current SNR; `freshness_bin` is the main cache-state dimension.
- `expired` cache is not recommended for critical tasks.
- Critical tasks can use cache only when `freshness_bin == fresh`, `cache_accuracy_lcb >= epsilon_k`, sample support exists, and uncertainty is acceptable.
- `candidate_path_metrics` is a compact per-candidate dictionary for environment logging and policy debugging. It includes at least `semantic_path`, candidate `accuracy_lcb`, `semantic_quality_gap`, `payload_kb`, `cache_accuracy_lcb`, `cache_quality_gap`, and `cache_recommended`.
- The semantic utility layer does not estimate future cache reuse value. Therefore `cache_update` must not be treated as actively recommended solely because the cache is missing or stale. Environment/Algorithm should provide a separate `future_reuse_value`, `cache_update_value`, or `expected_future_cache_hits` signal before preferring cache update over token/image service.

Paper-facing service semantics:

| service | semantic name | communication meaning | expected operating point |
|---:|---|---|---|
| 0 | cache answer | Reuse cached answer or cached semantic result. Payload is near zero, but quality depends on freshness/cache hit and is not SNR-sensitive. | Low-risk, fresh-cache tasks; avoid for stale critical tasks unless LCB clears epsilon. |
| 1 | semantic token / compact evidence | Transmit detector tags, boxes, counts, and compact evidence. This is the main lightweight semantic communication mode. | Low-SNR, edge-overload, and payload-sensitive settings where token LCB satisfies semantic QoS. |
| 2 | image evidence | Transmit raw/full visual evidence for VQA/VLM reasoning. Payload and edge workload are high, and delay is more sensitive to poor links. | Use when compact evidence is insufficient and the task can tolerate higher delay/energy. |

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
    'semantic_path': str,
    'task_status': str,
    'remaining_deadline_s': float,
    'defer_count': int,
    'expired': bool,
    'cache_exact_match': bool,
    'cache_nearby_match': bool,
    'cache_eligible': bool,
    'cache_quality_lcb': float,
    'cache_age': float,
    'cache_freshness_bin': str,
    'cache_hit_probability': float,
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

## Two-timescale RL Action Contract

The mobility-aware RL controller uses the same environment action schema, but factorizes the policy into slow mobility control and fast semantic-resource control:

```text
a_t = (a_mobility, a_resource)

a_mobility = {
  uav_assignment,
  mobility_mode,
  waypoint_delta,
  altitude_delta,
}

a_resource = {
  service_level,
  bandwidth,
  power,
  cpu_share,
  gpu_share,
}
```

Default timescale:

```text
mobility actor update interval K = 3 slots
semantic-resource actor update interval = every task/slot
critic = centralized V(s, a_mobility, a_resource)
```

The observation vector must include semantic-LCB utility fields, resource/network state, mobility/UTM diagnostics, and Lyapunov queues. The current queue fields are:

```text
Q_quality, Q_deadline, Q_energy, Q_utm
```

Reward components used by the two-timescale controller:

```text
semantic_accuracy_lcb / semantic_success
semantic_quality_gap
deadline_violation
energy_j
payload_kb
edge_queue delay or edge overload proxy
utm_conflict_risk / utm_conflict_violation
mobility_energy_j
arrival_delay_s
coverage_gain
```

Projection requirements:

- cache service (`service_level=0`) should not consume radio/edge resources unless explicit mobility is selected.
- semantic-token service (`service_level=1`) receives minimum bandwidth/power/cpu/gpu floors for compact evidence transmission and inference.
- image service (`service_level=2`) receives higher resource floors but remains bounded by edge/load/deadline pressure.
- mobility deltas are clipped to configured waypoint and altitude ranges.
- UAV assignment and mobility mode should respect battery and UTM masks when those masks are exposed by the environment.

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
