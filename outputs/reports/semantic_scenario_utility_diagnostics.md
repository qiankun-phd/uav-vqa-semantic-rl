# Semantic Scenario Utility Diagnostics

Generated from existing V1.9 semantic utility outputs and Algorithm v2 scenario benchmark rollouts. No Qwen/VLM predictions were regenerated and no RL algorithm code was modified.

## Inputs

- Utility model: `outputs/lut/v1_9_semantic_utility_with_ci.csv`
- Benchmark summary: `outputs/rl/semantic_scenario_benchmark_v2/scenario_comparison_summary.csv`
- Benchmark rollouts: `outputs/rl/semantic_scenario_benchmark_v2/*/seed*/{baselines,proposed_ppo,...}/v1_9_resource_alloc_rollout.csv`
- Diagnostic CSV: `outputs/reports/semantic_scenario_utility_diagnostics.csv`

## Important Caveat

The rollout CSV exposes `semantic_quality_gap = max(0, epsilon_k - accuracy_lcb)` but does not persist `epsilon_k` directly. For failed rows, `epsilon_k` is exactly reconstructable as `accuracy_lcb + semantic_quality_gap`. For successful rows, this report uses the scenario risk-level floor from the environment preset as a conservative diagnostic estimate. Future benchmark traces should persist `epsilon_k` explicitly.

## Global Utility Snapshot

| service | utility cells | mean accuracy | mean LCB | mean uncertainty | mean sample count | mean payload KB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 216 | 0.525 | 0.461 | 0.268 | 247.8 | 0.000 |
| 1 | 216 | 0.720 | 0.550 | 0.319 | 247.8 | 0.920 |
| 2 | 216 | 0.598 | 0.486 | 0.305 | 247.8 | 90.198 |

Interpretation: semantic tokens (`s=1`) are the strongest utility class on average and are roughly two orders of magnitude lighter than image evidence (`s=2`). Cache (`s=0`) is payload-free but has a low conservative LCB, so it should be risk-gated rather than used as a default escape route.

## Scenario-Level Findings

| scenario | proposed semantic success | proposed LCB | proposed gap | deadline vio | cache/token/image | key diagnosis | recommendation |
|---|---:|---:|---:|---:|---|---|---|
| nominal_patrol | 0.303 | 0.611 | 0.186 | 0.330 | 0.367/0.451/0.182 | PPO improves over fixed baselines but still overuses cache and pays quality gaps. | Reduce cache attraction via LCB margin and uncertainty-aware reward. |
| disaster_hotspot | 0.238 | 0.585 | 0.267 | 0.289 | 0.185/0.799/0.017 | High critical mix and epsilon floor 0.84 make token LCB insufficient for many critical rows. | Use layered epsilon by task type/risk; require image/ROI for critical counting only. |
| low_snr_blockage | 0.785 | 0.746 | 0.108 | 0.699 | 0.276/0.697/0.027 | Semantic greedy is much stronger than PPO, indicating routing/exploration rather than utility model failure. | Increase semantic-token prior and distill greedy/oracle routing under low SNR. |
| edge_overload | 0.000 | 0.691 | 0.121 | 0.531 | 0.309/0.691/0.000 | LCB is close to normal/critical floors but average gap remains 0.121 and deadline violations are 0.531. | Keep epsilon; add compute-aware routing/projection and avoid high cache ratio for critical rows. |
| utm_conflict | 0.000 | 0.581 | 0.239 | 0.177 | 0.420/0.580/0.000 | LCB is below critical floor; cache ratio 0.420 leaves many rows below epsilon before UTM/delay effects. | Use risk-aware cache penalty and UTM-aware token/image escalation. |

## nominal_patrol

medium/good SNR, medium/good view, moderate edge load; sanity baseline.

### Proposed PPO Task/Condition Distribution

- `question_type`: counting: 864 (48.0%); presence: 936 (52.0%)
- `risk_level`: critical: 1418 (78.8%); normal: 382 (21.2%)
- `snr_bin`: -5dB: 661 (36.7%); 20dB: 1139 (63.3%)
- `view_quality_bin`: good: 215 (11.9%); medium: 1585 (88.1%)
- `freshness_bin`: expired: 1206 (67.0%); fresh: 292 (16.2%); stale: 302 (16.8%)
- `service_level`: 0: 661 (36.7%); 1: 811 (45.1%); 2: 328 (18.2%)
- failed-row exact epsilon range: 0.760 to 0.820, mean 0.800, n=1254

### Selected-Service Utility Breakdown

| service | selected | acc_mean | acc_lcb | uncertainty | samples | payload KB | quality gap | semantic success | deadline vio | final success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.367 | 0.609 | 0.514 | 0.333 | 22.7 | 0.000 | 0.289 | 0.003 | 0.000 | 0.003 |
| 1 | 0.451 | 0.843 | 0.611 | 0.404 | 16.1 | 1.154 | 0.178 | 0.266 | 0.392 | 0.144 |
| 2 | 0.182 | 0.974 | 0.805 | 0.306 | 34.3 | 184.935 | 0.000 | 1.000 | 0.841 | 0.159 |

### Policy Contrast

| policy | semantic success | task success | LCB | mean accuracy | uncertainty | quality gap | deadline vio | payload KB | cache/token/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| always_cache | 0.000 | 0.000 | 0.340 | 0.433 | 0.374 | 0.450 | 0.000 | 0.000 | 1.000/0.000/0.000 |
| always_semantic_token | 0.106 | 0.061 | 0.495 | 0.732 | 0.415 | 0.304 | 0.358 | 1.142 | 0.000/1.000/0.000 |
| always_image | 0.111 | 0.013 | 0.350 | 0.453 | 0.391 | 0.454 | 0.897 | 198.091 | 0.000/0.000/1.000 |
| semantic_greedy | 0.087 | 0.036 | 0.338 | 0.437 | 0.388 | 0.467 | 0.866 | 180.500 | 0.002/0.079/0.919 |
| proposed_ppo | 0.303 | 0.095 | 0.611 | 0.781 | 0.360 | 0.186 | 0.330 | 34.219 | 0.367/0.451/0.182 |
| ppo_without_lcb | 0.303 | 0.095 | 0.612 | 0.783 | 0.360 | 0.185 | 0.335 | 34.052 | 0.363/0.456/0.181 |
| ppo_without_queues | 0.272 | 0.098 | 0.630 | 0.798 | 0.357 | 0.170 | 0.353 | 14.800 | 0.329/0.579/0.092 |
| ppo_without_projection | 0.107 | 0.033 | 0.464 | 0.628 | 0.385 | 0.332 | 0.411 | 16.035 | 0.454/0.463/0.083 |

## disaster_hotspot

burst traffic, high critical ratio, tighter deadlines, stricter epsilon floor.

### Proposed PPO Task/Condition Distribution

- `question_type`: counting: 775 (51.7%); presence: 725 (48.3%)
- `risk_level`: critical: 1500 (100.0%)
- `snr_bin`: -5dB: 277 (18.5%); 20dB: 1223 (81.5%)
- `view_quality_bin`: good: 479 (31.9%); medium: 664 (44.3%); poor: 357 (23.8%)
- `freshness_bin`: expired: 1275 (85.0%); fresh: 24 (1.6%); stale: 201 (13.4%)
- `service_level`: 0: 277 (18.5%); 1: 1198 (79.9%); 2: 25 (1.7%)
- failed-row exact epsilon range: 0.840 to 0.840, mean 0.840, n=1143

### Selected-Service Utility Breakdown

| service | selected | acc_mean | acc_lcb | uncertainty | samples | payload KB | quality gap | semantic success | deadline vio | final success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.185 | 0.678 | 0.567 | 0.433 | 140.4 | 0.000 | 0.275 | 0.307 | 0.000 | 0.307 |
| 1 | 0.799 | 0.850 | 0.582 | 0.479 | 65.6 | 1.169 | 0.271 | 0.206 | 0.341 | 0.082 |
| 2 | 0.017 | 0.975 | 0.949 | 0.079 | 281.0 | 183.878 | 0.000 | 1.000 | 1.000 | 0.000 |

### Policy Contrast

| policy | semantic success | task success | LCB | mean accuracy | uncertainty | quality gap | deadline vio | payload KB | cache/token/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| always_cache | 0.015 | 0.015 | 0.265 | 0.342 | 0.364 | 0.576 | 0.000 | 0.000 | 1.000/0.000/0.000 |
| always_semantic_token | 0.105 | 0.039 | 0.471 | 0.691 | 0.404 | 0.376 | 0.298 | 1.170 | 0.000/1.000/0.000 |
| always_image | 0.220 | 0.000 | 0.469 | 0.575 | 0.339 | 0.395 | 1.000 | 188.950 | 0.000/0.000/1.000 |
| semantic_greedy | 0.107 | 0.041 | 0.429 | 0.577 | 0.454 | 0.416 | 0.959 | 169.315 | 0.017/0.090/0.893 |
| proposed_ppo | 0.238 | 0.122 | 0.585 | 0.820 | 0.464 | 0.267 | 0.289 | 3.998 | 0.185/0.799/0.017 |
| ppo_without_lcb | 0.226 | 0.126 | 0.596 | 0.816 | 0.449 | 0.255 | 0.273 | 0.865 | 0.260/0.740/0.000 |
| ppo_without_queues | 0.235 | 0.124 | 0.585 | 0.819 | 0.464 | 0.267 | 0.284 | 2.894 | 0.192/0.797/0.011 |
| ppo_without_projection | 0.119 | 0.009 | 0.441 | 0.582 | 0.365 | 0.406 | 0.490 | 0.573 | 0.510/0.490/0.000 |

## low_snr_blockage

weak A2G links, low bandwidth, image payload delay pressure; cache/tokens can dominate.

### Proposed PPO Task/Condition Distribution

- `question_type`: counting: 116 (6.4%); presence: 1684 (93.6%)
- `risk_level`: critical: 1710 (95.0%); normal: 90 (5.0%)
- `snr_bin`: -5dB: 1493 (82.9%); 0dB: 108 (6.0%); 10dB: 48 (2.7%); 15dB: 36 (2.0%); 20dB: 50 (2.8%); 5dB: 65 (3.6%)
- `view_quality_bin`: good: 27 (1.5%); medium: 63 (3.5%); poor: 1710 (95.0%)
- `freshness_bin`: expired: 1491 (82.8%); stale: 309 (17.2%)
- `service_level`: 0: 497 (27.6%); 1: 1255 (69.7%); 2: 48 (2.7%)
- failed-row exact epsilon range: 0.650 to 0.820, mean 0.808, n=387

### Selected-Service Utility Breakdown

| service | selected | acc_mean | acc_lcb | uncertainty | samples | payload KB | quality gap | semantic success | deadline vio | final success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.276 | 0.456 | 0.433 | 0.090 | 281.9 | 0.000 | 0.392 | 0.227 | 0.000 | 0.227 |
| 1 | 0.697 | 0.903 | 0.863 | 0.094 | 280.7 | 1.048 | 0.000 | 0.998 | 0.965 | 0.035 |
| 2 | 0.027 | 0.952 | 0.919 | 0.087 | 276.5 | 37.353 | 0.001 | 0.979 | 1.000 | 0.000 |

### Policy Contrast

| policy | semantic success | task success | LCB | mean accuracy | uncertainty | quality gap | deadline vio | payload KB | cache/token/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| always_cache | 0.053 | 0.053 | 0.343 | 0.365 | 0.084 | 0.465 | 0.000 | 0.000 | 1.000/0.000/0.000 |
| always_semantic_token | 0.661 | 0.018 | 0.634 | 0.673 | 0.092 | 0.210 | 0.955 | 1.047 | 0.000/1.000/0.000 |
| always_image | 0.673 | 0.002 | 0.623 | 0.644 | 0.073 | 0.265 | 0.988 | 53.872 | 0.000/0.000/1.000 |
| semantic_greedy | 0.950 | 0.087 | 0.856 | 0.897 | 0.097 | 0.005 | 0.913 | 2.557 | 0.063/0.887/0.050 |
| proposed_ppo | 0.785 | 0.087 | 0.746 | 0.781 | 0.093 | 0.108 | 0.699 | 1.726 | 0.276/0.697/0.027 |
| ppo_without_lcb | 0.785 | 0.087 | 0.750 | 0.785 | 0.093 | 0.108 | 0.703 | 4.065 | 0.273/0.649/0.078 |
| ppo_without_queues | 0.784 | 0.087 | 0.744 | 0.779 | 0.093 | 0.109 | 0.697 | 0.756 | 0.278/0.722/0.000 |
| ppo_without_projection | 0.358 | 0.046 | 0.482 | 0.509 | 0.086 | 0.346 | 0.368 | 4.467 | 0.631/0.258/0.111 |

## edge_overload

high CPU/GPU queue load and small model cache; semantic quality may be decent while deadline/resource queues dominate.

### Proposed PPO Task/Condition Distribution

- `question_type`: counting: 388 (21.6%); presence: 1412 (78.4%)
- `risk_level`: critical: 1754 (97.4%); normal: 46 (2.6%)
- `snr_bin`: -5dB: 557 (30.9%); 20dB: 1243 (69.1%)
- `view_quality_bin`: good: 15 (0.8%); medium: 1775 (98.6%); poor: 10 (0.6%)
- `freshness_bin`: expired: 1500 (83.3%); stale: 300 (16.7%)
- `service_level`: 0: 557 (30.9%); 1: 1243 (69.1%)
- failed-row exact epsilon range: 0.800 to 0.820, mean 0.812, n=1800

### Selected-Service Utility Breakdown

| service | selected | acc_mean | acc_lcb | uncertainty | samples | payload KB | quality gap | semantic success | deadline vio | final success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.309 | 0.662 | 0.559 | 0.359 | 36.8 | 0.000 | 0.253 | 0.000 | 0.000 | 0.000 |
| 1 | 0.691 | 0.980 | 0.750 | 0.395 | 13.0 | 1.188 | 0.062 | 0.000 | 0.769 | 0.000 |
| 2 | 0.000 | - | - | - | - | - | - | - | - | - |

### Policy Contrast

| policy | semantic success | task success | LCB | mean accuracy | uncertainty | quality gap | deadline vio | payload KB | cache/token/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| always_cache | 0.000 | 0.000 | 0.267 | 0.354 | 0.371 | 0.545 | 0.000 | 0.000 | 1.000/0.000/0.000 |
| always_semantic_token | 0.000 | 0.000 | 0.588 | 0.827 | 0.415 | 0.225 | 0.671 | 1.177 | 0.000/1.000/0.000 |
| always_image | 0.000 | 0.000 | 0.516 | 0.669 | 0.391 | 0.297 | 1.000 | 201.578 | 0.000/0.000/1.000 |
| semantic_greedy | 0.000 | 0.000 | 0.516 | 0.669 | 0.391 | 0.297 | 1.000 | 201.578 | 0.000/0.000/1.000 |
| proposed_ppo | 0.000 | 0.000 | 0.691 | 0.881 | 0.384 | 0.121 | 0.531 | 0.820 | 0.309/0.691/0.000 |
| ppo_without_lcb | 0.000 | 0.000 | 0.691 | 0.882 | 0.384 | 0.121 | 0.534 | 0.821 | 0.309/0.691/0.000 |
| ppo_without_queues | 0.000 | 0.000 | 0.683 | 0.881 | 0.383 | 0.129 | 0.550 | 0.894 | 0.248/0.752/0.000 |
| ppo_without_projection | 0.000 | 0.000 | 0.630 | 0.864 | 0.411 | 0.184 | 0.957 | 1.151 | 0.031/0.969/0.000 |

## utm_conflict

conflict-heavy Area4D/UTM state; quality is coupled with deadline and risk pressure.

### Proposed PPO Task/Condition Distribution

- `question_type`: counting: 869 (57.9%); presence: 631 (42.1%)
- `risk_level`: critical: 1489 (99.3%); normal: 11 (0.7%)
- `snr_bin`: -5dB: 630 (42.0%); 20dB: 870 (58.0%)
- `view_quality_bin`: medium: 1489 (99.3%); poor: 11 (0.7%)
- `freshness_bin`: expired: 1359 (90.6%); stale: 141 (9.4%)
- `service_level`: 0: 630 (42.0%); 1: 870 (58.0%)
- failed-row exact epsilon range: 0.820 to 0.820, mean 0.820, n=1500

### Selected-Service Utility Breakdown

| service | selected | acc_mean | acc_lcb | uncertainty | samples | payload KB | quality gap | semantic success | deadline vio | final success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.420 | 0.645 | 0.536 | 0.380 | 13.0 | 0.000 | 0.284 | 0.000 | 0.000 | 0.000 |
| 1 | 0.580 | 0.851 | 0.614 | 0.411 | 28.4 | 1.183 | 0.206 | 0.000 | 0.305 | 0.000 |
| 2 | 0.000 | - | - | - | - | - | - | - | - | - |

### Policy Contrast

| policy | semantic success | task success | LCB | mean accuracy | uncertainty | quality gap | deadline vio | payload KB | cache/token/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| always_cache | 0.000 | 0.000 | 0.313 | 0.405 | 0.374 | 0.507 | 0.000 | 0.000 | 1.000/0.000/0.000 |
| always_semantic_token | 0.000 | 0.000 | 0.463 | 0.714 | 0.436 | 0.357 | 0.287 | 1.186 | 0.000/1.000/0.000 |
| always_image | 0.000 | 0.000 | 0.359 | 0.465 | 0.391 | 0.461 | 0.960 | 198.388 | 0.000/0.000/1.000 |
| semantic_greedy | 0.000 | 0.000 | 0.359 | 0.465 | 0.391 | 0.461 | 0.960 | 198.388 | 0.000/0.000/1.000 |
| proposed_ppo | 0.000 | 0.000 | 0.581 | 0.765 | 0.398 | 0.239 | 0.177 | 0.686 | 0.420/0.580/0.000 |
| ppo_without_lcb | 0.002 | 0.000 | 0.571 | 0.763 | 0.400 | 0.249 | 0.179 | 1.062 | 0.361/0.637/0.002 |
| ppo_without_queues | 0.001 | 0.000 | 0.578 | 0.764 | 0.399 | 0.242 | 0.177 | 0.812 | 0.400/0.599/0.001 |
| ppo_without_projection | 0.000 | 0.000 | 0.357 | 0.466 | 0.379 | 0.463 | 0.103 | 0.122 | 0.897/0.103/0.000 |

## Focus Diagnostics

### Why edge_overload has LCB around 0.691 but semantic success is 0

`edge_overload` proposed PPO selects cache/token only (`0.309/0.691/0.000`). The selected-token LCB is useful, but the scenario floors are `normal=0.60` and `critical=0.80`; many critical token/cache rows remain below epsilon. The average semantic gap is still 0.121, so the LCB is close but not enough. Separately, deadline violation is 0.531 because edge queues are heavy; that explains why final task success also remains impossible even when some rows approach the quality floor.

### Why utm_conflict has LCB around 0.581 but semantic success is 0

`utm_conflict` uses `normal=0.60` and `critical=0.82`, while proposed PPO averages only 0.581 LCB and keeps a 0.420 cache ratio. That puts the average below the normal floor and far below the critical floor. The semantic gap remains 0.239, so this is primarily a conservative-quality failure, not only an airspace failure. UTM delay/risk then further reduces final success.

### Is disaster_hotspot epsilon too high?

Partly yes, but not globally. The floor `critical=0.84` is above the average LCB of semantic tokens in many poor/stale cells, and the scenario intentionally has a critical-heavy burst. Lowering all critical epsilon would weaken the emergency story. A better fix is layered epsilon: lower/medium threshold for presence with fresh/good token evidence, high threshold for critical counting/risk, and optional ROI/image escalation for high-risk counting.

### Does low_snr_blockage show semantic_greedy is stronger?

Yes. `semantic_greedy` reaches 0.950 semantic success versus proposed PPO at 0.785, with a small quality gap of 0.005. This means the utility model contains feasible token/image choices, but PPO has not fully learned the routing policy. The next algorithm step should use a stronger semantic-token exploration prior, behavior cloning/distillation from greedy/oracle, or a projection that prevents avoidable cache choices when a token cell satisfies LCB.

### Is cache LCB too low but still attractive?

Yes. The global cache mean LCB is about 0.461 with zero payload and very low delay/energy. This makes cache attractive to reward terms unless the semantic shortfall penalty is strong and risk-aware. Cache should be allowed for low-risk/fresh rows, but penalized more aggressively for critical or high-epsilon rows when `accuracy_lcb < epsilon_k`.

## Recommendations for Algorithm and Environment Threads

| item | recommendation |
|---|---|
| Scenario epsilon | Keep nominal and low-SNR floors; for disaster/UTM use layered epsilon by task type and risk rather than lowering all critical thresholds. |
| High-risk tasks | Split critical presence, critical counting, and UTM-risk tasks. Presence can accept token LCB when fresh/good; counting should require higher LCB or ROI/image escalation. |
| Cache penalty | Make cache shortfall penalty depend on `risk_level` and `epsilon_k`; cache should not be a cheap default for critical stale/expired rows. |
| Uncertainty penalty | Current uncertainty is useful but not dominant enough to prevent sparse/cache attraction. Use `accuracy_lcb - beta*uncertainty` or increase uncertainty penalty for critical tasks. |
| Token exploration prior | Increase token prior in low-SNR and edge-overload scenarios. Tokens provide much lower payload than images and higher LCB than cache, but PPO still sometimes collapses to cache. |
| Edge overload | Add compute-aware service projection: if image violates queue/deadline, prefer token; if token LCB still below epsilon, mark infeasible instead of cache-collapsing. |
| UTM conflict | Couple risk queue with service routing: nonconforming/contingent critical rows should raise cache penalty and encourage high-confidence token/image only when deadline feasible. |
| Trace schema | Persist `epsilon_k` directly in rollout records. Current diagnostics can reconstruct failed epsilon exactly, but successful-row epsilon requires scenario-floor approximation. |
