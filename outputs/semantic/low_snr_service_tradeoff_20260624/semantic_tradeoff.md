# Low-SNR Service Tradeoff Diagnosis

This report diagnoses the semantic-utility side of `low_snr_blockage` without modifying the original LUT or VLM prediction data.

## Inputs

- Semantic utility CSV: `outputs/lut/v1_9_semantic_utility_with_ci.csv`
- Low-SNR bins: `-5dB, 0dB`
- Benchmark context:
  - `outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv`
  - `outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv`
  - `outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv`

## Service-Level Low-SNR Utility Summary

| service_level | service_name | cells | accuracy_lcb_mean | accuracy_lcb_p10 | accuracy_lcb_p50 | accuracy_lcb_p90 | payload_kb_mean | payload_kb_p50 | payload_kb_p90 | uncertainty_mean | sample_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cache_answer | 72 | 0.461 | 0.164 | 0.467 | 0.747 | 0.000 | 0.000 | 0.000 | 0.268 | 247.750 |
| 1 | semantic_token | 72 | 0.493 | 0.167 | 0.510 | 0.772 | 0.870 | 0.820 | 1.083 | 0.323 | 247.750 |
| 2 | image_evidence | 72 | 0.476 | 0.000 | 0.516 | 0.830 | 42.899 | 43.344 | 55.510 | 0.308 | 247.750 |

## Token vs Cache Semantic Gain

- Mean token LCB gain over cache across low-SNR cells: `0.032`.
- Mean token payload: `0.870 KB`; P90 token payload: `1.083 KB`.
- Mean image payload: `42.899 KB`; P90 image payload: `55.510 KB`.
- Mean image LCB gain over token: `-0.017`.

By semantic utility alone, service level 1 is the right low-SNR **main candidate** when its LCB clears `epsilon_k`: it keeps payload near the semantic-token scale and gives clear gains for presence tasks. It is not universally stronger than cache, especially for critical counting cells where detector count errors make token LCB conservative. Image evidence can improve or match LCB for some cells, but its payload is tens of times larger.

## Breakdown by Task, Risk, View, and Freshness

| question_type | risk_level | view_quality_bin | freshness_bin | cache_lcb | token_lcb | image_lcb | token_gain_vs_cache | image_gain_vs_token | token_payload_kb | image_payload_kb | image_payload_multiplier_vs_token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| counting | critical | good | expired | 0.314 | 0.301 | 0.000 | -0.014 | -0.301 | 0.934 | 37.697 | 40.019 |
| counting | critical | good | fresh | 0.314 | 0.301 | 0.000 | -0.014 | -0.301 | 0.934 | 37.697 | 40.019 |
| counting | critical | good | stale | 0.120 | 0.301 | 0.000 | 0.181 | -0.301 | 0.934 | 37.697 | 40.019 |
| counting | critical | medium | expired | 0.504 | 0.048 | 0.000 | -0.457 | -0.048 | 1.060 | 47.366 | 44.509 |
| counting | critical | medium | fresh | 0.504 | 0.048 | 0.000 | -0.457 | -0.048 | 1.060 | 47.366 | 44.509 |
| counting | critical | medium | stale | 0.750 | 0.048 | 0.000 | -0.702 | -0.048 | 1.060 | 47.366 | 44.509 |
| counting | critical | poor | expired | 0.848 | 0.179 | 0.000 | -0.669 | -0.179 | 1.056 | 46.382 | 43.731 |
| counting | critical | poor | fresh | 0.912 | 0.179 | 0.000 | -0.734 | -0.179 | 1.056 | 46.382 | 43.731 |
| counting | critical | poor | stale | 0.891 | 0.179 | 0.000 | -0.712 | -0.179 | 1.056 | 46.382 | 43.731 |
| counting | normal | good | expired | 0.300 | 0.557 | 0.538 | 0.258 | -0.019 | 0.731 | 38.585 | 52.710 |
| counting | normal | good | fresh | 0.494 | 0.557 | 0.538 | 0.064 | -0.019 | 0.731 | 38.585 | 52.710 |
| counting | normal | good | stale | 0.395 | 0.557 | 0.538 | 0.162 | -0.019 | 0.731 | 38.585 | 52.710 |
| counting | normal | medium | expired | 0.369 | 0.311 | 0.371 | -0.058 | 0.061 | 0.743 | 40.999 | 55.012 |
| counting | normal | medium | fresh | 0.502 | 0.311 | 0.371 | -0.191 | 0.061 | 0.743 | 40.999 | 55.012 |
| counting | normal | medium | stale | 0.584 | 0.311 | 0.371 | -0.273 | 0.061 | 0.743 | 40.999 | 55.012 |
| counting | normal | poor | expired | 0.297 | 0.401 | 0.427 | 0.104 | 0.026 | 0.730 | 43.494 | 59.385 |
| counting | normal | poor | fresh | 0.591 | 0.401 | 0.427 | -0.191 | 0.026 | 0.730 | 43.494 | 59.385 |
| counting | normal | poor | stale | 0.430 | 0.401 | 0.427 | -0.030 | 0.026 | 0.730 | 43.494 | 59.385 |
| presence | critical | good | expired | 0.120 | 0.510 | 0.510 | 0.390 | 0.000 | 0.934 | 37.697 | 40.019 |
| presence | critical | good | fresh | 0.314 | 0.510 | 0.510 | 0.196 | 0.000 | 0.934 | 37.697 | 40.019 |
| presence | critical | good | stale | 0.000 | 0.510 | 0.510 | 0.510 | 0.000 | 0.934 | 37.697 | 40.019 |
| presence | critical | medium | expired | 0.151 | 0.772 | 0.772 | 0.621 | 0.000 | 1.061 | 51.683 | 48.502 |
| presence | critical | medium | fresh | 0.504 | 0.772 | 0.772 | 0.267 | 0.000 | 1.061 | 51.683 | 48.502 |
| presence | critical | medium | stale | 0.583 | 0.772 | 0.772 | 0.189 | 0.000 | 1.061 | 51.683 | 48.502 |
| presence | critical | poor | expired | 0.281 | 0.864 | 0.927 | 0.583 | 0.063 | 1.052 | 46.498 | 44.017 |
| presence | critical | poor | fresh | 0.621 | 0.864 | 0.927 | 0.243 | 0.063 | 1.052 | 46.498 | 44.017 |
| presence | critical | poor | stale | 0.506 | 0.864 | 0.927 | 0.357 | 0.063 | 1.052 | 46.498 | 44.017 |
| presence | normal | good | expired | 0.298 | 0.683 | 0.801 | 0.385 | 0.118 | 0.709 | 38.529 | 54.325 |
| presence | normal | good | fresh | 0.623 | 0.683 | 0.801 | 0.060 | 0.118 | 0.709 | 38.529 | 54.325 |
| presence | normal | good | stale | 0.419 | 0.683 | 0.801 | 0.264 | 0.118 | 0.709 | 38.529 | 54.325 |
| presence | normal | medium | expired | 0.350 | 0.722 | 0.709 | 0.372 | -0.013 | 0.717 | 41.305 | 57.458 |
| presence | normal | medium | fresh | 0.720 | 0.722 | 0.709 | 0.002 | -0.013 | 0.717 | 41.305 | 57.458 |
| presence | normal | medium | stale | 0.440 | 0.722 | 0.709 | 0.282 | -0.013 | 0.717 | 41.305 | 57.458 |
| presence | normal | poor | expired | 0.360 | 0.573 | 0.656 | 0.213 | 0.083 | 0.713 | 44.549 | 62.353 |
| presence | normal | poor | fresh | 0.665 | 0.573 | 0.656 | -0.091 | 0.083 | 0.713 | 44.549 | 62.353 |
| presence | normal | poor | stale | 0.526 | 0.573 | 0.656 | 0.048 | 0.083 | 0.713 | 44.549 | 62.353 |

## Compact Breakdown by Task and Risk

| question_type | risk_level | cache_lcb | token_lcb | image_lcb | token_gain_vs_cache | image_gain_vs_token | token_payload_kb | image_payload_kb | image_payload_multiplier_vs_token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| counting | critical | 0.573 | 0.176 | 0.000 | -0.397 | -0.176 | 1.017 | 43.815 | 42.753 |
| counting | normal | 0.440 | 0.423 | 0.445 | -0.017 | 0.023 | 0.735 | 41.026 | 55.702 |
| presence | critical | 0.342 | 0.715 | 0.736 | 0.373 | 0.021 | 1.016 | 45.293 | 44.179 |
| presence | normal | 0.489 | 0.659 | 0.722 | 0.170 | 0.063 | 0.713 | 41.461 | 58.045 |

## Cache Fallback Threshold

A deadline-aware fallback should not require cache to fully satisfy the semantic threshold. It can be allowed when cache is close enough and the alternative token/image service would violate deadline. Define:

```text
cache_gap = max(0, epsilon_k - cache_accuracy_lcb)
allow_cache_deadline_fallback if cache_gap <= delta_cache and token/image is deadline-infeasible
```

| epsilon | cache_gap_threshold | acceptable_cell_ratio | mean_cache_gap | p90_cache_gap |
| --- | --- | --- | --- | --- |
| 0.700 | 0.050 | 0.167 | 0.256 | 0.536 |
| 0.700 | 0.080 | 0.222 | 0.256 | 0.536 |
| 0.750 | 0.050 | 0.139 | 0.300 | 0.586 |
| 0.750 | 0.080 | 0.139 | 0.300 | 0.586 |
| 0.800 | 0.050 | 0.111 | 0.346 | 0.636 |
| 0.800 | 0.080 | 0.139 | 0.346 | 0.636 |
| 0.840 | 0.050 | 0.083 | 0.382 | 0.676 |
| 0.840 | 0.080 | 0.083 | 0.382 | 0.676 |


Recommendation: expose a `deadline_aware_semantic_fallback_threshold` to Algorithm. Start with `delta_cache = 0.05` for strict runs and `delta_cache = 0.08` for stress-scenario runs. The fallback should only fire when the deadline queue is high or no candidate service is jointly feasible; otherwise the controller should still prefer semantic tokens.

## Image: Semantically Strong but Deadline-Infeasible

| source | benchmark_policy | semantic_success_rate_mean | task_success_rate_mean | average_accuracy_mean | average_semantic_quality_gap_mean | average_delay_mean | average_payload_kb_mean | deadline_violation_rate_mean | service_level_0_ratio_mean | service_level_1_ratio_mean | service_level_2_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | always_cache | 0.028 | 0.028 | 0.329 | 0.478 | 0.174 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | always_image | 0.825 | 0.004 | 0.766 | 0.139 | 30.832 | 64.261 | 0.983 | 0.000 | 0.000 | 1.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | always_semantic_token | 0.794 | 0.029 | 0.730 | 0.122 | 12.298 | 1.056 | 0.932 | 0.000 | 1.000 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | lyapunov_greedy | 0.028 | 0.028 | 0.329 | 0.478 | 0.174 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | proposed_two_timescale_ppo | 1.000 | 0.028 | 0.922 | 0.000 | 666.307 | 36.446 | 0.972 | 0.028 | 0.000 | 0.972 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | proposed_v2_deadline_guard | 0.933 | 0.058 | 0.844 | 0.010 | 26.674 | 0.968 | 0.938 | 0.037 | 0.963 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | proposed_v2_nearest_uav_mobility | 0.696 | 0.106 | 0.757 | 0.091 | 15.394 | 1.494 | 0.714 | 0.237 | 0.738 | 0.025 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | proposed_v2_no_image_under_low_snr | 0.933 | 0.058 | 0.844 | 0.010 | 26.674 | 0.968 | 0.938 | 0.037 | 0.963 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_20260623/scenario_comparison_summary.csv | semantic_greedy | 0.996 | 0.081 | 0.863 | 0.001 | 11.752 | 1.136 | 0.919 | 0.047 | 0.949 | 0.004 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | always_cache | 0.028 | 0.028 | 0.329 | 0.478 | 0.174 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | always_image | 0.825 | 0.004 | 0.766 | 0.139 | 30.832 | 64.261 | 0.983 | 0.000 | 0.000 | 1.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | always_semantic_token | 0.794 | 0.029 | 0.730 | 0.122 | 12.298 | 1.056 | 0.932 | 0.000 | 1.000 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | lyapunov_greedy | 0.028 | 0.028 | 0.329 | 0.478 | 0.174 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | proposed_two_timescale_ppo | 1.000 | 0.028 | 0.922 | 0.000 | 666.307 | 36.446 | 0.972 | 0.028 | 0.000 | 0.972 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | proposed_v2_deadline_guard | 0.933 | 0.058 | 0.844 | 0.010 | 26.674 | 0.968 | 0.938 | 0.037 | 0.963 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | proposed_v2_nearest_uav_mobility | 0.696 | 0.106 | 0.757 | 0.091 | 15.394 | 1.494 | 0.714 | 0.237 | 0.738 | 0.025 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | proposed_v2_no_image_under_low_snr | 0.933 | 0.058 | 0.844 | 0.010 | 26.674 | 0.968 | 0.938 | 0.037 | 0.963 | 0.000 |
| outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/scenario_comparison_summary.csv | semantic_greedy | 0.996 | 0.081 | 0.863 | 0.001 | 11.752 | 1.136 | 0.919 | 0.047 | 0.949 | 0.004 |
| outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv | always_cache | 0.053 | 0.053 | 0.343 | 0.465 | 0.173 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv | always_image | 0.673 | 0.002 | 0.623 | 0.265 | 35.056 | 53.872 | 0.988 | 0.000 | 0.000 | 1.000 |
| outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv | always_semantic_token | 0.661 | 0.018 | 0.634 | 0.210 | 13.674 | 1.047 | 0.955 | 0.000 | 1.000 | 0.000 |
| outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv | lyapunov_greedy | 0.053 | 0.053 | 0.343 | 0.465 | 0.173 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv | proposed_two_timescale_ppo | 0.955 | 0.053 | 0.912 | 0.004 | 509.965 | 34.288 | 0.939 | 0.053 | 0.030 | 0.917 |
| outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv | semantic_greedy | 0.950 | 0.087 | 0.856 | 0.005 | 15.411 | 2.557 | 0.913 | 0.063 | 0.887 | 0.050 |


The benchmark context confirms the mechanism: `always_image` can have high semantic success, but deadline violation is near one in `low_snr_blockage`. `always_semantic_token` sharply reduces payload but can still violate deadlines because the scenario also contains arrival/mobility delay under blockage. Therefore, semantic utility supports token as the main low-SNR service for many presence/normal tasks, but the algorithm still needs task-aware and deadline-aware fallback/projection rather than simply maximizing semantic LCB.

## Conclusions for Algorithm

1. **Service level 1 should be the low-SNR main service candidate** when `accuracy_lcb >= epsilon_k`: it provides the best semantic gain per payload for presence and several normal-risk cells while avoiding image overuse.
2. **Image evidence is not the default low-SNR service**: it may be semantically strong, but its large payload makes it deadline-infeasible in the blockage stress scenario.
3. **Cache fallback is acceptable only as a deadline safety valve**: allow cache when `cache_gap <= 0.05` in strict settings or `<= 0.08` in stress settings, and only when token/image are deadline-infeasible or deadline queues are high.
4. **Critical counting needs special care**: detector-token count errors make semantic tokens conservative for some critical counting cells, so fallback/projection should consider task type and risk level rather than using one service rule for all low-SNR tasks.
5. **Do not weaken the LUT**: the issue is not the semantic utility table. The table correctly exposes that token/image are semantically stronger for many cells but not all cells; the controller needs a conservative fallback threshold to trade a small semantic gap for a large deadline gain.
6. **Expose the threshold explicitly**: add Algorithm-side config such as `deadline_aware_semantic_fallback_threshold: 0.05` and optionally `stress_fallback_threshold: 0.08` for low-SNR stress presets.

## Limitations

- This is a LUT/summary-level analysis; it does not rerun Qwen, retrain detector, or change environment dynamics.
- Cache fallback ratios are evaluated over low-SNR utility cells, not over a new rollout distribution.
- Deadline feasibility in the benchmark includes mobility/arrival and queue effects; semantic payload alone does not fully determine task success.
