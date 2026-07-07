# Semantic Path Utility Diagnosis

This report checks cache/path utility recommendations for semantic-path cache-defer control. It does not modify the original VQA predictions, semantic LUT, environment dynamics, or PPO logic.

## Rule Checks

- Critical stale/expired cache recommendation violations: `0`.
- Expired cache recommendation violations: `0`.
- Cache update future reuse value available in utility layer: `False`.
- Conclusion: `cache_update` must not be recommended solely because cache is missing. The utility layer can score the current token/image update path, but it has no future reuse value model. PPO/environment logic must supply reuse value, cache pressure, or expected future hit probability before actively preferring `cache_update`.

## Scenario Path Utility Summary

| scenario | semantic_path | cells | accuracy_lcb | uncertainty | quality_gap | payload_kb | path_recommended_rate | cache_recommended_rate | critical_task_recommendation_rate | stale_expired_cache_recommendation_rate | cache_update_limitation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal_patrol | cache | 48 | 0.454 | 0.369 | 0.231 | 0.000 | 0.188 | 0.188 | 0.000 | 0.062 | 0.000 |
| normal_patrol | token | 48 | 0.590 | 0.434 | 0.153 | 0.952 | 0.458 | 0.188 | 0.000 | 0.000 | 0.000 |
| normal_patrol | image | 48 | 0.483 | 0.422 | 0.246 | 127.215 | 0.250 | 0.188 | 0.000 | 0.000 | 0.000 |
| normal_patrol | cache_update | 48 | 0.590 | 0.434 | 0.153 | 0.952 | 0.000 | 0.188 | 0.000 | 0.000 | 1.000 |
| disaster_hotspot | cache | 108 | 0.461 | 0.268 | 0.277 | 0.000 | 0.111 | 0.111 | 0.028 | 0.000 | 0.000 |
| disaster_hotspot | token | 108 | 0.571 | 0.316 | 0.193 | 0.938 | 0.333 | 0.111 | 0.083 | 0.000 | 0.000 |
| disaster_hotspot | image | 108 | 0.490 | 0.305 | 0.280 | 93.964 | 0.333 | 0.111 | 0.083 | 0.000 | 0.000 |
| disaster_hotspot | cache_update | 108 | 0.571 | 0.316 | 0.193 | 0.938 | 0.000 | 0.111 | 0.000 | 0.000 | 1.000 |
| low_snr_soft | cache | 108 | 0.461 | 0.268 | 0.228 | 0.000 | 0.167 | 0.167 | 0.028 | 0.028 | 0.000 |
| low_snr_soft | token | 108 | 0.546 | 0.316 | 0.176 | 0.917 | 0.417 | 0.167 | 0.083 | 0.000 | 0.000 |
| low_snr_soft | image | 108 | 0.485 | 0.306 | 0.242 | 69.454 | 0.333 | 0.167 | 0.083 | 0.000 | 0.000 |
| low_snr_soft | cache_update | 108 | 0.546 | 0.316 | 0.176 | 0.917 | 0.000 | 0.167 | 0.000 | 0.000 | 1.000 |
| low_snr_blockage | cache | 72 | 0.461 | 0.268 | 0.228 | 0.000 | 0.167 | 0.167 | 0.028 | 0.028 | 0.000 |
| low_snr_blockage | token | 72 | 0.493 | 0.323 | 0.211 | 0.870 | 0.375 | 0.167 | 0.083 | 0.000 | 0.000 |
| low_snr_blockage | image | 72 | 0.476 | 0.308 | 0.247 | 42.899 | 0.333 | 0.167 | 0.083 | 0.000 | 0.000 |
| low_snr_blockage | cache_update | 72 | 0.493 | 0.323 | 0.211 | 0.870 | 0.000 | 0.167 | 0.000 | 0.000 | 1.000 |
| edge_overload | cache | 48 | 0.454 | 0.369 | 0.162 | 0.000 | 0.188 | 0.188 | 0.000 | 0.062 | 0.000 |
| edge_overload | token | 48 | 0.590 | 0.434 | 0.099 | 0.952 | 0.583 | 0.188 | 0.125 | 0.000 | 0.000 |
| edge_overload | image | 48 | 0.483 | 0.422 | 0.189 | 127.215 | 0.500 | 0.188 | 0.125 | 0.000 | 0.000 |
| edge_overload | cache_update | 48 | 0.590 | 0.434 | 0.099 | 0.952 | 0.000 | 0.188 | 0.000 | 0.000 | 1.000 |
| utm_conflict | cache | 108 | 0.461 | 0.268 | 0.260 | 0.000 | 0.111 | 0.111 | 0.028 | 0.000 | 0.000 |
| utm_conflict | token | 108 | 0.571 | 0.316 | 0.180 | 0.938 | 0.389 | 0.111 | 0.083 | 0.000 | 0.000 |
| utm_conflict | image | 108 | 0.490 | 0.305 | 0.267 | 93.964 | 0.333 | 0.111 | 0.083 | 0.000 | 0.000 |
| utm_conflict | cache_update | 108 | 0.571 | 0.316 | 0.180 | 0.938 | 0.000 | 0.111 | 0.000 | 0.000 | 1.000 |

## Edge/UTM Path-Ratio Context from Existing Short Runs

| scenario | benchmark_policy | semantic_path_cache_ratio_mean | semantic_path_token_ratio_mean | semantic_path_image_ratio_mean | semantic_path_cache_update_ratio_mean | cache_eligible_ratio_mean | deadline_violation_rate_mean | task_success_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge_overload | always_cache | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| edge_overload | always_image | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| edge_overload | always_semantic_token | 0.000 | 1.000 | 0.000 | 0.000 | 0.034 | 0.895 | 0.004 |
| edge_overload | lyapunov_greedy | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| edge_overload | semantic_greedy | 0.000 | 0.080 | 0.920 | 0.000 | 0.000 | 1.000 | 0.000 |
| edge_overload | semantic_path_two_timescale_ppo | 0.034 | 0.585 | 0.000 | 0.381 | 0.131 | 0.569 | 0.231 |
| utm_conflict | always_cache | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| utm_conflict | always_image | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| utm_conflict | always_semantic_token | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| utm_conflict | lyapunov_greedy | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| utm_conflict | semantic_greedy | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| utm_conflict | semantic_path_two_timescale_ppo | 0.501 | 0.499 | 0.000 | 0.000 | 0.017 | 0.541 | 0.000 |


## Interpretation

- `edge_overload`: token paths should remain attractive because image payload is high and cache update has no utility-layer future reuse estimate. If PPO selects `cache_update`, that should be justified by environment-side future cache value, not by this utility score alone.
- `utm_conflict`: critical/stale/expired cache is blocked by the cache recommendation rule. Cache update may be useful operationally only if it reduces future UTM-constrained revisits or cache misses; that value is outside the current utility API.
- `low_snr_blockage` and `low_snr_soft`: token is the lightweight evidence path when LCB clears epsilon; cache is a deadline fallback only when fresh and semantically eligible.
- `disaster_hotspot`: stricter epsilon makes cache less eligible; cache update should be treated as a deliberate future-cache investment, not as an automatic replacement for missing cache.

## Recommendation

Keep cache/path utility as a conservative evaluator. Add a separate Algorithm/Environment-side `future_reuse_value` or `cache_update_value` field before using `cache_update` as an actively recommended path. Until then, reports and policies should treat `cache_update` as a candidate action with current-task token/image utility plus an explicit limitation.
