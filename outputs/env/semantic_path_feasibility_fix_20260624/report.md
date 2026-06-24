# Semantic Path Feasibility Fix Smoke

Environment-only oracle/path-greedy smoke; no PPO training or LUT regeneration.

| scenario | path feasible | oracle/path-greedy success | deadline viol. | UTM viol. | cache/token/image/defer/cache_update |
|---|---:|---:|---:|---:|---:|
| normal_patrol | 0.020 | 0.067 | 0.367 | 0.000 | 0.367 / 0.633 / 0.000 / 0.000 / 0.000 |
| disaster_hotspot | 0.000 | 0.000 | 0.111 | 0.000 | 0.889 / 0.111 / 0.000 / 0.000 / 0.000 |
| low_snr_soft | 0.067 | 0.167 | 0.306 | 0.000 | 0.444 / 0.556 / 0.000 / 0.000 / 0.000 |
| low_snr_blockage | 0.022 | 0.111 | 0.370 | 0.000 | 0.630 / 0.370 / 0.000 / 0.000 / 0.000 |
| edge_overload | 0.000 | 0.000 | 0.056 | 0.000 | 0.667 / 0.333 / 0.000 / 0.000 / 0.000 |
| utm_conflict | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 / 0.000 / 0.000 / 0.000 / 0.000 |
