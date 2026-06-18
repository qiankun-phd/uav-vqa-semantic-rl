# V1.9 LUT Resource Allocation Summary

- rollout rows: 56000
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.017 | 0.411 | 0.091 | 1.610 | 0.000 | 1.000 | 0.983 | 0.000 | -0.951 |
| always_light | 0.019 | 0.483 | 2.415 | 296.701 | 1.146 | 0.994 | 0.857 | 0.260 | -1.650 |
| always_image | 0.000 | 0.439 | 4.475 | 359.267 | 184.118 | 0.000 | 0.553 | 1.000 | -2.959 |
| greedy_min_sufficient_evidence | 0.050 | 0.280 | 5.017 | 511.405 | 132.701 | 0.279 | 0.711 | 0.920 | -2.935 |
| no_cache_greedy | 0.019 | 0.149 | 4.818 | 446.278 | 157.979 | 0.142 | 0.846 | 0.969 | -3.202 |
| no_semantic_tokens_greedy | 0.018 | 0.723 | 4.529 | 372.561 | 180.671 | 0.019 | 0.265 | 0.982 | -2.612 |
| oracle_best_feasible_evidence | 0.049 | 0.820 | 2.529 | 286.506 | 47.293 | 0.743 | 0.697 | 0.270 | -1.516 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.018 | 0.262 | 0.720 | 0.000 |
| no_cache_greedy | 0.000 | 0.143 | 0.857 | 0.000 |
| no_semantic_tokens_greedy | 0.017 | 0.000 | 0.983 | 0.000 |
| oracle_best_feasible_evidence | 0.691 | 0.051 | 0.258 | 0.000 |
