# V1.9 LUT Resource Allocation Summary

- rollout rows: 6400
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.244 | 0.551 | 0.050 | 0.200 | 0.000 | 1.000 | 0.756 | 0.000 | -0.008 |
| always_light | 0.115 | 0.519 | 0.355 | 0.843 | 0.823 | 0.996 | 0.885 | 0.000 | -0.899 |
| always_image | 0.469 | 0.601 | 1.691 | 2.558 | 186.426 | 0.000 | 0.531 | 0.000 | 0.966 |
| greedy_min_sufficient_evidence | 0.720 | 0.733 | 1.156 | 1.857 | 106.379 | 0.429 | 0.280 | 0.000 | 2.754 |
| no_cache_greedy | 0.519 | 0.610 | 1.527 | 2.378 | 144.970 | 0.222 | 0.481 | 0.000 | 1.324 |
| no_semantic_tokens_greedy | 0.686 | 0.727 | 1.269 | 1.982 | 134.161 | 0.280 | 0.314 | 0.000 | 2.512 |
| oracle_best_feasible_evidence | 0.720 | 0.761 | 0.935 | 1.577 | 79.720 | 0.572 | 0.280 | 0.000 | 2.841 |
| ppo | 0.469 | 0.601 | 1.691 | 2.558 | 186.426 | 0.000 | 0.531 | 0.000 | 0.966 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.244 | 0.080 | 0.676 | 0.000 |
| no_cache_greedy | 0.000 | 0.115 | 0.885 | 0.000 |
| no_semantic_tokens_greedy | 0.244 | 0.000 | 0.756 | 0.000 |
| oracle_best_feasible_evidence | 0.305 | 0.158 | 0.537 | 0.000 |
| ppo | 0.000 | 0.000 | 1.000 | 0.000 |
