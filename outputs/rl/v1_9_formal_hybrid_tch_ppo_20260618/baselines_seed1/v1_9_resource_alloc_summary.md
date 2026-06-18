# V1.9 LUT Resource Allocation Summary

- rollout rows: 56000
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.017 | 0.411 | 0.091 | 1.610 | 0.000 | 1.000 | 0.983 | 0.000 | -0.951 |
| always_light | 0.019 | 0.483 | 2.407 | 295.548 | 1.146 | 0.994 | 0.858 | 0.259 | -1.647 |
| always_image | 0.000 | 0.439 | 4.474 | 359.092 | 184.117 | 0.000 | 0.553 | 1.000 | -2.959 |
| greedy_min_sufficient_evidence | 0.050 | 0.279 | 5.010 | 510.341 | 132.792 | 0.279 | 0.711 | 0.920 | -2.934 |
| no_cache_greedy | 0.019 | 0.149 | 4.812 | 445.214 | 158.069 | 0.141 | 0.846 | 0.969 | -3.201 |
| no_semantic_tokens_greedy | 0.018 | 0.723 | 4.528 | 372.386 | 180.670 | 0.019 | 0.266 | 0.982 | -2.612 |
| oracle_best_feasible_evidence | 0.049 | 0.819 | 2.527 | 286.277 | 47.246 | 0.743 | 0.698 | 0.270 | -1.515 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.018 | 0.262 | 0.721 | 0.000 |
| no_cache_greedy | 0.000 | 0.142 | 0.858 | 0.000 |
| no_semantic_tokens_greedy | 0.017 | 0.000 | 0.983 | 0.000 |
| oracle_best_feasible_evidence | 0.692 | 0.051 | 0.257 | 0.000 |
