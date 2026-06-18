# V1.9 LUT Resource Allocation Summary

- rollout rows: 14
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.000 | 0.353 | 0.213 | 1.610 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.043 |
| always_light | 0.000 | 0.534 | 6.217 | 837.935 | 0.788 | 0.995 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -3.580 |
| always_image | 0.000 | 0.491 | 8.970 | 1019.936 | 175.122 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -4.552 |
| greedy_min_sufficient_evidence | 0.000 | 0.491 | 8.970 | 1019.936 | 175.122 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -4.552 |
| no_cache_greedy | 0.000 | 0.491 | 8.970 | 1019.936 | 175.122 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -4.552 |
| no_semantic_tokens_greedy | 0.000 | 0.491 | 8.970 | 1019.936 | 175.122 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -4.552 |
| oracle_best_feasible_evidence | 0.000 | 0.534 | 6.217 | 837.935 | 0.788 | 0.995 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -3.580 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.000 | 0.000 | 1.000 | 0.000 |
| no_cache_greedy | 0.000 | 0.000 | 1.000 | 0.000 |
| no_semantic_tokens_greedy | 0.000 | 0.000 | 1.000 | 0.000 |
| oracle_best_feasible_evidence | 0.000 | 1.000 | 0.000 | 0.000 |
