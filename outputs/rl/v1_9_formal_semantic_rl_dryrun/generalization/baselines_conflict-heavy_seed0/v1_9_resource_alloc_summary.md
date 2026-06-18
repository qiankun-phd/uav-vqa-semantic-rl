# V1.9 LUT Resource Allocation Summary

- rollout rows: 62
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.083 | 0.411 | 0.121 | 1.610 | 0.000 | 1.000 | 0.917 | 0.000 | 0.000 | 0.000 | 0.000 | -0.655 |
| always_light | 0.000 | 0.401 | 1.682 | 186.728 | 1.158 | 0.994 | 1.000 | 0.083 | 0.000 | 0.000 | 0.000 | -1.497 |
| always_image | 0.000 | 0.000 | 4.441 | 360.894 | 184.215 | 0.000 | 1.000 | 0.333 | 0.000 | 0.000 | 0.000 | -2.734 |
| greedy_min_sufficient_evidence | 0.500 | 0.890 | 3.441 | 440.132 | 0.596 | 0.997 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | -0.056 |
| no_cache_greedy | 0.000 | 0.000 | 4.441 | 360.894 | 184.215 | 0.000 | 1.000 | 0.333 | 0.000 | 0.000 | 0.000 | -2.734 |
| no_semantic_tokens_greedy | 0.333 | 0.727 | 5.296 | 514.511 | 128.335 | 0.303 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 | -1.382 |
| oracle_best_feasible_evidence | 0.500 | 0.890 | 3.441 | 440.132 | 0.596 | 0.997 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | -0.056 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.250 | 0.750 | 0.000 | 0.000 |
| no_cache_greedy | 0.000 | 0.000 | 1.000 | 0.000 |
| no_semantic_tokens_greedy | 0.167 | 0.000 | 0.833 | 0.000 |
| oracle_best_feasible_evidence | 0.250 | 0.750 | 0.000 | 0.000 |
