# V1.9 LUT Resource Allocation Summary

- rollout rows: 320
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.025 | 0.326 | 0.095 | 1.450 | 0.000 | 1.000 | 0.975 | 0.000 | -0.912 |
| always_light | 0.125 | 0.672 | 5.148 | 680.022 | 1.022 | 0.998 | 0.300 | 0.650 | -1.837 |
| always_image | 0.000 | 0.833 | 4.871 | 374.731 | 498.325 | 0.000 | 0.000 | 1.000 | -3.121 |
| greedy_min_sufficient_evidence | 0.100 | 0.832 | 5.936 | 780.997 | 14.456 | 0.971 | 0.025 | 0.700 | -1.921 |
| no_cache_greedy | 0.125 | 0.603 | 6.010 | 728.015 | 120.945 | 0.757 | 0.275 | 0.875 | -2.469 |
| no_semantic_tokens_greedy | 0.025 | 0.832 | 5.694 | 499.132 | 479.884 | 0.037 | 0.025 | 0.975 | -3.215 |
| oracle_best_feasible_evidence | 0.100 | 0.832 | 5.936 | 780.997 | 14.456 | 0.971 | 0.025 | 0.700 | -1.921 |
| ppo | 0.125 | 0.672 | 5.148 | 680.022 | 1.022 | 0.998 | 0.300 | 0.650 | -1.837 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.025 | 0.925 | 0.050 | 0.000 |
| no_cache_greedy | 0.000 | 0.700 | 0.300 | 0.000 |
| no_semantic_tokens_greedy | 0.025 | 0.000 | 0.975 | 0.000 |
| oracle_best_feasible_evidence | 0.025 | 0.925 | 0.050 | 0.000 |
| ppo | 0.000 | 1.000 | 0.000 | 0.000 |
