# V1.9 LUT Resource Allocation Summary

- rollout rows: 56000
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.017 | 0.411 | 0.091 | 1.610 | 0.000 | 1.000 | 0.983 | 0.000 | -0.951 |
| always_light | 0.019 | 0.482 | 2.397 | 294.033 | 1.146 | 0.994 | 0.859 | 0.257 | -1.644 |
| always_image | 0.000 | 0.439 | 4.475 | 359.185 | 184.117 | 0.000 | 0.553 | 1.000 | -2.959 |
| greedy_min_sufficient_evidence | 0.050 | 0.278 | 5.004 | 509.044 | 133.020 | 0.278 | 0.713 | 0.920 | -2.933 |
| no_cache_greedy | 0.019 | 0.148 | 4.805 | 443.917 | 158.298 | 0.140 | 0.848 | 0.969 | -3.201 |
| no_semantic_tokens_greedy | 0.018 | 0.723 | 4.529 | 372.480 | 180.670 | 0.019 | 0.266 | 0.982 | -2.612 |
| oracle_best_feasible_evidence | 0.049 | 0.819 | 2.515 | 284.763 | 47.039 | 0.745 | 0.699 | 0.268 | -1.512 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.018 | 0.261 | 0.722 | 0.000 |
| no_cache_greedy | 0.000 | 0.141 | 0.859 | 0.000 |
| no_semantic_tokens_greedy | 0.017 | 0.000 | 0.983 | 0.000 |
| oracle_best_feasible_evidence | 0.693 | 0.051 | 0.256 | 0.000 |
