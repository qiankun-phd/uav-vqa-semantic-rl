# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.265 | 0.554 | 0.687 | 0.200 | 0.000 | 1.000 | 0.735 | 0.000 |
| always_light | 0.526 | 0.612 | 2.075 | 1.000 | 0.562 | 0.993 | 0.474 | 0.000 |
| always_image | 0.329 | 0.586 | 3.898 | 2.500 | 85.834 | 0.000 | 0.493 | 0.347 |
| greedy_min_sufficient_evidence | 0.731 | 0.728 | 2.325 | 1.272 | 24.000 | 0.720 | 0.269 | 0.061 |
| oracle_best_feasible_evidence | 0.730 | 0.772 | 1.870 | 0.956 | 14.810 | 0.827 | 0.270 | 0.000 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 |
|---|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.265 | 0.412 | 0.323 |
| oracle_best_feasible_evidence | 0.406 | 0.406 | 0.187 |
