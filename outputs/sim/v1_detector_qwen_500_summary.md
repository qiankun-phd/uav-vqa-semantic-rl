# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.283 | 0.571 | 0.689 | 0.200 | 0.000 | 1.000 | 0.717 | 0.000 |
| always_light | 0.519 | 0.595 | 2.073 | 1.000 | 0.560 | 0.994 | 0.480 | 0.000 |
| always_image | 0.100 | 0.540 | 3.902 | 2.500 | 94.680 | 0.000 | 0.730 | 0.343 |
| greedy_min_sufficient_evidence | 0.692 | 0.778 | 2.251 | 1.239 | 28.680 | 0.697 | 0.308 | 0.000 |
| oracle_best_feasible_evidence | 0.692 | 0.787 | 1.948 | 1.028 | 21.986 | 0.768 | 0.308 | 0.000 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 |
|---|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.279 | 0.412 | 0.308 |
| oracle_best_feasible_evidence | 0.370 | 0.414 | 0.216 |

## Approximate 95% CI for Rate Metrics

| policy | success CI | quality violation CI | deadline violation CI | tasks |
|---|---:|---:|---:|---:|
| always_cache | [0.273, 0.293] | [0.707, 0.727] | [0.000, 0.000] | 8000 |
| always_light | [0.509, 0.530] | [0.470, 0.491] | [0.000, 0.000] | 8000 |
| always_image | [0.093, 0.107] | [0.720, 0.739] | [0.332, 0.353] | 8000 |
| greedy_min_sufficient_evidence | [0.682, 0.702] | [0.298, 0.318] | [0.000, 0.000] | 8000 |
| oracle_best_feasible_evidence | [0.682, 0.702] | [0.298, 0.318] | [0.000, 0.000] | 8000 |
