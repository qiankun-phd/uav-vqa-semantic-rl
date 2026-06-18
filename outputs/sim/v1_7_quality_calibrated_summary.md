# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.262 | 0.548 | 0.688 | 0.200 | 0.000 | 1.000 | 0.738 | 0.000 |
| always_light | 0.202 | 0.607 | 2.075 | 1.000 | 0.841 | 0.991 | 0.798 | 0.000 |
| always_image | 0.417 | 0.596 | 3.909 | 2.500 | 95.560 | 0.000 | 0.441 | 0.323 |
| greedy_min_sufficient_evidence | 0.697 | 0.700 | 2.743 | 1.618 | 50.613 | 0.470 | 0.302 | 0.053 |
| no_cache_greedy | 0.571 | 0.592 | 3.533 | 2.196 | 76.595 | 0.198 | 0.428 | 0.180 |
| no_semantic_tokens_greedy | 0.544 | 0.704 | 3.069 | 1.897 | 69.603 | 0.272 | 0.314 | 0.196 |
| oracle_best_feasible_evidence | 0.697 | 0.715 | 2.266 | 1.252 | 30.782 | 0.678 | 0.303 | 0.052 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.258 | 0.193 | 0.549 | 0.000 |
| no_cache_greedy | 0.000 | 0.203 | 0.797 | 0.000 |
| no_semantic_tokens_greedy | 0.262 | 0.000 | 0.738 | 0.000 |
| oracle_best_feasible_evidence | 0.351 | 0.294 | 0.355 | 0.000 |

## Approximate 95% CI for Rate Metrics

| policy | success CI | quality violation CI | deadline violation CI | tasks |
|---|---:|---:|---:|---:|
| always_cache | [0.252, 0.272] | [0.728, 0.748] | [0.000, 0.000] | 8000 |
| always_light | [0.193, 0.211] | [0.789, 0.807] | [0.000, 0.000] | 8000 |
| always_image | [0.406, 0.428] | [0.430, 0.452] | [0.313, 0.333] | 8000 |
| greedy_min_sufficient_evidence | [0.687, 0.707] | [0.292, 0.312] | [0.048, 0.058] | 8000 |
| no_cache_greedy | [0.560, 0.582] | [0.418, 0.439] | [0.172, 0.189] | 8000 |
| no_semantic_tokens_greedy | [0.533, 0.555] | [0.304, 0.324] | [0.187, 0.205] | 8000 |
| oracle_best_feasible_evidence | [0.687, 0.707] | [0.293, 0.313] | [0.047, 0.057] | 8000 |
