# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.267 | 0.551 | 0.691 | 0.200 | 0.000 | 1.000 | 0.734 | 0.000 |
| always_light | 0.201 | 0.604 | 2.069 | 1.000 | 0.846 | 0.991 | 0.799 | 0.000 |
| always_image | 0.421 | 0.589 | 3.931 | 2.500 | 94.177 | 0.000 | 0.437 | 0.327 |
| always_roi | 0.420 | 0.514 | 2.992 | 1.800 | 146.974 | -0.561 | 0.567 | 0.107 |
| greedy_min_sufficient_evidence | 0.713 | 0.729 | 2.452 | 1.412 | 77.068 | 0.182 | 0.273 | 0.047 |
| no_cache_greedy | 0.575 | 0.621 | 3.182 | 1.925 | 114.948 | -0.221 | 0.410 | 0.185 |
| no_semantic_tokens_greedy | 0.603 | 0.724 | 2.740 | 1.652 | 96.379 | -0.023 | 0.286 | 0.145 |
| no_roi_greedy | 0.713 | 0.703 | 2.672 | 1.581 | 50.746 | 0.461 | 0.286 | 0.047 |
| oracle_best_feasible_evidence | 0.713 | 0.723 | 2.203 | 1.208 | 33.887 | 0.640 | 0.271 | 0.018 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| always_roi | 0.000 | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.266 | 0.191 | 0.273 | 0.270 |
| no_cache_greedy | 0.000 | 0.202 | 0.410 | 0.387 |
| no_semantic_tokens_greedy | 0.277 | 0.000 | 0.422 | 0.301 |
| no_roi_greedy | 0.274 | 0.192 | 0.534 | 0.000 |
| oracle_best_feasible_evidence | 0.353 | 0.299 | 0.304 | 0.043 |

## Approximate 95% CI for Rate Metrics

| policy | success CI | quality violation CI | deadline violation CI | tasks |
|---|---:|---:|---:|---:|
| always_cache | [0.257, 0.276] | [0.724, 0.743] | [0.000, 0.000] | 8000 |
| always_light | [0.192, 0.210] | [0.790, 0.808] | [0.000, 0.000] | 8000 |
| always_image | [0.410, 0.432] | [0.427, 0.448] | [0.316, 0.337] | 8000 |
| always_roi | [0.409, 0.431] | [0.556, 0.577] | [0.100, 0.114] | 8000 |
| greedy_min_sufficient_evidence | [0.703, 0.723] | [0.263, 0.283] | [0.042, 0.052] | 8000 |
| no_cache_greedy | [0.564, 0.586] | [0.399, 0.421] | [0.176, 0.193] | 8000 |
| no_semantic_tokens_greedy | [0.593, 0.614] | [0.276, 0.296] | [0.137, 0.152] | 8000 |
| no_roi_greedy | [0.703, 0.723] | [0.277, 0.296] | [0.042, 0.052] | 8000 |
| oracle_best_feasible_evidence | [0.703, 0.723] | [0.261, 0.280] | [0.015, 0.021] | 8000 |
