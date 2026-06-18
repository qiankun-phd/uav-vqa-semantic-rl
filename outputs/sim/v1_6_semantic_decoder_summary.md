# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.265 | 0.558 | 0.688 | 0.200 | 0.000 | 1.000 | 0.735 | 0.000 |
| always_light | 0.362 | 0.617 | 2.061 | 1.000 | 0.871 | 0.990 | 0.638 | 0.000 |
| always_image | 0.324 | 0.574 | 3.916 | 2.500 | 85.446 | 0.000 | 0.509 | 0.348 |
| always_roi | 0.000 | 0.000 | 2.969 | 1.800 | 78.125 | 0.086 | 1.000 | 0.110 |
| greedy_min_sufficient_evidence | 0.686 | 0.683 | 2.493 | 1.389 | 29.267 | 0.657 | 0.314 | 0.069 |
| no_cache_greedy | 0.530 | 0.574 | 3.248 | 1.957 | 51.748 | 0.394 | 0.469 | 0.182 |
| no_semantic_tokens_greedy | 0.492 | 0.681 | 3.105 | 1.890 | 60.608 | 0.291 | 0.342 | 0.236 |
| no_roi_greedy | 0.693 | 0.684 | 2.468 | 1.388 | 29.927 | 0.650 | 0.307 | 0.069 |
| oracle_best_feasible_evidence | 0.697 | 0.756 | 1.898 | 0.962 | 12.645 | 0.852 | 0.303 | 0.000 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| always_roi | 0.000 | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.278 | 0.314 | 0.408 | 0.000 |
| no_cache_greedy | 0.000 | 0.362 | 0.638 | 0.000 |
| no_semantic_tokens_greedy | 0.265 | 0.000 | 0.735 | 0.000 |
| no_roi_greedy | 0.276 | 0.319 | 0.406 | 0.000 |
| oracle_best_feasible_evidence | 0.353 | 0.483 | 0.164 | 0.000 |

## Approximate 95% CI for Rate Metrics

| policy | success CI | quality violation CI | deadline violation CI | tasks |
|---|---:|---:|---:|---:|
| always_cache | [0.251, 0.279] | [0.721, 0.749] | [0.000, 0.000] | 4000 |
| always_light | [0.347, 0.377] | [0.623, 0.653] | [0.000, 0.000] | 4000 |
| always_image | [0.310, 0.339] | [0.494, 0.525] | [0.333, 0.363] | 4000 |
| always_roi | [0.000, 0.000] | [1.000, 1.000] | [0.101, 0.120] | 4000 |
| greedy_min_sufficient_evidence | [0.671, 0.700] | [0.300, 0.329] | [0.061, 0.077] | 4000 |
| no_cache_greedy | [0.515, 0.546] | [0.454, 0.485] | [0.170, 0.194] | 4000 |
| no_semantic_tokens_greedy | [0.477, 0.508] | [0.327, 0.356] | [0.223, 0.249] | 4000 |
| no_roi_greedy | [0.678, 0.707] | [0.293, 0.322] | [0.061, 0.076] | 4000 |
| oracle_best_feasible_evidence | [0.683, 0.711] | [0.289, 0.317] | [0.000, 0.000] | 4000 |
