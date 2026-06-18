# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.265 | 0.558 | 0.688 | 0.200 | 0.000 | 1.000 | 0.735 | 0.000 |
| always_light | 0.521 | 0.660 | 2.061 | 1.000 | 0.871 | 0.990 | 0.479 | 0.000 |
| always_image | 0.324 | 0.574 | 3.916 | 2.500 | 85.446 | 0.000 | 0.509 | 0.348 |
| always_roi | 0.299 | 0.540 | 2.969 | 1.800 | 52.604 | 0.384 | 0.643 | 0.110 |
| greedy_min_sufficient_evidence | 0.729 | 0.756 | 2.322 | 1.261 | 24.247 | 0.716 | 0.271 | 0.069 |
| no_cache_greedy | 0.584 | 0.683 | 2.955 | 1.718 | 39.592 | 0.537 | 0.416 | 0.182 |
| no_semantic_tokens_greedy | 0.597 | 0.668 | 2.853 | 1.694 | 49.491 | 0.421 | 0.342 | 0.131 |
| no_roi_greedy | 0.730 | 0.757 | 2.305 | 1.265 | 25.220 | 0.705 | 0.271 | 0.069 |
| oracle_best_feasible_evidence | 0.727 | 0.808 | 1.772 | 0.892 | 11.494 | 0.865 | 0.274 | 0.000 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| always_roi | 0.000 | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.278 | 0.399 | 0.323 | 0.000 |
| no_cache_greedy | 0.000 | 0.521 | 0.479 | 0.000 |
| no_semantic_tokens_greedy | 0.265 | 0.000 | 0.455 | 0.280 |
| no_roi_greedy | 0.276 | 0.401 | 0.324 | 0.000 |
| oracle_best_feasible_evidence | 0.413 | 0.393 | 0.095 | 0.099 |

## Approximate 95% CI for Rate Metrics

| policy | success CI | quality violation CI | deadline violation CI | tasks |
|---|---:|---:|---:|---:|
| always_cache | [0.251, 0.279] | [0.721, 0.749] | [0.000, 0.000] | 4000 |
| always_light | [0.506, 0.536] | [0.464, 0.494] | [0.000, 0.000] | 4000 |
| always_image | [0.310, 0.339] | [0.494, 0.525] | [0.333, 0.363] | 4000 |
| always_roi | [0.285, 0.314] | [0.628, 0.658] | [0.101, 0.120] | 4000 |
| greedy_min_sufficient_evidence | [0.715, 0.743] | [0.257, 0.285] | [0.061, 0.077] | 4000 |
| no_cache_greedy | [0.568, 0.599] | [0.401, 0.432] | [0.170, 0.194] | 4000 |
| no_semantic_tokens_greedy | [0.582, 0.612] | [0.327, 0.356] | [0.121, 0.142] | 4000 |
| no_roi_greedy | [0.716, 0.743] | [0.257, 0.284] | [0.061, 0.076] | 4000 |
| oracle_best_feasible_evidence | [0.713, 0.740] | [0.260, 0.287] | [0.000, 0.000] | 4000 |
