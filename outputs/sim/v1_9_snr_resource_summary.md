# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.286 | 0.560 | 0.050 | 0.200 | 0.000 | 1.000 | 0.714 | 0.000 |
| always_light | 0.203 | 0.599 | 0.355 | 1.000 | 0.850 | 0.991 | 0.796 | 0.000 |
| always_image | 0.560 | 0.588 | 1.546 | 2.500 | 93.044 | 0.000 | 0.440 | 0.000 |
| greedy_min_sufficient_evidence | 0.711 | 0.707 | 0.896 | 1.562 | 48.024 | 0.484 | 0.289 | 0.000 |
| no_cache_greedy | 0.567 | 0.583 | 1.305 | 2.195 | 73.879 | 0.206 | 0.433 | 0.000 |
| no_semantic_tokens_greedy | 0.704 | 0.712 | 1.119 | 1.843 | 65.779 | 0.293 | 0.296 | 0.000 |
| oracle_best_feasible_evidence | 0.711 | 0.721 | 0.664 | 1.219 | 29.122 | 0.687 | 0.289 | 0.000 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.286 | 0.188 | 0.527 | 0.000 |
| no_cache_greedy | 0.000 | 0.203 | 0.796 | 0.000 |
| no_semantic_tokens_greedy | 0.286 | 0.000 | 0.714 | 0.000 |
| oracle_best_feasible_evidence | 0.367 | 0.291 | 0.342 | 0.000 |

## Approximate 95% CI for Rate Metrics

| policy | success CI | quality violation CI | deadline violation CI | tasks |
|---|---:|---:|---:|---:|
| always_cache | [0.276, 0.296] | [0.704, 0.724] | [0.000, 0.000] | 8000 |
| always_light | [0.195, 0.212] | [0.788, 0.805] | [0.000, 0.000] | 8000 |
| always_image | [0.549, 0.571] | [0.429, 0.451] | [0.000, 0.000] | 8000 |
| greedy_min_sufficient_evidence | [0.701, 0.721] | [0.279, 0.299] | [0.000, 0.000] | 8000 |
| no_cache_greedy | [0.556, 0.578] | [0.422, 0.444] | [0.000, 0.000] | 8000 |
| no_semantic_tokens_greedy | [0.694, 0.714] | [0.286, 0.306] | [0.000, 0.000] | 8000 |
| oracle_best_feasible_evidence | [0.701, 0.721] | [0.279, 0.299] | [0.000, 0.000] | 8000 |
