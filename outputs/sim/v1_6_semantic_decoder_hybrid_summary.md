# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.277 | 0.566 | 0.689 | 0.200 | 0.000 | 1.000 | 0.723 | 0.000 |
| always_light | 0.625 | 0.781 | 2.069 | 1.000 | 0.874 | 0.991 | 0.376 | 0.000 |
| always_image | 0.108 | 0.542 | 3.909 | 2.500 | 95.153 | 0.000 | 0.723 | 0.339 |
| greedy_min_sufficient_evidence | 0.794 | 0.816 | 2.076 | 1.087 | 19.412 | 0.796 | 0.206 | 0.000 |
| no_cache_greedy | 0.625 | 0.698 | 2.759 | 1.563 | 35.516 | 0.627 | 0.376 | 0.170 |
| no_semantic_tokens_greedy | 0.365 | 0.704 | 3.006 | 1.847 | 66.854 | 0.297 | 0.466 | 0.169 |
| oracle_best_feasible_evidence | 0.794 | 0.834 | 1.634 | 0.742 | 0.567 | 0.994 | 0.206 | 0.000 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.277 | 0.517 | 0.206 | 0.000 |
| no_cache_greedy | 0.000 | 0.625 | 0.376 | 0.000 |
| no_semantic_tokens_greedy | 0.284 | 0.000 | 0.716 | 0.000 |
| oracle_best_feasible_evidence | 0.323 | 0.677 | 0.000 | 0.000 |

## Approximate 95% CI for Rate Metrics

| policy | success CI | quality violation CI | deadline violation CI | tasks |
|---|---:|---:|---:|---:|
| always_cache | [0.267, 0.286] | [0.714, 0.733] | [0.000, 0.000] | 8000 |
| always_light | [0.614, 0.635] | [0.365, 0.386] | [0.000, 0.000] | 8000 |
| always_image | [0.101, 0.114] | [0.714, 0.733] | [0.328, 0.349] | 8000 |
| greedy_min_sufficient_evidence | [0.785, 0.803] | [0.197, 0.215] | [0.000, 0.000] | 8000 |
| no_cache_greedy | [0.614, 0.635] | [0.365, 0.386] | [0.161, 0.178] | 8000 |
| no_semantic_tokens_greedy | [0.355, 0.376] | [0.455, 0.477] | [0.161, 0.177] | 8000 |
| oracle_best_feasible_evidence | [0.785, 0.803] | [0.197, 0.215] | [0.000, 0.000] | 8000 |
