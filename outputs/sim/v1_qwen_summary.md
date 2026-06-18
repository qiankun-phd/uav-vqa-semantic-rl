# V1 Qwen Resource Simulation Summary

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.280 | 0.568 | 0.689 | 0.200 | 0.000 | 1.000 | 0.720 | 0.000 |
| always_light | 0.519 | 0.664 | 2.073 | 1.000 | 0.644 | 0.993 | 0.480 | 0.000 |
| always_image | 0.174 | 0.554 | 3.902 | 2.500 | 87.352 | 0.000 | 0.655 | 0.343 |
| greedy_min_sufficient_evidence | 0.721 | 0.781 | 2.284 | 1.253 | 24.307 | 0.722 | 0.279 | 0.026 |
| oracle_best_feasible_evidence | 0.723 | 0.809 | 1.972 | 1.039 | 18.874 | 0.784 | 0.277 | 0.000 |

## Service Level Selection Ratio

| policy | cache s=0 | light s=1 | image s=2 |
|---|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.271 | 0.417 | 0.313 |
| oracle_best_feasible_evidence | 0.355 | 0.429 | 0.215 |
