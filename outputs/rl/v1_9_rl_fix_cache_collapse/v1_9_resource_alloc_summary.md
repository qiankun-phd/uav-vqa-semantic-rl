# V1.9 LUT Resource Allocation Summary

- rollout rows: 384
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy LCB | accuracy mean | uncertainty | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM violation | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.104 | 0.417 | 0.437 | 0.078 | 0.121 | 1.610 | 0.000 | 1.000 | 0.896 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.601 |
| always_light | 0.042 | 0.572 | 0.610 | 0.083 | 5.877 | 809.751 | 1.059 | 0.994 | 0.708 | 0.792 | 0.000 | 0.000 | 0.000 | 0.000 | -2.845 |
| always_image | 0.000 | 0.429 | 0.442 | 0.060 | 7.885 | 868.899 | 182.724 | 0.000 | 0.688 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | -3.977 |
| greedy_min_sufficient_evidence | 0.188 | 0.770 | 0.802 | 0.077 | 7.032 | 879.639 | 74.465 | 0.592 | 0.312 | 0.812 | 0.000 | 0.000 | 0.000 | 0.000 | -2.444 |
| no_cache_greedy | 0.042 | 0.415 | 0.431 | 0.063 | 7.858 | 934.927 | 129.426 | 0.292 | 0.688 | 0.958 | 0.000 | 0.000 | 0.000 | 0.000 | -3.694 |
| no_semantic_tokens_greedy | 0.125 | 0.796 | 0.828 | 0.080 | 7.347 | 813.471 | 160.513 | 0.122 | 0.312 | 0.875 | 0.000 | 0.000 | 0.000 | 0.000 | -2.950 |
| oracle_best_feasible_evidence | 0.146 | 0.797 | 0.826 | 0.073 | 7.747 | 919.718 | 128.797 | 0.295 | 0.312 | 0.812 | 0.000 | 0.000 | 0.000 | 0.000 | -2.867 |
| ppo | 0.188 | 0.719 | 0.745 | 0.072 | 3.735 | 493.786 | 19.118 | 0.895 | 0.312 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | -1.207 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.104 | 0.479 | 0.417 | 0.000 |
| no_cache_greedy | 0.000 | 0.292 | 0.708 | 0.000 |
| no_semantic_tokens_greedy | 0.104 | 0.000 | 0.896 | 0.000 |
| oracle_best_feasible_evidence | 0.125 | 0.167 | 0.708 | 0.000 |
| ppo | 0.417 | 0.479 | 0.104 | 0.000 |
