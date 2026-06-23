# Two-timescale Mobility-aware Semantic PPO Formal Benchmark

Generated from five parallel 300-train-episode scenario benchmark runs on 2026-06-23 Asia/Shanghai.

## Run Directories

- `proposed_two_timescale_ppo`: `outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300`
- `monolithic_ppo`: `outputs/rl/two_timescale_mobility_formal_20260623_monolithic_300`
- `no_mobility_actor`: `outputs/rl/two_timescale_mobility_formal_20260623_no_mobility_300`
- `no_lyapunov_queues`: `outputs/rl/two_timescale_mobility_formal_20260623_no_queue_300_retry1`
- `no_projection`: `outputs/rl/two_timescale_mobility_formal_20260623_no_projection_300_retry1`

The requested labels `no_lyapunov_queues` and `no_projection` were mapped to runner variants `ppo_without_queues` and `ppo_without_projection`; corrected outputs use `_retry1` directories to avoid overwriting the failed partial launch directories.

## Metrics

Each value is the mean across seeds `0,1,2`, with `50` evaluation episodes per seed and `12` tasks per episode. PPO variants used `300` training episodes per scenario/seed.

## disaster_hotspot

| policy | sem succ | task succ | acc LCB | quality gap | delay | energy | mob energy | arrival | coverage | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.015 | 0.015 | 0.265 | 0.576 | 0.224 | 85.850 | 78.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| always_semantic_token | 0.105 | 0.039 | 0.471 | 0.376 | 1.596 | 157.645 | 122.748 | 0.812 | 0.149 | 0.298 | 0.000 | 0.000 | 1.000 | 0.000 |
| always_image | 0.220 | 0.000 | 0.469 | 0.395 | 4.330 | 328.079 | 119.008 | 0.787 | 0.148 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| semantic_greedy | 0.107 | 0.041 | 0.429 | 0.416 | 4.024 | 310.712 | 120.670 | 0.789 | 0.148 | 0.959 | 0.000 | 0.017 | 0.090 | 0.893 |
| proposed_two_timescale_ppo | 0.317 | 0.064 | 0.659 | 0.194 | 4.252 | 569.053 | 538.538 | 3.562 | 0.561 | 0.733 | 0.000 | 0.162 | 0.838 | 0.000 |
| monolithic_ppo | 0.228 | 0.130 | 0.577 | 0.274 | 1.576 | 165.534 | 132.132 | 0.828 | 0.148 | 0.291 | 0.000 | 0.089 | 0.911 | 0.000 |
| no_mobility_actor | 0.194 | 0.094 | 0.542 | 0.305 | 1.541 | 156.685 | 125.454 | 0.830 | 0.149 | 0.292 | 0.000 | 0.129 | 0.871 | 0.000 |
| no_lyapunov_queues | 0.228 | 0.130 | 0.577 | 0.274 | 1.578 | 165.791 | 132.132 | 0.828 | 0.148 | 0.291 | 0.000 | 0.089 | 0.911 | 0.000 |
| no_projection | 0.220 | 0.000 | 0.590 | 0.264 | 19.931 | 1594.434 | 119.008 | 0.787 | 0.148 | 1.000 | 0.000 | 0.000 | 0.900 | 0.100 |

## edge_overload

| policy | sem succ | task succ | acc LCB | quality gap | delay | energy | mob energy | arrival | coverage | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.000 | 0.000 | 0.172 | 0.468 | 0.355 | 85.850 | 78.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| always_semantic_token | 0.607 | 0.344 | 0.651 | 0.065 | 5.070 | 825.477 | 759.275 | 3.766 | 0.098 | 0.431 | 0.000 | 0.000 | 1.000 | 0.000 |
| always_image | 0.000 | 0.000 | 0.287 | 0.353 | 5.269 | 472.769 | 263.707 | 1.308 | 0.030 | 0.330 | 0.000 | 0.000 | 0.000 | 1.000 |
| semantic_greedy | 0.000 | 0.000 | 0.287 | 0.353 | 5.269 | 472.769 | 263.707 | 1.308 | 0.030 | 0.330 | 0.000 | 0.000 | 0.000 | 1.000 |
| proposed_two_timescale_ppo | 0.697 | 0.681 | 0.674 | 0.050 | 1.742 | 231.837 | 166.961 | 0.476 | 0.010 | 0.046 | 0.000 | 0.058 | 0.942 | 0.000 |
| monolithic_ppo | 0.606 | 0.386 | 0.644 | 0.067 | 4.840 | 795.128 | 738.649 | 3.648 | 0.099 | 0.381 | 0.000 | 0.041 | 0.959 | 0.000 |
| no_mobility_actor | 0.708 | 0.674 | 0.677 | 0.047 | 2.203 | 306.261 | 242.432 | 0.900 | 0.032 | 0.056 | 0.000 | 0.058 | 0.942 | 0.000 |
| no_lyapunov_queues | 0.608 | 0.393 | 0.645 | 0.066 | 4.835 | 794.843 | 739.633 | 3.653 | 0.099 | 0.373 | 0.000 | 0.041 | 0.959 | 0.000 |
| no_projection | 0.381 | 0.235 | 0.501 | 0.186 | 4.220 | 651.669 | 570.732 | 2.722 | 0.070 | 0.334 | 0.000 | 0.282 | 0.718 | 0.000 |

## low_snr_blockage

| policy | sem succ | task succ | acc LCB | quality gap | delay | energy | mob energy | arrival | coverage | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.053 | 0.053 | 0.343 | 0.465 | 0.173 | 85.850 | 78.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| always_semantic_token | 0.661 | 0.018 | 0.634 | 0.210 | 13.674 | 1909.435 | 1873.971 | 12.394 | 0.015 | 0.955 | 0.000 | 0.000 | 1.000 | 0.000 |
| always_image | 0.673 | 0.002 | 0.623 | 0.265 | 35.056 | 2037.674 | 1808.993 | 11.964 | 0.012 | 0.988 | 0.000 | 0.000 | 0.000 | 1.000 |
| semantic_greedy | 0.950 | 0.087 | 0.856 | 0.005 | 15.411 | 1896.658 | 1849.872 | 12.202 | 0.026 | 0.913 | 0.000 | 0.063 | 0.887 | 0.050 |
| proposed_two_timescale_ppo | 0.955 | 0.053 | 0.912 | 0.004 | 509.965 | 779.164 | 78.000 | 0.000 | 0.000 | 0.939 | 0.000 | 0.053 | 0.030 | 0.917 |
| monolithic_ppo | 0.792 | 0.087 | 0.757 | 0.096 | 10.451 | 1468.604 | 1438.910 | 9.392 | 0.020 | 0.734 | 0.000 | 0.241 | 0.759 | 0.000 |
| no_mobility_actor | 0.956 | 0.026 | 0.834 | 0.023 | 17.285 | 2424.149 | 2399.280 | 15.868 | 0.042 | 0.974 | 0.000 | 0.371 | 0.629 | 0.000 |
| no_lyapunov_queues | 0.792 | 0.087 | 0.757 | 0.095 | 10.458 | 1466.017 | 1437.718 | 9.385 | 0.020 | 0.736 | 0.000 | 0.241 | 0.759 | 0.000 |
| no_projection | 0.482 | 0.033 | 0.541 | 0.292 | 18.497 | 2040.147 | 1478.993 | 9.607 | 0.021 | 0.659 | 0.000 | 0.338 | 0.634 | 0.028 |

## nominal_patrol

| policy | sem succ | task succ | acc LCB | quality gap | delay | energy | mob energy | arrival | coverage | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.000 | 0.000 | 0.340 | 0.450 | 0.185 | 85.850 | 78.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| always_semantic_token | 0.106 | 0.061 | 0.495 | 0.304 | 3.201 | 407.329 | 367.883 | 2.433 | 0.048 | 0.358 | 0.000 | 0.000 | 1.000 | 0.000 |
| always_image | 0.111 | 0.013 | 0.350 | 0.454 | 5.106 | 456.307 | 247.118 | 1.634 | 0.038 | 0.897 | 0.000 | 0.000 | 0.000 | 1.000 |
| semantic_greedy | 0.087 | 0.036 | 0.338 | 0.467 | 5.052 | 465.274 | 269.325 | 1.780 | 0.041 | 0.866 | 0.000 | 0.002 | 0.079 | 0.919 |
| proposed_two_timescale_ppo | 0.386 | 0.157 | 0.617 | 0.185 | 1.802 | 192.303 | 78.000 | 0.000 | 0.000 | 0.229 | 0.000 | 0.093 | 0.547 | 0.359 |
| monolithic_ppo | 0.283 | 0.098 | 0.617 | 0.182 | 4.251 | 552.299 | 495.438 | 3.228 | 0.057 | 0.450 | 0.000 | 0.094 | 0.794 | 0.112 |
| no_mobility_actor | 0.211 | 0.083 | 0.566 | 0.234 | 3.813 | 485.377 | 431.930 | 2.857 | 0.055 | 0.409 | 0.000 | 0.103 | 0.823 | 0.073 |
| no_lyapunov_queues | 0.283 | 0.098 | 0.618 | 0.182 | 4.343 | 559.145 | 494.949 | 3.225 | 0.057 | 0.450 | 0.000 | 0.094 | 0.793 | 0.113 |
| no_projection | 0.219 | 0.094 | 0.521 | 0.274 | 29.997 | 2514.530 | 449.494 | 2.867 | 0.054 | 0.387 | 0.000 | 0.206 | 0.518 | 0.277 |

## utm_conflict

| policy | sem succ | task succ | acc LCB | quality gap | delay | energy | mob energy | arrival | coverage | deadline vio | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.000 | 0.000 | 0.313 | 0.507 | 0.178 | 85.850 | 78.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| always_semantic_token | 0.000 | 0.000 | 0.551 | 0.269 | 6.424 | 873.849 | 838.952 | 5.549 | 0.078 | 0.883 | 0.176 | 0.000 | 1.000 | 0.000 |
| always_image | 0.000 | 0.000 | 0.365 | 0.455 | 9.137 | 1036.495 | 827.393 | 5.472 | 0.076 | 1.000 | 0.176 | 0.000 | 0.000 | 1.000 |
| semantic_greedy | 0.000 | 0.000 | 0.365 | 0.455 | 9.137 | 1036.495 | 827.393 | 5.472 | 0.076 | 1.000 | 0.176 | 0.000 | 0.000 | 1.000 |
| proposed_two_timescale_ppo | 0.007 | 0.000 | 0.482 | 0.338 | 1.145 | 138.846 | 99.749 | 0.272 | -0.005 | 0.030 | 0.094 | 0.099 | 0.872 | 0.029 |
| monolithic_ppo | 0.000 | 0.000 | 0.627 | 0.193 | 6.169 | 852.209 | 820.017 | 5.372 | 0.081 | 0.843 | 0.151 | 0.100 | 0.900 | 0.000 |
| no_mobility_actor | 0.000 | 0.000 | 0.567 | 0.253 | 6.330 | 870.603 | 838.952 | 5.549 | 0.078 | 0.883 | 0.147 | 0.120 | 0.880 | 0.000 |
| no_lyapunov_queues | 0.000 | 0.000 | 0.627 | 0.193 | 6.169 | 852.209 | 820.017 | 5.372 | 0.081 | 0.843 | 0.151 | 0.100 | 0.900 | 0.000 |
| no_projection | 0.000 | 0.000 | 0.394 | 0.426 | 26.453 | 2368.955 | 641.993 | 3.977 | 0.058 | 0.479 | 0.075 | 0.521 | 0.204 | 0.275 |

## Quick Observations

- `disaster_hotspot`: proposed semantic success 0.317 vs monolithic 0.228, no-mobility 0.194, semantic_greedy 0.107; task success 0.064, deadline violation 0.733, UTM conflict 0.000.
- `edge_overload`: proposed semantic success 0.697 vs monolithic 0.606, no-mobility 0.708, semantic_greedy 0.000; task success 0.681, deadline violation 0.046, UTM conflict 0.000.
- `low_snr_blockage`: proposed semantic success 0.955 vs monolithic 0.792, no-mobility 0.956, semantic_greedy 0.950; task success 0.053, deadline violation 0.939, UTM conflict 0.000.
- `nominal_patrol`: proposed semantic success 0.386 vs monolithic 0.283, no-mobility 0.211, semantic_greedy 0.087; task success 0.157, deadline violation 0.229, UTM conflict 0.000.
- `utm_conflict`: proposed semantic success 0.007 vs monolithic 0.000, no-mobility 0.000, semantic_greedy 0.000; task success 0.000, deadline violation 0.030, UTM conflict 0.094.

## Log Health

- The corrected sessions completed and generated scenario summaries.
- `logs/hppo_tt_no_queue_300.log` and `logs/hppo_tt_no_projection_300.log` contain the initial invalid-variant tracebacks; corrected runs are in `*_retry1.log` and completed.
- No CUDA OOM or NaN failure was observed in the corrected completion logs.
