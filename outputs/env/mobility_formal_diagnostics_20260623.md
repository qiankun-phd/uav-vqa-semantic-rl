# Mobility Formal Diagnostics 2026-06-23

Scope: environment-side diagnosis of the completed 300-episode two-timescale mobility benchmark. No RL code, LUT, or algorithm output was modified.

Source artifacts:

- outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/scenario_comparison_summary.csv
- outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/*/seed*/proposed_two_timescale_ppo/v1_9_resource_alloc_rollout.csv
- outputs/rl/two_timescale_mobility_formal_20260623_proposed_k3_300/*/seed*/baselines/v1_9_resource_alloc_rollout.csv

Incomplete placeholder directories from failed variant-name launches were ignored; corrected retry directories are algorithm artifacts and were not modified by this diagnostic pass.

## Proposed Policy Summary

| scenario | semantic success | deadline violation | avg delay s | mobility energy J | arrival delay s | payload KB | UTM conflict | UTM risk | service mix s0/s1/s2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | 0.386 | 0.229 | 1.802 | 78.000 | 0.000 | 61.638 | 0.000 | 0.067 | 0.093/0.547/0.359 |
| low_snr_blockage | 0.955 | 0.939 | 509.965 | 78.000 | 0.000 | 34.288 | 0.000 | 0.141 | 0.053/0.030/0.917 |
| utm_conflict | 0.007 | 0.030 | 1.145 | 99.749 | 0.272 | 6.092 | 0.094 | 0.007 | 0.099/0.872/0.029 |

## Proposed Rollout By Service Level

### nominal_patrol

| service | ratio | deadline vio | delay s | arrival s | mobility J | LCB | epsilon | gap | payload KB | sensed SNR dB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.093 | 0.000 | 0.315 | 0.000 | 78.000 | 0.718 | 0.770 | 0.060 | 0.000 | -35.264 |
| 1 | 0.547 | 0.000 | 0.859 | 0.000 | 78.000 | 0.463 | 0.789 | 0.328 | 1.078 | 25.490 |
| 2 | 0.359 | 0.638 | 3.625 | 0.000 | 78.000 | 0.825 | 0.790 | 0.000 | 169.841 | 30.980 |

### low_snr_blockage

| service | ratio | deadline vio | delay s | arrival s | mobility J | LCB | epsilon | gap | payload KB | sensed SNR dB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.053 | 0.000 | 0.268 | 0.000 | 78.000 | 0.887 | 0.800 | 0.000 | 0.000 | -90.756 |
| 1 | 0.030 | 0.759 | 47.943 | 0.000 | 78.000 | 0.709 | 0.820 | 0.111 | 0.710 | -29.047 |
| 2 | 0.917 | 1.000 | 554.741 | 0.000 | 78.000 | 0.920 | 0.803 | 0.001 | 37.382 | -29.591 |

### utm_conflict

| service | ratio | deadline vio | delay s | arrival s | mobility J | LCB | epsilon | gap | payload KB | sensed SNR dB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.099 | 0.000 | 0.422 | 0.133 | 88.637 | 0.591 | 0.820 | 0.229 | 0.000 | -35.803 |
| 1 | 0.872 | 0.001 | 1.109 | 0.284 | 100.668 | 0.461 | 0.820 | 0.359 | 1.185 | 23.946 |
| 2 | 0.029 | 1.000 | 4.649 | 0.398 | 109.804 | 0.726 | 0.820 | 0.102 | 172.453 | 23.997 |

## Baseline Cross-Check

| scenario | policy | deadline vio | delay s | arrival s | mobility J | LCB | epsilon | gap | payload KB | UTM conflict | UTM risk | mobility modes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nominal_patrol | always_cache | 0.000 | 0.185 | 0.000 | 78.000 | 0.340 | 0.789 | 0.450 | 0.000 | 0.000 | 0.000 | stay:1800 |
| nominal_patrol | always_semantic_token | 0.358 | 3.201 | 2.433 | 367.883 | 0.495 | 0.797 | 0.304 | 1.142 | 0.000 | 0.100 | serve_task:1800 |
| nominal_patrol | always_image | 0.897 | 5.106 | 1.634 | 247.118 | 0.350 | 0.803 | 0.454 | 198.091 | 0.000 | 0.128 | serve_task:1800 |
| nominal_patrol | semantic_greedy | 0.866 | 5.052 | 1.780 | 269.325 | 0.338 | 0.804 | 0.467 | 180.500 | 0.000 | 0.106 | serve_task:1797, stay:3 |
| low_snr_blockage | always_cache | 0.000 | 0.173 | 0.000 | 78.000 | 0.343 | 0.803 | 0.465 | 0.000 | 0.000 | 0.000 | stay:1800 |
| low_snr_blockage | always_semantic_token | 0.955 | 13.674 | 12.394 | 1873.971 | 0.634 | 0.808 | 0.210 | 1.047 | 0.000 | 0.144 | serve_task:1800 |
| low_snr_blockage | always_image | 0.988 | 35.056 | 11.964 | 1808.993 | 0.623 | 0.808 | 0.265 | 53.872 | 0.000 | 0.144 | serve_task:1800 |
| low_snr_blockage | semantic_greedy | 0.913 | 15.411 | 12.202 | 1849.872 | 0.856 | 0.803 | 0.005 | 2.557 | 0.000 | 0.139 | serve_task:1687, stay:113 |
| utm_conflict | always_cache | 0.000 | 0.178 | 0.000 | 78.000 | 0.313 | 0.820 | 0.507 | 0.000 | 0.000 | 0.000 | stay:1500 |
| utm_conflict | always_semantic_token | 0.883 | 6.424 | 5.549 | 838.952 | 0.551 | 0.820 | 0.269 | 1.188 | 0.176 | 0.017 | serve_task:1500 |
| utm_conflict | always_image | 1.000 | 9.137 | 5.472 | 827.393 | 0.365 | 0.820 | 0.455 | 198.513 | 0.176 | 0.017 | serve_task:1500 |
| utm_conflict | semantic_greedy | 1.000 | 9.137 | 5.472 | 827.393 | 0.365 | 0.820 | 0.455 | 198.513 | 0.176 | 0.017 | serve_task:1500 |

## UTM Mobility-Mode Diagnostic

| mobility mode | ratio | avg risk | conflict rate | deadline vio | arrival s | mobility J | fly distance m |
|---|---:|---:|---:|---:|---:|---:|---:|
| avoid_conflict | 0.249 | 0.000 | 0.161 | 0.046 | 1.094 | 165.461 | 19.698 |
| stay | 0.751 | 0.010 | 0.072 | 0.025 | 0.000 | 78.000 | 0.000 |

Interpretation: the formal PPO used avoid_conflict in about 24.9% of utm_conflict attempts and stay in about 75.1%. Compared with serve-task baselines, the proposed policy reduces UTM conflict from 0.176 to 0.094 and reduces arrival delay from about 5.5s to 0.27s. However, within the proposed rollout, avoid_conflict rows still have higher realized conflict rate than stay rows because the action is triggered on harder/high-conflict states. Therefore this benchmark supports the aggregate claim that the mobility-aware policy reduces UTM pressure, but it is not clean evidence that avoid_conflict alone is sufficient; a paired counterfactual rollout would be needed for that narrower claim.

## Diagnosis

1. nominal_patrol: deadline pressure is not abnormally mobility-driven for the proposed policy. Proposed has zero arrival delay, 78 J hover/mobility energy, and deadline violation 0.229. The larger violations in always-image and semantic-greedy are mostly service/payload choices plus serve-task movement. No immediate uav_speed_mps, flight_energy_j_per_m, or area_spacing_m change is needed for the formal proposed run.

2. low_snr_blockage: the high proposed deadline violation is link/payload dominated, not flight dominated. Proposed has arrival delay 0.000s and mobility energy 78 J, yet image rows have delay about 554.7s at sensed SNR around -29.6 dB. Token rows have lower payload but still insufficient conservative semantic quality (LCB=0.709, epsilon=0.820, gap=0.111 in the proposed rollout; always-token LCB is 0.634 with gap=0.210). This explains the high image ratio: the policy is trying to satisfy semantic QoS under low SNR, then image payload makes the deadline infeasible.

3. utm_conflict: proposed deadline behavior is healthy. Deadline violation is only 0.030, arrival delay is 0.272s, and UTM conflict is reduced versus serve-task baselines. The remaining near-zero semantic success is mostly a conservative semantic-QoS issue: scenario epsilon is fixed at 0.82, while proposed LCB averages 0.482 and semantic_quality_gap averages 0.338. This is not evidence of excessive UAV flight energy or an over-tight deadline.

## Tuning Recommendations

- Do not globally tune uav_speed_mps or flight_energy_j_per_m based on this run. The main proposed-policy failures in low_snr_blockage occur with zero arrival delay, so faster UAV motion would not fix the bottleneck.
- Do not reduce the UTM buffer just to improve utm_conflict: proposed already lowers conflict to 0.094 and deadline violation to 0.030. If the paper needs a stronger UTM trade-off, use a paired mobility ablation/counterfactual rather than weakening the UTM model.
- Consider a scenario-local tau_scale increase for low_snr_blockage only if the scenario is meant to be partially feasible for image evidence. Otherwise keep it as a stress test that exposes the semantic QoS versus payload/deadline trade-off.
- Consider reducing area_spacing_m in low_snr_blockage only if baseline comparisons should not be dominated by serve-task travel. For the proposed policy, travel is already avoided, so this is a baseline-presentation choice rather than a model bug.
- Keep hover_power_w/base mobility energy documented: cache/stay actions still incur about 78 J per attempt because the UAV remains active and hovering. That is physically plausible, but should be described in the benchmark protocol so cache is not interpreted as zero UAV-system energy.

## Verdict

The formal mobility benchmark constraints are mostly coherent. The only clearly sharp edge is low_snr_blockage, where conservative semantic quality pushes the policy toward image evidence while the low-SNR link makes image upload nearly impossible within deadline. That is a useful stress scenario, but paper claims should present it as a semantic-QoS/payload trade-off rather than as a mobility limitation.
