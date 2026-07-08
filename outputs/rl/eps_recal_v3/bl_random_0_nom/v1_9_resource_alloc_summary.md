# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.697 | 0.350 | 0.411 | 0.432 | 0.314 | 0.279 | 0.504 | 0.504-0.504 | 0.064 | 0.600 | 29.155 | 2643.841 | 0.000 | 5.596 | 612.140 | 566.550 | 4.654 | 0.039 | 44.198 | 0.000 | 0.303 | 0.497 | 0.000 | 0.000 | 0.000 | 0.000 | -2.270 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.502 | 0.255 | 0.243 | 0.000 | 0.247 | 0.255 | 0.243 | 0.000 | 0.000 | 0.000 | 0.002 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.350 | 0.350 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.148 |
