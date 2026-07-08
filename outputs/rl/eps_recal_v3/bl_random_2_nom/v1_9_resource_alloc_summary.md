# V1.9 LUT Resource Allocation Summary

- rollout rows: 982
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.686 | 0.356 | 0.406 | 0.428 | 0.315 | 0.282 | 0.504 | 0.504-0.504 | 0.064 | 0.579 | 27.123 | 2453.893 | 0.000 | 5.266 | 578.864 | 536.682 | 4.391 | 0.040 | 39.406 | 0.000 | 0.314 | 0.484 | 0.000 | 0.000 | 0.000 | 0.000 | -2.176 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.515 | 0.269 | 0.216 | 0.000 | 0.261 | 0.269 | 0.216 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.356 | 0.356 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.127 |
