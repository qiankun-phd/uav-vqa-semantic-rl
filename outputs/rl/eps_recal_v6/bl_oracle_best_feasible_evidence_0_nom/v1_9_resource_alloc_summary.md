# V1.9 LUT Resource Allocation Summary

- rollout rows: 973
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.952 | 0.691 | 0.485 | 0.495 | 0.269 | 0.414 | 0.529 | 0.529-0.529 | 0.015 | 0.093 | 0.000 | 4091.083 | 0.000 | 7.984 | 859.563 | 759.757 | 6.436 | 0.064 | 89.326 | 0.000 | 0.048 | 0.281 | 0.000 | 0.000 | 0.000 | 0.000 | -0.043 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.294 | 0.377 | 0.329 | 0.000 | 0.037 | 0.377 | 0.329 | 0.000 | 0.000 | 0.000 | 0.105 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.691 | 0.691 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.410 |
