# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.540 | 0.540 | 0.372 | 0.386 | 0.308 | 0.279 | 0.504 | 0.504-0.504 | 0.044 | 0.335 | 0.000 | 0.000 | 0.000 | 0.126 | 64.016 | 58.163 | 0.000 | 0.000 | 0.000 | 0.000 | 0.460 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.796 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 | 0.746 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.540 | 0.540 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.540 |
