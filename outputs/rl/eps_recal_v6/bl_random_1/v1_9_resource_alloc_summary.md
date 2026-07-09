# V1.9 LUT Resource Allocation Summary

- rollout rows: 250
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.568 | 0.104 | 0.468 | 0.487 | 0.123 | 0.629 | 0.610 | 0.464-0.696 | 0.252 | 0.769 | 0.006 | 2168.351 | 1.500 | 9.903 | 1072.996 | 1010.831 | 8.389 | 0.108 | 85.776 | 0.000 | 0.432 | 0.364 | 0.000 | 0.000 | 0.456 | 0.456 | -2.569 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.312 | 0.356 | 0.332 | 0.000 | 0.312 | 0.356 | 0.332 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.104 | 0.104 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.172 |
