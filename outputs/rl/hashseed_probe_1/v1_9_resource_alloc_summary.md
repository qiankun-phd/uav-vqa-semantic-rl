# V1.9 LUT Resource Allocation Summary

- rollout rows: 10
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.000 | 0.000 | 0.506 | 0.646 | 0.383 | 0.820 | 0.820 | 0.820-0.820 | 0.314 | 0.744 | 19.442 | 1697.984 | 1.400 | 7.648 | 820.032 | 748.670 | 6.035 | 0.100 | 81.259 | 0.000 | 1.000 | 0.700 | 0.000 | 0.000 | 0.400 | 0.400 | -4.257 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.500 | 0.100 | 0.400 | 0.000 | 0.500 | 0.100 | 0.400 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.500 |
