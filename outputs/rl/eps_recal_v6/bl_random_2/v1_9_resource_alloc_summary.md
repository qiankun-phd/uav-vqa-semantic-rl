# V1.9 LUT Resource Allocation Summary

- rollout rows: 250
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.552 | 0.112 | 0.449 | 0.467 | 0.133 | 0.631 | 0.611 | 0.464-0.696 | 0.263 | 0.775 | 0.008 | 2036.139 | 1.320 | 9.278 | 1009.093 | 950.951 | 7.850 | 0.102 | 78.111 | 0.000 | 0.448 | 0.316 | 0.000 | 0.000 | 0.408 | 0.408 | -2.345 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.356 | 0.336 | 0.308 | 0.000 | 0.356 | 0.336 | 0.308 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.112 | 0.112 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.020 |
