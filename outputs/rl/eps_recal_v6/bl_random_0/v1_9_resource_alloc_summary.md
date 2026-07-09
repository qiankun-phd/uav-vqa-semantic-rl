# V1.9 LUT Resource Allocation Summary

- rollout rows: 250
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.536 | 0.144 | 0.445 | 0.466 | 0.138 | 0.628 | 0.620 | 0.464-0.696 | 0.268 | 0.832 | 0.008 | 2011.734 | 1.204 | 9.372 | 1023.297 | 966.404 | 7.987 | 0.105 | 75.585 | 0.000 | 0.464 | 0.320 | 0.000 | 0.000 | 0.392 | 0.392 | -2.203 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.348 | 0.356 | 0.296 | 0.000 | 0.348 | 0.356 | 0.296 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.144 | 0.144 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.960 |
