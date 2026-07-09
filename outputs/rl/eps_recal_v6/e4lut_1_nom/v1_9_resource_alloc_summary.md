# V1.9 LUT Resource Allocation Summary

- rollout rows: 987
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.512 | 0.511 | 0.215 | 0.225 | 0.309 | 0.396 | 0.492 | 0.464-0.696 | 0.222 | 1.891 | 0.000 | 2363.991 | 0.000 | 2.519 | 314.193 | 303.072 | 2.236 | 0.012 | 0.228 | 0.000 | 0.488 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.761 | 0.239 | 0.000 | 0.000 | 0.508 | 0.239 | 0.000 | 0.000 | 0.000 | 0.000 | 0.067 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.511 | 0.511 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.510 |
