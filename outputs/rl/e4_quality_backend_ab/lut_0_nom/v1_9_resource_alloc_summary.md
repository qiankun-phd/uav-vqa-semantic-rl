# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.748 | 0.486 | 0.532 | 0.552 | 0.312 | 0.571 | 0.716 | 0.650-0.820 | 0.068 | 0.571 | 25.703 | 2732.891 | 0.000 | 4.279 | 501.976 | 462.747 | 3.587 | 0.013 | 20.792 | 0.000 | 0.252 | 0.295 | 0.000 | 0.000 | 0.000 | 0.000 | -1.857 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.665 | 0.220 | 0.115 | 0.000 | 0.411 | 0.220 | 0.115 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.486 | 0.486 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.191 |
