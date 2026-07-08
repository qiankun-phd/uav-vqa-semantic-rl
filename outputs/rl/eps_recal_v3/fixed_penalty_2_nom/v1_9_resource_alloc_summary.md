# V1.9 LUT Resource Allocation Summary

- rollout rows: 982
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.632 | 0.533 | 0.441 | 0.456 | 0.307 | 0.282 | 0.504 | 0.504-0.504 | 0.021 | 0.114 | 0.371 | 0.000 | 0.000 | 0.472 | 84.395 | 58.992 | 0.009 | 0.000 | 17.840 | 0.000 | 0.368 | 0.100 | 0.000 | 0.000 | 0.000 | 0.000 | -0.904 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.899 | 0.004 | 0.097 | 0.000 | 0.645 | 0.004 | 0.097 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.533 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.433 |
