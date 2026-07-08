# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.499 | 0.485 | 0.477 | 0.500 | 0.294 | 0.572 | 0.747 | 0.650-0.820 | 0.105 | 0.802 | 14.545 | 1169.587 | 0.000 | 2.747 | 282.892 | 192.240 | 1.242 | -0.001 | 67.292 | 0.000 | 0.501 | 0.383 | 0.000 | 0.000 | 0.000 | 0.000 | -1.541 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.608 | 0.018 | 0.373 | 0.000 | 0.354 | 0.018 | 0.373 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.485 | 0.485 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.103 |
