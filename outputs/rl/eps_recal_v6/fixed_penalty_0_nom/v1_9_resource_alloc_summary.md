# V1.9 LUT Resource Allocation Summary

- rollout rows: 981
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.742 | 0.651 | 0.388 | 0.394 | 0.280 | 0.415 | 0.557 | 0.464-0.696 | 0.084 | 0.643 | 0.000 | 6661.391 | 0.000 | 9.407 | 1074.499 | 1034.449 | 8.616 | 0.045 | 36.132 | 0.000 | 0.258 | 0.088 | 0.008 | 0.000 | 0.000 | 0.000 | 0.021 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.526 | 0.343 | 0.131 | 0.000 | 0.271 | 0.343 | 0.131 | 0.000 | 0.000 | 0.000 | 0.137 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.651 | 0.651 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.564 |
