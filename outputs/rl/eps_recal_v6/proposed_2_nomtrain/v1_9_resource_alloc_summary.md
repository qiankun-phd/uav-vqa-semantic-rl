# V1.9 LUT Resource Allocation Summary

- rollout rows: 979
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.768 | 0.672 | 0.397 | 0.403 | 0.280 | 0.417 | 0.549 | 0.464-0.696 | 0.078 | 0.654 | 0.000 | 0.000 | 0.000 | 0.861 | 100.776 | 58.082 | 0.000 | 0.000 | 43.975 | 0.000 | 0.232 | 0.096 | 0.000 | 0.000 | 0.000 | 0.000 | 0.394 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.521 | 0.319 | 0.160 | 0.000 | 0.266 | 0.319 | 0.160 | 0.000 | 0.000 | 0.000 | 0.141 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.672 | 0.672 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.576 |
