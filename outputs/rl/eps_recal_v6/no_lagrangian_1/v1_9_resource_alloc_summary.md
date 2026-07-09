# V1.9 LUT Resource Allocation Summary

- rollout rows: 377
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.191 | 0.244 | 0.158 | 0.162 | 0.623 | 0.606 | 0.603 | 0.464-0.696 | 0.483 | 2.404 | 0.000 | 0.000 | 0.027 | 0.212 | 38.739 | 31.426 | 0.024 | -0.000 | 0.129 | 0.000 | 0.716 | 0.037 | 0.000 | 0.000 | 0.005 | 0.005 | -0.146 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.401 | 0.143 | 0.000 | 0.000 | 0.401 | 0.143 | 0.000 | 0.000 | 0.000 | 0.456 | 0.058 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.613 | 0.288 | 0.456 | 0.456 | 0.000 | 0.456 | 11.192 | 0.027 | 0.653 |
