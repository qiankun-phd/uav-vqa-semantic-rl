# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.649 | 0.540 | 0.445 | 0.461 | 0.308 | 0.279 | 0.504 | 0.504-0.504 | 0.018 | 0.093 | 18.003 | 1766.033 | 0.000 | 2.371 | 297.437 | 270.183 | 1.876 | 0.008 | 19.507 | 0.000 | 0.351 | 0.111 | 0.000 | 0.000 | 0.000 | 0.000 | -1.376 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.889 | 0.005 | 0.106 | 0.000 | 0.635 | 0.005 | 0.106 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.540 | 0.540 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.429 |
