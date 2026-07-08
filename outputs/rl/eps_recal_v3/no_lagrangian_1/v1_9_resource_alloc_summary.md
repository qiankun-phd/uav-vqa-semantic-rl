# V1.9 LUT Resource Allocation Summary

- rollout rows: 400
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.007 | 0.007 | 0.216 | 0.257 | 0.765 | 0.499 | 0.499 | 0.166-0.504 | 0.309 | 1.338 | 0.000 | 0.000 | 0.000 | 0.096 | 33.042 | 29.876 | 0.009 | -0.000 | 0.009 | 0.000 | 0.993 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | -0.690 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.372 | 0.007 | 0.000 | 0.000 | 0.372 | 0.007 | 0.000 | 0.000 | 0.000 | 0.620 | 0.007 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.627 | 0.020 | 0.620 | 0.620 | 0.000 | 0.620 | 13.534 | 0.021 | 0.623 |
