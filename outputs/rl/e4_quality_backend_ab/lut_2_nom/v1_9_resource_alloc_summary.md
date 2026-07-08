# V1.9 LUT Resource Allocation Summary

- rollout rows: 979
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.641 | 0.641 | 0.469 | 0.485 | 0.308 | 0.564 | 0.694 | 0.650-0.820 | 0.114 | 0.715 | 0.000 | 0.000 | 0.000 | 0.210 | 58.332 | 49.255 | 0.004 | -0.000 | 0.136 | 0.000 | 0.359 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.257 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.880 | 0.120 | 0.000 | 0.000 | 0.625 | 0.120 | 0.000 | 0.000 | 0.000 | 0.000 | 0.026 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.641 | 0.641 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.641 |
