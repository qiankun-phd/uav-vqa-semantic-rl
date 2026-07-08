# V1.9 LUT Resource Allocation Summary

- rollout rows: 397
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.010 | 0.008 | 0.221 | 0.264 | 0.760 | 0.500 | 0.500 | 0.166-0.504 | 0.306 | 1.301 | 0.000 | 0.000 | 0.025 | 0.103 | 33.937 | 30.639 | 0.012 | -0.000 | 0.012 | 0.000 | 0.990 | 0.015 | 0.000 | 0.000 | 0.003 | 0.003 | -0.718 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.388 | 0.010 | 0.000 | 0.000 | 0.388 | 0.010 | 0.000 | 0.000 | 0.000 | 0.602 | 0.013 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.610 | 0.019 | 0.602 | 0.602 | 0.000 | 0.602 | 13.435 | 0.021 | 0.589 |
