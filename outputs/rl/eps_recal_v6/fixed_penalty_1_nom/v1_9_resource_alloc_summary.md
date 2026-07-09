# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.831 | 0.577 | 0.428 | 0.436 | 0.273 | 0.419 | 0.542 | 0.464-0.696 | 0.062 | 0.567 | 0.000 | 3957.410 | 0.000 | 4.383 | 482.381 | 401.775 | 3.013 | 0.026 | 85.210 | 0.000 | 0.169 | 0.266 | 0.000 | 0.000 | 0.000 | 0.000 | -0.213 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.423 | 0.262 | 0.314 | 0.000 | 0.169 | 0.262 | 0.314 | 0.000 | 0.000 | 0.000 | 0.113 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.577 | 0.577 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.311 |
