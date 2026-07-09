# V1.9 LUT Resource Allocation Summary

- rollout rows: 374
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.016 | 0.487 | 0.016 | 0.020 | 0.692 | 0.637 | 0.636 | 0.464-0.696 | 0.623 | 2.787 | 0.000 | 0.000 | 0.000 | 0.122 | 43.059 | 38.835 | 0.009 | -0.000 | 0.016 | 0.000 | 0.497 | 0.037 | 0.000 | 0.000 | 0.000 | 0.000 | -0.743 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.516 | 0.013 | 0.000 | 0.000 | 0.516 | 0.013 | 0.000 | 0.000 | 0.000 | 0.471 | 0.008 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.487 | 0.030 | 0.471 | 0.471 | 0.000 | 0.471 | 12.346 | 0.018 | 0.920 |
