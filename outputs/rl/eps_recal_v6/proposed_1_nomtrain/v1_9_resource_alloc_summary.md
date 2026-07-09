# V1.9 LUT Resource Allocation Summary

- rollout rows: 968
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.758 | 0.750 | 0.425 | 0.434 | 0.276 | 0.407 | 0.607 | 0.464-0.696 | 0.049 | 0.260 | 0.000 | 0.000 | 0.000 | 0.583 | 37.824 | 12.419 | 0.010 | -0.000 | 8.116 | 0.000 | 0.242 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 1.196 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.389 | 0.583 | 0.028 | 0.000 | 0.131 | 0.583 | 0.028 | 0.000 | 0.000 | 0.000 | 0.176 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.742 |
