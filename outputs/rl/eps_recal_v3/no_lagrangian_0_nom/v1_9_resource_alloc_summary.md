# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.631 | 0.542 | 0.433 | 0.447 | 0.306 | 0.279 | 0.504 | 0.504-0.504 | 0.022 | 0.122 | 1.291 | 0.000 | 0.000 | 0.530 | 86.131 | 62.407 | 0.096 | -0.001 | 16.323 | 0.000 | 0.369 | 0.089 | 0.000 | 0.000 | 0.000 | 0.000 | -0.909 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.909 | 0.002 | 0.089 | 0.000 | 0.655 | 0.002 | 0.089 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.542 | 0.542 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.454 |
