# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.654 | 0.542 | 0.448 | 0.463 | 0.306 | 0.279 | 0.504 | 0.504-0.504 | 0.017 | 0.082 | 0.405 | 0.000 | 0.000 | 0.515 | 86.596 | 58.163 | 0.000 | 0.000 | 20.572 | 0.000 | 0.346 | 0.112 | 0.000 | 0.000 | 0.000 | 0.000 | -0.914 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.886 | 0.002 | 0.112 | 0.000 | 0.632 | 0.002 | 0.112 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.542 | 0.542 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.430 |
