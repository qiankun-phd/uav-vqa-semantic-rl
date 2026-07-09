# V1.9 LUT Resource Allocation Summary

- rollout rows: 343
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.201 | 0.236 | 0.172 | 0.179 | 0.530 | 0.609 | 0.601 | 0.464-0.696 | 0.468 | 2.086 | 0.000 | 1158.099 | 0.000 | 2.935 | 359.746 | 349.644 | 2.687 | 0.027 | 0.166 | 0.000 | 0.743 | 0.020 | 0.000 | 0.000 | 0.000 | 0.000 | -0.305 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.452 | 0.192 | 0.000 | 0.000 | 0.452 | 0.192 | 0.000 | 0.000 | 0.000 | 0.356 | 0.035 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.542 | 0.290 | 0.356 | 0.356 | 0.000 | 0.356 | 7.427 | 0.018 | 0.571 |
