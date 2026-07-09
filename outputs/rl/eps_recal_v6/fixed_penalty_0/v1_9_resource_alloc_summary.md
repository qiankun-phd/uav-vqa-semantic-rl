# V1.9 LUT Resource Allocation Summary

- rollout rows: 355
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.197 | 0.228 | 0.169 | 0.175 | 0.553 | 0.607 | 0.598 | 0.464-0.696 | 0.469 | 2.128 | 0.000 | 1169.292 | 0.000 | 2.834 | 346.860 | 337.166 | 2.596 | 0.027 | 0.161 | 0.000 | 0.749 | 0.023 | 0.000 | 0.000 | 0.000 | 0.000 | -0.301 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.431 | 0.186 | 0.000 | 0.000 | 0.431 | 0.186 | 0.000 | 0.000 | 0.000 | 0.383 | 0.037 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.563 | 0.292 | 0.383 | 0.383 | 0.000 | 0.383 | 7.849 | 0.021 | 0.589 |
