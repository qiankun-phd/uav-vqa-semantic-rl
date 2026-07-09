# V1.9 LUT Resource Allocation Summary

- rollout rows: 338
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.183 | 0.192 | 0.165 | 0.172 | 0.537 | 0.606 | 0.601 | 0.464-0.696 | 0.470 | 2.087 | 0.000 | 0.000 | 0.018 | 0.253 | 58.730 | 48.971 | 0.010 | -0.000 | 0.161 | 0.000 | 0.769 | 0.036 | 0.000 | 0.000 | 0.006 | 0.006 | -0.278 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.473 | 0.180 | 0.000 | 0.000 | 0.473 | 0.180 | 0.000 | 0.000 | 0.000 | 0.346 | 0.041 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.506 | 0.244 | 0.346 | 0.346 | 0.000 | 0.346 | 8.244 | 0.019 | 0.491 |
