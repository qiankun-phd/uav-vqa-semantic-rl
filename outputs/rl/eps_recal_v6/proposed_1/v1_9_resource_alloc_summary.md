# V1.9 LUT Resource Allocation Summary

- rollout rows: 368
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.207 | 0.212 | 0.170 | 0.176 | 0.590 | 0.604 | 0.598 | 0.464-0.696 | 0.467 | 2.257 | 0.000 | 1044.303 | 0.024 | 2.655 | 325.220 | 316.434 | 2.438 | 0.024 | 0.147 | 0.000 | 0.747 | 0.041 | 0.000 | 0.000 | 0.005 | 0.005 | -0.342 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.418 | 0.168 | 0.000 | 0.000 | 0.418 | 0.168 | 0.000 | 0.000 | 0.000 | 0.413 | 0.054 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.584 | 0.292 | 0.413 | 0.413 | 0.000 | 0.413 | 9.735 | 0.024 | 0.573 |
