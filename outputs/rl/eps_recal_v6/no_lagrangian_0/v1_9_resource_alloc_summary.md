# V1.9 LUT Resource Allocation Summary

- rollout rows: 342
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.190 | 0.249 | 0.170 | 0.177 | 0.540 | 0.612 | 0.605 | 0.464-0.696 | 0.473 | 2.103 | 0.000 | 1262.631 | 0.000 | 2.969 | 362.440 | 352.638 | 2.714 | 0.025 | 0.160 | 0.000 | 0.728 | 0.023 | 0.000 | 0.000 | 0.000 | 0.000 | -0.313 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.453 | 0.184 | 0.000 | 0.000 | 0.453 | 0.184 | 0.000 | 0.000 | 0.000 | 0.363 | 0.038 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.544 | 0.284 | 0.363 | 0.363 | 0.000 | 0.363 | 7.682 | 0.023 | 0.588 |
