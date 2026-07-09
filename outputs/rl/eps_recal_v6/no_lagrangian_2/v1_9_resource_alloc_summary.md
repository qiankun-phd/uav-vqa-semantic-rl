# V1.9 LUT Resource Allocation Summary

- rollout rows: 355
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.152 | 0.234 | 0.156 | 0.162 | 0.595 | 0.612 | 0.606 | 0.464-0.696 | 0.484 | 2.197 | 0.000 | 1057.682 | 0.023 | 2.468 | 304.473 | 298.194 | 2.267 | 0.021 | 0.126 | 0.000 | 0.715 | 0.048 | 0.000 | 0.000 | 0.003 | 0.003 | -0.427 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.454 | 0.146 | 0.000 | 0.000 | 0.454 | 0.146 | 0.000 | 0.000 | 0.000 | 0.400 | 0.056 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.532 | 0.221 | 0.400 | 0.400 | 0.000 | 0.400 | 9.419 | 0.033 | 0.580 |
