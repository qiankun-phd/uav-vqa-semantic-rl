# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.490 | 0.483 | 0.424 | 0.443 | 0.302 | 0.571 | 0.745 | 0.650-0.820 | 0.156 | 1.300 | 4.623 | 420.267 | 0.000 | 2.257 | 266.007 | 219.892 | 1.498 | -0.015 | 30.272 | 0.000 | 0.510 | 0.185 | 0.000 | 0.000 | 0.000 | 0.000 | -1.373 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.815 | 0.016 | 0.169 | 0.000 | 0.561 | 0.016 | 0.169 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.483 | 0.483 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.298 |
