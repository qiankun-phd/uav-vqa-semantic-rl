# V1.9 LUT Resource Allocation Summary

- rollout rows: 980
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.774 | 0.678 | 0.401 | 0.408 | 0.279 | 0.417 | 0.548 | 0.464-0.696 | 0.075 | 0.640 | 0.000 | 0.000 | 0.000 | 1.374 | 124.258 | 80.966 | 0.518 | -0.006 | 43.930 | 0.000 | 0.226 | 0.097 | 0.000 | 0.000 | 0.000 | 0.000 | 0.397 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.512 | 0.328 | 0.160 | 0.000 | 0.257 | 0.328 | 0.160 | 0.000 | 0.000 | 0.000 | 0.135 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.678 | 0.678 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.581 |
