# V1.9 LUT Resource Allocation Summary

- rollout rows: 330
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 0.270 | 0.329 | 0.629 | 0.818 | 0.818 | 0.650-0.820 | 0.548 | 2.198 | 0.000 | 0.000 | 0.000 | 0.136 | 51.242 | 46.439 | 0.007 | -0.000 | 0.007 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.813 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.585 | 0.006 | 0.000 | 0.000 | 0.585 | 0.006 | 0.000 | 0.000 | 0.000 | 0.409 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.409 | 0.000 | 0.409 | 0.409 | 0.000 | 0.409 | 8.685 | 0.012 | 0.409 |
