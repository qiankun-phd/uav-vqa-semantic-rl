# V1.9 LUT Resource Allocation Summary

- rollout rows: 982
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.489 | 0.489 | 0.376 | 0.390 | 0.306 | 0.572 | 0.744 | 0.650-0.820 | 0.204 | 1.451 | 1.849 | 208.434 | 0.000 | 0.324 | 86.145 | 80.052 | 0.192 | 0.001 | 0.010 | 0.000 | 0.511 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | -0.848 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.991 | 0.009 | 0.000 | 0.000 | 0.736 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.489 | 0.489 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.480 |
