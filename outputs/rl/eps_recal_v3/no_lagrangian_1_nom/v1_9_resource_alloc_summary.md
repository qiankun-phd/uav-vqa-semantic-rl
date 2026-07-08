# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.668 | 0.534 | 0.460 | 0.476 | 0.308 | 0.280 | 0.504 | 0.504-0.504 | 0.014 | 0.063 | 24.568 | 2488.893 | 0.000 | 3.145 | 382.451 | 350.837 | 2.577 | 0.012 | 23.277 | 0.000 | 0.332 | 0.133 | 0.006 | 0.000 | 0.000 | 0.000 | -1.572 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.867 | 0.006 | 0.127 | 0.000 | 0.612 | 0.006 | 0.127 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.534 | 0.534 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.401 |
