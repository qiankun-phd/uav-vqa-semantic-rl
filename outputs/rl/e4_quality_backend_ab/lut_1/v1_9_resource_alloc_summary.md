# V1.9 LUT Resource Allocation Summary

- rollout rows: 335
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 0.280 | 0.342 | 0.624 | 0.819 | 0.819 | 0.650-0.820 | 0.539 | 2.194 | 0.000 | 0.000 | 0.000 | 0.143 | 52.523 | 47.547 | 0.010 | -0.000 | 0.011 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.782 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.594 | 0.009 | 0.000 | 0.000 | 0.594 | 0.009 | 0.000 | 0.000 | 0.000 | 0.397 | 0.006 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.397 | 0.000 | 0.397 | 0.397 | 0.000 | 0.397 | 8.317 | 0.011 | 0.397 |
