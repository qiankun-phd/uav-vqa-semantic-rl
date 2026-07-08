# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.671 | 0.538 | 0.463 | 0.478 | 0.307 | 0.280 | 0.504 | 0.504-0.504 | 0.013 | 0.057 | 17.893 | 1807.296 | 0.000 | 2.607 | 318.619 | 286.444 | 2.030 | 0.008 | 23.835 | 0.000 | 0.329 | 0.133 | 0.000 | 0.000 | 0.000 | 0.000 | -1.436 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.866 | 0.004 | 0.130 | 0.000 | 0.611 | 0.004 | 0.130 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.538 | 0.538 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.405 |
