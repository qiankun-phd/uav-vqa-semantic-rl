# V1.9 LUT Resource Allocation Summary

- rollout rows: 380
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.005 | 0.518 | 0.004 | 0.005 | 0.709 | 0.638 | 0.637 | 0.464-0.696 | 0.634 | 2.901 | 0.000 | 0.000 | 0.000 | 0.110 | 40.433 | 36.634 | 0.006 | -0.000 | 0.006 | 0.000 | 0.482 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.742 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.461 | 0.005 | 0.000 | 0.000 | 0.461 | 0.005 | 0.000 | 0.000 | 0.000 | 0.534 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.539 | 0.011 | 0.534 | 0.534 | 0.000 | 0.534 | 12.361 | 0.018 | 1.053 |
