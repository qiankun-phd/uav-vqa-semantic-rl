# V1.9 LUT Resource Allocation Summary

- rollout rows: 980
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.782 | 0.688 | 0.402 | 0.409 | 0.278 | 0.416 | 0.545 | 0.464-0.696 | 0.074 | 0.628 | 0.000 | 3066.304 | 0.000 | 3.389 | 361.844 | 317.780 | 2.532 | 0.020 | 43.661 | 0.000 | 0.218 | 0.094 | 0.000 | 0.000 | 0.000 | 0.000 | 0.318 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.511 | 0.330 | 0.159 | 0.000 | 0.256 | 0.330 | 0.159 | 0.000 | 0.000 | 0.000 | 0.138 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.688 | 0.688 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.594 |
