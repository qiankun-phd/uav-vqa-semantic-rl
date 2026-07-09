# V1.9 LUT Resource Allocation Summary

- rollout rows: 980
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.772 | 0.672 | 0.425 | 0.432 | 0.277 | 0.418 | 0.579 | 0.464-0.696 | 0.054 | 0.337 | 0.000 | 4238.211 | 0.000 | 4.456 | 512.052 | 473.077 | 3.659 | 0.027 | 33.989 | 0.000 | 0.228 | 0.100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.337 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.477 | 0.400 | 0.123 | 0.000 | 0.221 | 0.400 | 0.123 | 0.000 | 0.000 | 0.000 | 0.149 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.672 | 0.672 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.572 |
