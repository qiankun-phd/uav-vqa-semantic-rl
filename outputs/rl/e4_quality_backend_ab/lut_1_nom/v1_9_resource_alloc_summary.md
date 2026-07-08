# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.741 | 0.484 | 0.528 | 0.548 | 0.312 | 0.572 | 0.714 | 0.650-0.820 | 0.072 | 0.563 | 26.156 | 2751.475 | 0.000 | 4.838 | 562.497 | 526.573 | 4.194 | -0.003 | 18.117 | 0.000 | 0.259 | 0.314 | 0.006 | 0.000 | 0.000 | 0.000 | -2.019 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.682 | 0.217 | 0.102 | 0.000 | 0.427 | 0.217 | 0.102 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.484 | 0.484 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.170 |
