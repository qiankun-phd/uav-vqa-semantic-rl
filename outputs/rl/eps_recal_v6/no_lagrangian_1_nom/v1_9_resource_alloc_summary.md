# V1.9 LUT Resource Allocation Summary

- rollout rows: 980
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.801 | 0.699 | 0.416 | 0.423 | 0.278 | 0.416 | 0.548 | 0.464-0.696 | 0.062 | 0.572 | 0.000 | 66.000 | 0.000 | 1.173 | 102.868 | 57.052 | 0.327 | -0.004 | 45.046 | 0.000 | 0.199 | 0.102 | 0.000 | 0.000 | 0.000 | 0.000 | 0.449 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.494 | 0.342 | 0.164 | 0.000 | 0.239 | 0.342 | 0.164 | 0.000 | 0.000 | 0.000 | 0.142 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.699 | 0.699 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.597 |
