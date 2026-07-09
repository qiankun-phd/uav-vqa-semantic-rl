# V1.9 LUT Resource Allocation Summary

- rollout rows: 975
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.778 | 0.694 | 0.396 | 0.404 | 0.276 | 0.414 | 0.556 | 0.464-0.696 | 0.078 | 0.651 | 0.000 | 0.000 | 0.000 | 0.841 | 103.280 | 58.048 | 0.001 | -0.000 | 38.376 | 0.000 | 0.222 | 0.084 | 0.000 | 0.000 | 0.000 | 0.000 | 0.597 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.442 | 0.418 | 0.139 | 0.000 | 0.186 | 0.418 | 0.139 | 0.000 | 0.000 | 0.000 | 0.130 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.694 | 0.694 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.610 |
