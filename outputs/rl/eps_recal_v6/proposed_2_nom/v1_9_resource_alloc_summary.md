# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.879 | 0.528 | 0.460 | 0.469 | 0.270 | 0.418 | 0.543 | 0.464-0.696 | 0.042 | 0.456 | 0.000 | 4706.094 | 0.000 | 8.153 | 871.595 | 766.362 | 6.448 | -0.021 | 111.157 | 0.000 | 0.121 | 0.368 | 0.000 | 0.000 | 0.000 | 0.000 | -0.604 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.358 | 0.230 | 0.412 | 0.000 | 0.104 | 0.230 | 0.412 | 0.000 | 0.000 | 0.000 | 0.091 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.528 | 0.528 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.160 |
