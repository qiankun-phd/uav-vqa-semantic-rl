# V1.9 LUT Resource Allocation Summary

- rollout rows: 395
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 0.201 | 0.239 | 0.767 | 0.816 | 0.816 | 0.650-0.820 | 0.615 | 2.813 | 0.578 | 44.302 | 0.025 | 0.187 | 41.410 | 38.000 | 0.095 | -0.002 | 0.024 | 0.000 | 1.000 | 0.018 | 0.000 | 0.000 | 0.003 | 0.003 | -0.661 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.344 | 0.020 | 0.000 | 0.000 | 0.344 | 0.020 | 0.000 | 0.000 | 0.000 | 0.635 | 0.013 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.635 | 0.000 | 0.635 | 0.635 | 0.000 | 0.635 | 19.348 | 0.038 | 0.613 |
