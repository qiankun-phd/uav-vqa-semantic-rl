# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.772 | 0.487 | 0.548 | 0.569 | 0.313 | 0.571 | 0.724 | 0.650-0.820 | 0.055 | 0.504 | 27.040 | 2862.149 | 0.000 | 4.584 | 511.481 | 456.378 | 3.640 | 0.011 | 33.790 | 0.000 | 0.228 | 0.343 | 0.009 | 0.000 | 0.000 | 0.000 | -1.958 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.617 | 0.195 | 0.187 | 0.000 | 0.363 | 0.195 | 0.187 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.487 | 0.487 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.144 |
