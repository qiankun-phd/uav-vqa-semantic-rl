# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.727 | 0.540 | 0.487 | 0.504 | 0.310 | 0.279 | 0.504 | 0.504-0.504 | 0.002 | 0.022 | 11.213 | 1045.019 | 0.000 | 2.078 | 270.653 | 258.918 | 1.834 | 0.018 | 0.873 | 0.000 | 0.273 | 0.191 | 0.000 | 0.000 | 0.000 | 0.000 | -1.275 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.809 | 0.187 | 0.004 | 0.000 | 0.554 | 0.187 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.540 | 0.540 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.349 |
