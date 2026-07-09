# V1.9 LUT Resource Allocation Summary

- rollout rows: 976
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.907 | 0.626 | 0.461 | 0.470 | 0.268 | 0.418 | 0.552 | 0.464-0.696 | 0.034 | 0.225 | 0.000 | 3881.652 | 0.000 | 7.610 | 825.445 | 741.331 | 6.269 | 0.061 | 67.600 | 0.000 | 0.093 | 0.339 | 0.000 | 0.000 | 0.000 | 0.000 | -0.265 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.308 | 0.443 | 0.249 | 0.000 | 0.052 | 0.443 | 0.249 | 0.000 | 0.000 | 0.000 | 0.111 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.626 | 0.626 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.287 |
