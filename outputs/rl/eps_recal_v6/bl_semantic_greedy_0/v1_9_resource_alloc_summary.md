# V1.9 LUT Resource Allocation Summary

- rollout rows: 250
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.916 | 0.268 | 0.719 | 0.750 | 0.032 | 0.631 | 0.696 | 0.696-0.696 | 0.058 | 0.189 | 0.040 | 2934.424 | 1.916 | 13.385 | 1464.204 | 1406.743 | 11.957 | 0.157 | 24.451 | 0.000 | 0.084 | 0.380 | 0.000 | 0.000 | 0.612 | 0.612 | -2.302 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.008 | 0.908 | 0.084 | 0.000 | 0.008 | 0.908 | 0.084 | 0.000 | 0.000 | 0.000 | 0.008 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_greedy | 0.268 | 0.268 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.336 |
