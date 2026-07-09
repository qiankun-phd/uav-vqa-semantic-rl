# V1.9 LUT Resource Allocation Summary

- rollout rows: 250
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.916 | 0.260 | 0.757 | 0.795 | 0.039 | 0.631 | 0.696 | 0.696-0.696 | 0.028 | 0.091 | 0.080 | 2979.994 | 1.916 | 13.827 | 1493.049 | 1406.743 | 11.957 | 0.157 | 56.044 | 0.000 | 0.084 | 0.412 | 0.000 | 0.000 | 0.612 | 0.612 | -2.490 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.008 | 0.748 | 0.244 | 0.000 | 0.008 | 0.748 | 0.244 | 0.000 | 0.000 | 0.000 | 0.008 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.260 | 0.260 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.376 |
