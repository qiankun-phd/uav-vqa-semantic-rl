# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.771 | 0.540 | 0.515 | 0.531 | 0.308 | 0.279 | 0.504 | 0.504-0.504 | 0.001 | 0.009 | 20.141 | 1701.238 | 0.000 | 3.083 | 349.076 | 297.581 | 2.190 | 0.023 | 41.535 | 0.000 | 0.229 | 0.233 | 0.000 | 0.000 | 0.000 | 0.000 | -1.581 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.767 | 0.007 | 0.226 | 0.000 | 0.513 | 0.007 | 0.226 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_best_feasible_evidence | 0.540 | 0.540 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.307 |
