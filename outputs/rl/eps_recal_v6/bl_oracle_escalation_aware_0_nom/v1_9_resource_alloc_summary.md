# V1.9 LUT Resource Allocation Summary

- rollout rows: 966
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_escalation_aware | 0.912 | 0.703 | 0.454 | 0.465 | 0.302 | 0.405 | 0.592 | 0.529-0.696 | 0.040 | 0.297 | 0.000 | 4033.366 | 0.000 | 8.018 | 865.048 | 768.842 | 6.538 | 0.063 | 79.373 | 0.000 | 0.088 | 0.234 | 0.000 | 0.000 | 0.000 | 0.000 | -0.068 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_escalation_aware | 0.259 | 0.415 | 0.293 | 0.000 | 0.000 | 0.415 | 0.293 | 0.000 | 0.000 | 0.033 | 0.126 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_escalation_aware | 0.703 | 0.727 | 0.033 | 0.000 | 0.033 | 0.000 | 2.637 | 0.010 | 0.436 |
