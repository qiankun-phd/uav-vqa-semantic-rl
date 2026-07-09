# V1.9 LUT Resource Allocation Summary

- rollout rows: 435
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_escalation_aware | 0.354 | 0.789 | 0.286 | 0.296 | 0.656 | 0.652 | 0.660 | 0.464-0.696 | 0.426 | 1.855 | 0.000 | 2288.712 | 1.389 | 4.801 | 529.634 | 511.827 | 4.352 | 0.056 | 5.671 | 0.000 | 0.000 | 0.030 | 0.000 | 0.000 | 0.193 | 0.193 | -0.655 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_escalation_aware | 0.000 | 0.331 | 0.023 | 0.000 | 0.000 | 0.331 | 0.023 | 0.000 | 0.000 | 0.646 | 0.117 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_escalation_aware | 0.789 | 0.403 | 0.646 | 0.644 | 0.002 | 0.644 | 10.798 | 0.031 | 1.014 |
