# V1.9 LUT Resource Allocation Summary

- rollout rows: 1000
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.250 | 0.250 | 0.000 | 0.000 | 0.307 | 0.424 | 0.566 | 0.464-0.696 | 0.424 | 3.400 | 0.000 | 0.000 | 0.000 | 0.126 | 64.388 | 58.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.801 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 1.000 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 |
