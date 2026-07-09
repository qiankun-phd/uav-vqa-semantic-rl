# V1.9 LUT Resource Allocation Summary

- rollout rows: 986
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.540 | 0.441 | 0.262 | 0.274 | 0.311 | 0.398 | 0.490 | 0.464-0.696 | 0.187 | 1.798 | 0.000 | 2282.578 | 0.000 | 4.064 | 471.541 | 435.582 | 3.424 | 0.007 | 18.455 | 0.000 | 0.460 | 0.101 | 0.002 | 0.000 | 0.000 | 0.000 | -0.252 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.675 | 0.222 | 0.102 | 0.000 | 0.422 | 0.222 | 0.102 | 0.000 | 0.000 | 0.000 | 0.059 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.441 | 0.441 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.340 |
