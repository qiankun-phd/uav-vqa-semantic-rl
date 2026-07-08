# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.690 | 0.347 | 0.418 | 0.439 | 0.315 | 0.280 | 0.504 | 0.504-0.504 | 0.058 | 0.547 | 28.185 | 2577.792 | 0.000 | 5.499 | 602.492 | 556.297 | 4.560 | 0.039 | 42.684 | 0.000 | 0.310 | 0.486 | 0.000 | 0.000 | 0.000 | 0.000 | -2.226 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.512 | 0.253 | 0.235 | 0.000 | 0.257 | 0.253 | 0.235 | 0.000 | 0.000 | 0.000 | 0.003 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.347 | 0.347 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.139 |
