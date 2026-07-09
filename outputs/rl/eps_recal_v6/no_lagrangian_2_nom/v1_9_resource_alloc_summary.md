# V1.9 LUT Resource Allocation Summary

- rollout rows: 972
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.757 | 0.681 | 0.386 | 0.394 | 0.278 | 0.412 | 0.567 | 0.464-0.696 | 0.085 | 0.654 | 0.000 | 7365.291 | 0.000 | 11.028 | 1259.267 | 1222.579 | 10.263 | 0.059 | 32.329 | 0.000 | 0.243 | 0.065 | 0.019 | 0.000 | 0.000 | 0.000 | 0.186 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.458 | 0.425 | 0.117 | 0.000 | 0.201 | 0.425 | 0.117 | 0.000 | 0.000 | 0.000 | 0.134 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.681 | 0.681 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.616 |
