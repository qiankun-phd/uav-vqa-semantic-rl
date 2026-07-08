# V1.9 LUT Resource Allocation Summary

- rollout rows: 982
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.601 | 0.533 | 0.420 | 0.435 | 0.309 | 0.282 | 0.504 | 0.504-0.504 | 0.028 | 0.173 | 16.706 | 1723.402 | 0.000 | 1.915 | 255.236 | 236.160 | 1.560 | 0.007 | 12.037 | 0.000 | 0.399 | 0.070 | 0.001 | 0.000 | 0.000 | 0.000 | -1.253 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.930 | 0.005 | 0.065 | 0.000 | 0.675 | 0.005 | 0.065 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.533 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.462 |
