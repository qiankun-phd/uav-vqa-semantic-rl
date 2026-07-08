# V1.9 LUT Resource Allocation Summary

- rollout rows: 983
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.564 | 0.538 | 0.390 | 0.405 | 0.308 | 0.280 | 0.504 | 0.504-0.504 | 0.037 | 0.271 | 6.313 | 651.901 | 0.000 | 0.784 | 134.558 | 124.099 | 0.578 | 0.004 | 4.120 | 0.000 | 0.436 | 0.026 | 0.000 | 0.000 | 0.000 | 0.000 | -0.964 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.974 | 0.004 | 0.022 | 0.000 | 0.719 | 0.004 | 0.022 | 0.000 | 0.000 | 0.000 | 0.000 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.538 | 0.538 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.512 |
