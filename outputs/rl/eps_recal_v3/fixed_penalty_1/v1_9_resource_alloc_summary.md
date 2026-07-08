# V1.9 LUT Resource Allocation Summary

- rollout rows: 378
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.008 | 0.008 | 0.230 | 0.275 | 0.745 | 0.502 | 0.502 | 0.166-0.504 | 0.299 | 1.267 | 0.000 | 0.000 | 0.000 | 0.103 | 35.874 | 32.440 | 0.009 | -0.000 | 0.009 | 0.000 | 0.992 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | -0.715 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.407 | 0.008 | 0.000 | 0.000 | 0.407 | 0.008 | 0.000 | 0.000 | 0.000 | 0.585 | 0.008 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.593 | 0.019 | 0.585 | 0.585 | 0.000 | 0.585 | 13.479 | 0.021 | 0.585 |
