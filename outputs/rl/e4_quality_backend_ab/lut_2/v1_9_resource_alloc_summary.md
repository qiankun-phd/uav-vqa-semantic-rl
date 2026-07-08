# V1.9 LUT Resource Allocation Summary

- rollout rows: 320
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 0.299 | 0.365 | 0.598 | 0.817 | 0.817 | 0.650-0.820 | 0.518 | 1.988 | 0.000 | 0.000 | 0.000 | 0.163 | 56.586 | 51.110 | 0.018 | -0.000 | 0.019 | 0.000 | 1.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | -0.759 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.628 | 0.016 | 0.000 | 0.000 | 0.628 | 0.016 | 0.000 | 0.000 | 0.000 | 0.356 | 0.013 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.356 | 0.000 | 0.356 | 0.356 | 0.000 | 0.356 | 8.707 | 0.012 | 0.353 |
