# V1.9 LUT Resource Allocation Summary

- rollout rows: 356
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.152 | 0.236 | 0.156 | 0.161 | 0.597 | 0.612 | 0.606 | 0.464-0.696 | 0.484 | 2.202 | 0.000 | 941.882 | 0.022 | 2.277 | 282.075 | 275.813 | 2.077 | 0.023 | 0.126 | 0.000 | 0.713 | 0.048 | 0.000 | 0.000 | 0.003 | 0.003 | -0.419 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.452 | 0.146 | 0.000 | 0.000 | 0.452 | 0.146 | 0.000 | 0.000 | 0.000 | 0.402 | 0.056 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.534 | 0.221 | 0.402 | 0.402 | 0.000 | 0.402 | 9.392 | 0.032 | 0.584 |
