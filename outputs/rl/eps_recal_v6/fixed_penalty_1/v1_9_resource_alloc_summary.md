# V1.9 LUT Resource Allocation Summary

- rollout rows: 371
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.181 | 0.221 | 0.155 | 0.160 | 0.611 | 0.607 | 0.602 | 0.464-0.696 | 0.481 | 2.307 | 0.000 | 2139.541 | 0.000 | 4.027 | 487.019 | 479.510 | 3.827 | 0.035 | 0.132 | 0.000 | 0.739 | 0.040 | 0.000 | 0.000 | 0.000 | 0.000 | -0.424 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.418 | 0.151 | 0.000 | 0.000 | 0.418 | 0.151 | 0.000 | 0.000 | 0.000 | 0.431 | 0.051 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.585 | 0.270 | 0.431 | 0.431 | 0.000 | 0.431 | 10.085 | 0.030 | 0.612 |
