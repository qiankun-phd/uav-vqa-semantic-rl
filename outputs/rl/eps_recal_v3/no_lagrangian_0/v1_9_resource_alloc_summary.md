# V1.9 LUT Resource Allocation Summary

- rollout rows: 393
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.005 | 0.005 | 0.218 | 0.260 | 0.761 | 0.501 | 0.501 | 0.166-0.504 | 0.307 | 1.321 | 0.000 | 0.000 | 0.000 | 0.094 | 33.416 | 30.262 | 0.006 | -0.000 | 0.006 | 0.000 | 0.995 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | -0.716 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.389 | 0.005 | 0.000 | 0.000 | 0.389 | 0.005 | 0.000 | 0.000 | 0.000 | 0.606 | 0.005 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.611 | 0.013 | 0.606 | 0.606 | 0.000 | 0.606 | 13.775 | 0.021 | 0.601 |
