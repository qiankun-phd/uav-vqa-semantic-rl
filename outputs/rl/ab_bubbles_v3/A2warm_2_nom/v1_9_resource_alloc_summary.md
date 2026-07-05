# V1.9 LUT Resource Allocation Summary

- rollout rows: 1000
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.948 | 0.275 | 0.634 | 0.658 | 0.317 | 0.608 | 0.758 | 0.650-0.820 | 0.015 | 0.139 | 97.130 | 10471.338 | 0.000 | 12.891 | 1485.725 | 1466.584 | 12.471 | 0.066 | 2.225 | 0.000 | 0.052 | 0.725 | 0.074 | 0.000 | 0.000 | 0.000 | -3.869 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.593 | 0.397 | 0.010 | 0.000 |
