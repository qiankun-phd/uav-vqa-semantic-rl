# V1.9 LUT Resource Allocation Summary

- rollout rows: 4
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy LCB | accuracy mean | uncertainty | quality gap | Q_quality | Q_deadline | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM violation | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.364 | 0.381 | 0.062 | -0.286 | 0.649 | 0.000 | 0.176 | 1.610 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.036 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 1.000 | 0.000 | 0.000 | 0.000 |
