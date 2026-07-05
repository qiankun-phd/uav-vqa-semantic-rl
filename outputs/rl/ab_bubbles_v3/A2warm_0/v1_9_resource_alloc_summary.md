# V1.9 LUT Resource Allocation Summary

- rollout rows: 500
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 0.574 | 0.762 | 0.403 | 0.820 | 0.820 | 0.820-0.820 | 0.246 | 1.291 | 76.044 | 8147.981 | 1.654 | 16.455 | 1871.072 | 1847.048 | 15.706 | 0.131 | 0.711 | 0.000 | 1.000 | 0.748 | 0.110 | 0.000 | 0.296 | 0.296 | -5.863 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.402 | 0.598 | 0.000 | 0.000 |
