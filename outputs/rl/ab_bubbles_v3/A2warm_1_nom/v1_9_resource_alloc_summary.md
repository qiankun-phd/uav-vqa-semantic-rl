# V1.9 LUT Resource Allocation Summary

- rollout rows: 1000
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.948 | 0.274 | 0.634 | 0.659 | 0.317 | 0.608 | 0.758 | 0.650-0.820 | 0.015 | 0.139 | 98.740 | 10655.156 | 0.000 | 13.084 | 1508.505 | 1489.535 | 12.666 | 0.066 | 2.050 | 0.000 | 0.052 | 0.726 | 0.076 | 0.000 | 0.000 | 0.000 | -3.921 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.591 | 0.400 | 0.009 | 0.000 |
