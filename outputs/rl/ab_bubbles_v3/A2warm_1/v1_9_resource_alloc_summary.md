# V1.9 LUT Resource Allocation Summary

- rollout rows: 500
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 0.559 | 0.750 | 0.405 | 0.820 | 0.820 | 0.820-0.820 | 0.261 | 1.344 | 72.455 | 7778.059 | 1.746 | 15.540 | 1762.182 | 1737.792 | 14.777 | 0.116 | 0.727 | 0.000 | 1.000 | 0.684 | 0.108 | 0.000 | 0.302 | 0.302 | -5.545 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.388 | 0.612 | 0.000 | 0.000 |
