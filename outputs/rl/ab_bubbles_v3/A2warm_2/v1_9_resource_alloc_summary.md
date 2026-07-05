# V1.9 LUT Resource Allocation Summary

- rollout rows: 500
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 0.556 | 0.749 | 0.406 | 0.820 | 0.820 | 0.820-0.820 | 0.264 | 1.375 | 68.927 | 7385.692 | 1.852 | 15.373 | 1741.810 | 1717.245 | 14.602 | 0.121 | 0.734 | 0.000 | 1.000 | 0.726 | 0.102 | 0.000 | 0.310 | 0.310 | -5.621 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.382 | 0.618 | 0.000 | 0.000 |
