# V1.9 LUT Resource Allocation Summary

- rollout rows: 1000
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.943 | 0.279 | 0.632 | 0.657 | 0.317 | 0.608 | 0.754 | 0.650-0.820 | 0.016 | 0.147 | 95.418 | 10279.968 | 0.000 | 12.565 | 1447.403 | 1428.487 | 12.147 | 0.064 | 2.047 | 0.000 | 0.057 | 0.721 | 0.069 | 0.000 | 0.000 | 0.000 | -3.776 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.593 | 0.398 | 0.009 | 0.000 |
