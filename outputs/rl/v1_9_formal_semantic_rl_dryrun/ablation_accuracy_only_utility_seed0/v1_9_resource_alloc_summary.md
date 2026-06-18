# V1.9 LUT Resource Allocation Summary

- rollout rows: 2
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.491 | 8.117 | 945.713 | 175.122 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | -4.352 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 1.000 | 0.000 |
