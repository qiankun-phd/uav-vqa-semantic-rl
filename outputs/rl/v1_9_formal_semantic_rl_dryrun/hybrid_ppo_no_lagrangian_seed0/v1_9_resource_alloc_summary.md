# V1.9 LUT Resource Allocation Summary

- rollout rows: 2
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.407 | 3.199 | 432.934 | 0.394 | 0.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | -2.314 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.500 | 0.500 | 0.000 | 0.000 |
