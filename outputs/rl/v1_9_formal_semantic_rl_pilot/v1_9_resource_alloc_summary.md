# V1.9 LUT Resource Allocation Summary

- rollout rows: 8
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.000 | 0.733 | 8.056 | 937.811 | 179.500 | 0.000 | 0.500 | 1.000 | 0.000 | 0.000 | 0.000 | -3.845 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 0.000 | 0.000 | 1.000 | 0.000 |
