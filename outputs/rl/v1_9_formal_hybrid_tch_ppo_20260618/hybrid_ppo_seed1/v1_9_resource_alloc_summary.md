# V1.9 LUT Resource Allocation Summary

- rollout rows: 8000
- trained PPO: True
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality violation | deadline violation | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.017 | 0.411 | 0.356 | 1.610 | 0.000 | 0.000 | 0.983 | 0.000 | -1.004 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 |
|---|---:|---:|---:|---:|
| ppo | 1.000 | 0.000 | 0.000 | 0.000 |
