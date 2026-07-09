# V1.9 LUT Resource Allocation Summary

- rollout rows: 980
- trained PPO: False
- LUT oracle: outputs/lut/v1_9_snr_semantic_quality_lut.csv

| policy | semantic success | success | accuracy LCB | accuracy mean | uncertainty | epsilon | failed epsilon mean | failed epsilon range | quality gap | Q_quality | Q_deadline | Q_energy | Q_utm | delay | energy | flight energy | arrival delay | coverage gain | payload KB | payload reduction | quality violation | deadline violation | battery violation | GPU violation | conflict | UTM conflict | reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.793 | 0.684 | 0.405 | 0.413 | 0.274 | 0.415 | 0.562 | 0.464-0.696 | 0.074 | 0.609 | 0.000 | 5648.558 | 0.000 | 8.092 | 910.618 | 864.890 | 7.237 | 0.018 | 38.745 | 0.000 | 0.207 | 0.089 | 0.028 | 0.000 | 0.000 | 0.000 | 0.244 |

## Service Level Selection Ratio

| policy | cache s=0 | semantic tokens s=1 | image s=2 | roi s=3 | path cache | path token | path image | defer | cache update | reject | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.432 | 0.428 | 0.141 | 0.000 | 0.177 | 0.428 | 0.141 | 0.000 | 0.000 | 0.000 | 0.123 |

## Admission Control

| policy | admission success | admitted task success | reject | correct reject | wrong reject | reject feasible | saved energy J | saved delay s | infeasibility utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo | 0.684 | 0.684 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.595 |
