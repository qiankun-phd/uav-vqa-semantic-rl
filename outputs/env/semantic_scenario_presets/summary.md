# Semantic Scenario Preset Smoke Summary

Generated from `src/vqa_semcom/sim/multi_uav_env.py` using `configs/v1_9_snr_lut.yaml`.

| scenario | steps | mean LCB | mean gap | delay s | energy J | SINR dB | UTM conflict | risk violation | services |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nominal_patrol | 5 | 0.665 | 0.114 | 13.845 | 1968.702 | 30.289 | 0.000 | 1.000 | 1;0;2;1;0 |
| disaster_hotspot | 5 | 0.510 | 0.330 | 8.809 | 1069.477 | 41.055 | 0.000 | 1.000 | 2;1;2;1;2 |
| low_snr_blockage | 5 | 0.338 | 0.480 | 518.896 | 4496.708 | -34.574 | 0.000 | 0.800 | 0;1;1;2;0 |
| edge_overload | 5 | 0.366 | 0.274 | 9.777 | 1564.751 | 35.068 | 0.000 | 1.000 | 1;2;1;2;1 |
| utm_conflict | 5 | 0.213 | 0.607 | 20.347 | 2819.211 | 15.693 | 0.400 | 1.000 | 1;2;1;2;1 |

Required info fields were checked for every smoke step:

`semantic_accuracy_lcb`, `semantic_quality_gap`, `epsilon_k`, `deadline_s`, `energy_j`, `utm_delay_s`, `utm_conflict_violation`, `risk_violation`, `airspace_state`

Service level 3 remains disabled in all preset smokes.
