# Semantic Path Cache/Defer Short Benchmark

- controller: `semantic_path_two_timescale_ppo`
- semantic paths: cache / token / image / defer / cache_update
- train episodes: 300 per seed
- eval episodes: 50 per seed
- seeds: 0,1,2
- tasks per episode: 12
- device: cuda:0

## Proposed Controller Summary

| scenario | semantic success | task success | deadline vio | energy | payload KB | UTM vio | cache | token | image | defer | cache update | cache eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | 0.316 | 0.278 | 0.144 | 96.162 | 0.641 | 0.000 | 0.451 | 0.504 | 0.000 | 0.000 | 0.044 | 0.178 |
| edge_overload | 0.609 | 0.231 | 0.569 | 443.407 | 0.864 | 0.000 | 0.034 | 0.585 | 0.000 | 0.000 | 0.381 | 0.131 |
| low_snr_blockage | 0.931 | 0.271 | 0.699 | 114.598 | 0.672 | 0.000 | 0.305 | 0.450 | 0.000 | 0.010 | 0.235 | 0.278 |
| low_snr_soft | 0.931 | 0.271 | 0.699 | 114.598 | 0.672 | 0.000 | 0.305 | 0.450 | 0.000 | 0.010 | 0.235 | 0.278 |
| normal_patrol | 0.378 | 0.230 | 0.251 | 799.453 | 0.751 | 0.000 | 0.257 | 0.688 | 0.000 | 0.004 | 0.050 | 0.075 |
| utm_conflict | 0.000 | 0.000 | 0.541 | 1083.136 | 0.592 | 0.091 | 0.501 | 0.499 | 0.000 | 0.000 | 0.000 | 0.017 |

## Comparison With Previous B_state_v2_fixed_128x128

| scenario | prev semantic | new semantic | delta | prev task | new task | delta | prev deadline | new deadline | delta | prev payload | new payload | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | 0.180 | 0.316 | +0.136 | 0.099 | 0.278 | +0.179 | 0.211 | 0.144 | -0.067 | 4.208 | 0.641 | -3.567 |
| edge_overload | 0.649 | 0.609 | -0.041 | 0.649 | 0.231 | -0.418 | 0.000 | 0.569 | +0.569 | 0.823 | 0.864 | +0.041 |
| low_snr_blockage | 0.948 | 0.931 | -0.017 | 0.128 | 0.271 | +0.143 | 0.854 | 0.699 | -0.155 | 0.918 | 0.672 | -0.246 |
| low_snr_soft | 0.948 | 0.931 | -0.017 | 0.128 | 0.271 | +0.143 | 0.854 | 0.699 | -0.155 | 0.918 | 0.672 | -0.246 |
| normal_patrol | 0.156 | 0.378 | +0.222 | 0.156 | 0.230 | +0.075 | 0.000 | 0.251 | +0.251 | 30.742 | 0.751 | -29.990 |
| utm_conflict | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.541 | +0.541 | 0.997 | 0.592 | -0.405 |

## Cache Collapse Analysis

- `disaster_hotspot`: cache ratio 0.451, token 0.504, cache_update 0.044, quality gap 0.133.
- `edge_overload`: cache ratio 0.034, token 0.585, cache_update 0.381, quality gap 0.077.
- `low_snr_blockage`: cache ratio 0.305, token 0.450, cache_update 0.235, quality gap 0.023.
- `low_snr_soft`: cache ratio 0.305, token 0.450, cache_update 0.235, quality gap 0.023.
- `normal_patrol`: cache ratio 0.257, token 0.688, cache_update 0.050, quality gap 0.140.
- `utm_conflict`: cache ratio 0.501, token 0.499, cache_update 0.000, quality gap 0.154.

## Notes

- `normal_patrol` is routed to the calibrated `nominal_patrol` preset; `low_snr_soft` is currently an alias of `low_snr_blockage` until Environment adds a distinct soft preset.
- The controller does not collapse to always-cache: proposed cache ratios remain mixed with token/cache_update except UTM/cache-heavy pressure cases where cache is feasible but not semantically successful.
- Large rollout CSVs and `.pt` checkpoints remain in per-scenario directories for local analysis but should not be committed.
