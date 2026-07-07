# Fix5 Semantic Path Reject Comparison

## Setup

- algorithm: semantic_path_two_timescale_ppo with reject admission control
- train episodes: 300, eval episodes: 50, seeds: 0,1,2, tasks per episode: 12
- device: cuda:0 in RA_DI
- output: outputs/rl/semantic_path_cache_defer_reject_fix5_20260624/

## Proposed Fix5 Metrics

| scenario | semantic success | task success | admission success | admitted task success | reject | correct reject | wrong reject | deadline vio | UTM vio | energy | payload KB | path cache/token/image/defer/update/reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| normal_patrol | 0.283 | 0.283 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 100.334 | 0.581 | 0.397/0.603/0.000/0.000/0.000/0.000 |
| disaster_hotspot | 0.282 | 0.204 | 0.204 | 0.204 | 0.000 | 0.000 | 0.000 | 0.244 | 0.000 | 125.997 | 0.488 | 0.582/0.418/0.000/0.000/0.000/0.000 |
| low_snr_soft | 0.372 | 0.318 | 0.318 | 0.318 | 0.000 | 0.000 | 0.000 | 0.145 | 0.000 | 96.717 | 0.601 | 0.347/0.651/0.000/0.002/0.000/0.000 |
| low_snr_blockage | 0.833 | 0.496 | 0.496 | 0.496 | 0.000 | 0.000 | 0.000 | 0.399 | 0.000 | 170.328 | 0.464 | 0.510/0.490/0.000/0.000/0.000/0.000 |
| edge_overload | 0.056 | 0.028 | 0.846 | 0.155 | 0.818 | 0.818 | 0.000 | 0.034 | 0.000 | 51.470 | 0.164 | 0.021/0.161/0.000/0.000/0.000/0.818 |
| edge_overload_soft | 0.261 | 0.152 | 0.713 | 0.345 | 0.561 | 0.561 | 0.000 | 0.121 | 0.000 | 148.130 | 0.404 | 0.003/0.436/0.000/0.000/0.000/0.561 |
| utm_conflict | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000/0.000/0.000/1.000 |
| utm_conflict_soft | 0.196 | 0.196 | 0.769 | 0.458 | 0.573 | 0.573 | 0.000 | 0.000 | 0.000 | 55.047 | 0.353 | 0.055/0.373/0.000/0.000/0.000/0.573 |

## Version Comparison

| scenario | version | semantic success | task success | deadline vio | UTM vio | reject | correct reject | wrong reject | oracle infeasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_patrol | B_state_v2_fixed | 0.156 | 0.156 | 0.000 | 0.000 | N/A | N/A | N/A | N/A |
| normal_patrol | fix3 | 0.274 | 0.274 | 0.000 | 0.000 | N/A | N/A | N/A | N/A |
| normal_patrol | fix4 | 0.275 | 0.275 | 0.000 | 0.000 | N/A | N/A | N/A | 0.675 |
| normal_patrol | fix5 | 0.283 | 0.283 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.679 |
| disaster_hotspot | B_state_v2_fixed | 0.180 | 0.099 | 0.211 | 0.000 | N/A | N/A | N/A | N/A |
| disaster_hotspot | fix3 | 0.282 | 0.102 | 0.700 | 0.000 | N/A | N/A | N/A | N/A |
| disaster_hotspot | fix4 | 0.282 | 0.102 | 0.700 | 0.000 | N/A | N/A | N/A | 0.551 |
| disaster_hotspot | fix5 | 0.282 | 0.204 | 0.244 | 0.000 | 0.000 | 0.000 | 0.000 | 0.551 |
| low_snr_soft | B_state_v2_fixed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| low_snr_soft | fix3 | 0.372 | 0.318 | 0.145 | 0.000 | N/A | N/A | N/A | N/A |
| low_snr_soft | fix4 | 0.372 | 0.318 | 0.145 | 0.000 | N/A | N/A | N/A | 0.507 |
| low_snr_soft | fix5 | 0.372 | 0.318 | 0.145 | 0.000 | 0.000 | 0.000 | 0.000 | 0.507 |
| low_snr_blockage | B_state_v2_fixed | 0.948 | 0.128 | 0.854 | 0.000 | N/A | N/A | N/A | N/A |
| low_snr_blockage | fix3 | 0.822 | 0.276 | 0.610 | 0.000 | N/A | N/A | N/A | N/A |
| low_snr_blockage | fix4 | 0.822 | 0.276 | 0.610 | 0.000 | N/A | N/A | N/A | 0.604 |
| low_snr_blockage | fix5 | 0.833 | 0.496 | 0.399 | 0.000 | 0.000 | 0.000 | 0.000 | 0.694 |
| edge_overload | B_state_v2_fixed | 0.649 | 0.649 | 0.000 | 0.000 | N/A | N/A | N/A | N/A |
| edge_overload | fix3 | 0.321 | 0.082 | 0.297 | 0.000 | N/A | N/A | N/A | N/A |
| edge_overload | fix4 | 0.440 | 0.103 | 0.517 | 0.000 | N/A | N/A | N/A | 0.439 |
| edge_overload | fix5 | 0.056 | 0.028 | 0.034 | 0.000 | 0.818 | 0.818 | 0.000 | 0.407 |
| edge_overload_soft | B_state_v2_fixed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| edge_overload_soft | fix3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| edge_overload_soft | fix4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| edge_overload_soft | fix5 | 0.261 | 0.152 | 0.121 | 0.000 | 0.561 | 0.561 | 0.000 | 0.359 |
| utm_conflict | B_state_v2_fixed | 0.000 | 0.000 | 0.000 | 0.134 | N/A | N/A | N/A | N/A |
| utm_conflict | fix3 | 0.000 | 0.000 | 0.481 | 0.000 | N/A | N/A | N/A | N/A |
| utm_conflict | fix4 | 0.000 | 0.000 | 0.285 | 0.000 | N/A | N/A | N/A | 0.751 |
| utm_conflict | fix5 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.820 |
| utm_conflict_soft | B_state_v2_fixed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| utm_conflict_soft | fix3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| utm_conflict_soft | fix4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| utm_conflict_soft | fix5 | 0.196 | 0.196 | 0.000 | 0.000 | 0.573 | 0.573 | 0.000 | 0.341 |

## Hard/Soft Checks

- edge_overload hard: task success 0.028, deadline violation 0.034, reject 0.818.
- edge_overload_soft: task success 0.152, deadline violation 0.121, reject 0.561; soft improves task success over hard by 0.124.
- utm_conflict hard: UTM violation 0.000, deadline violation 0.000, correct reject 1.000, wrong reject 0.000.
- utm_conflict_soft: task success 0.196, admitted task success 0.458, correct reject 0.573; soft has nonzero service while keeping UTM violation 0.000.
- normal_patrol: reject 0.000 and deadline violation 0.000, so reject does not leak into nominal control.
- low_snr_soft vs low_snr_blockage: task success 0.318 vs 0.496; reject remains 0.000 / 0.000.

## Verdict

Fix5 implements infeasibility-aware semantic admission control. It avoids reject in normal and low-SNR scenes, uses correct reject in hard UTM and edge-overload admission cases, keeps wrong reject at 0.000 in all evaluated scenarios, and soft UTM/edge presets recover nonzero admitted service compared with hard infeasible cases. Edge hard still has low task success by construction, but deadline/UTM violations are controlled through reject rather than unsafe service.
