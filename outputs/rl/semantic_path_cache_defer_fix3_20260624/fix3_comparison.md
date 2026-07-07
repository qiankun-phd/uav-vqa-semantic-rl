# Semantic Path Cache/Defer Fix3 Comparison

## Run

- output: `outputs/rl/semantic_path_cache_defer_fix3_20260624/`
- PPO variant: `semantic_path_two_timescale_ppo`
- train episodes: 300
- eval episodes: 50
- seeds: 0,1,2
- tasks per episode: 12
- device: cuda:0 (NVIDIA GeForce RTX 4060)

## Fix3 Summary

| scenario | semantic success | task success | deadline vio | UTM vio | payload KB | cache/token/image/defer/update | joint feasible | deadline infeasible | UTM infeasible | expert agreement | BC loss | defer count |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| normal_patrol | 0.274 | 0.274 | 0.000 | 0.000 | 0.546 | 0.436/0.564/0.000/0.000/0.000 | 0.066 | 0.530 | 0.000 | 0.563 | 0.002 | 0.000 |
| disaster_hotspot | 0.282 | 0.102 | 0.700 | 0.000 | 0.488 | 0.582/0.418/0.000/0.000/0.000 | 0.071 | 0.418 | 0.000 | 0.409 | 0.000 | 0.000 |
| low_snr_soft | 0.372 | 0.318 | 0.145 | 0.000 | 0.601 | 0.347/0.651/0.000/0.002/0.000 | 0.197 | 0.544 | 0.000 | 0.644 | 0.103 | 0.005 |
| low_snr_blockage | 0.822 | 0.276 | 0.610 | 0.000 | 0.564 | 0.427/0.573/0.000/0.000/0.000 | 0.047 | 0.591 | 0.000 | 0.393 | 0.016 | 0.000 |
| edge_overload | 0.321 | 0.082 | 0.297 | 0.000 | 0.421 | 0.097/0.477/0.000/0.426/0.000 | 0.131 | 0.795 | 0.000 | 0.476 | 0.016 | 1.035 |
| utm_conflict | 0.000 | 0.000 | 0.481 | 0.000 | 0.450 | 0.621/0.379/0.000/0.000/0.000 | 0.004 | 0.417 | 0.000 | 0.548 | 0.106 | 0.000 |

## Fix2 To Fix3 Delta

| scenario | task success delta | deadline vio delta | UTM vio delta | defer ratio delta | cache update delta | deadline infeasible delta |
|---|---:|---:|---:|---:|---:|---:|
| normal_patrol | +0.176 | -0.212 | +0.000 | -0.454 | -0.005 | -0.269 |
| disaster_hotspot | +0.002 | -0.009 | +0.000 | +0.000 | +0.000 | +0.020 |
| low_snr_soft | +0.101 | +0.054 | +0.000 | -0.533 | +0.000 | -0.180 |
| low_snr_blockage | +0.000 | +0.002 | +0.000 | +0.000 | +0.000 | +0.002 |
| edge_overload | +0.001 | -0.360 | +0.000 | +0.426 | -0.019 | +0.084 |
| utm_conflict | +0.000 | +0.169 | +0.000 | -0.637 | +0.000 | -0.579 |

## Judgment Against Fix3 Criteria

- MISS: normal_patrol task success >= 0.275 and deadline near 0
- PASS: low_snr_soft task success improves over fix2 without large deadline regression
- PASS: edge_overload improves over fix2 and cache_update < 0.05
- PASS: utm_conflict UTM violation remains near zero
- PASS: no always-defer collapse overall

## Interpretation

- Fix3 removes the fix2 over-defer failure in normal/low-SNR/UTM paths by using the candidate-path expert as BC target and logit bias. Defer is now mostly a last resort except edge_overload, where the expert also sees few feasible service paths.
- UTM safety is preserved: PPO selects cache/token with UTM infeasible selection near zero and UTM violation near zero in `utm_conflict`, but task success remains zero because candidate paths still do not satisfy semantic success under the UTM-constrained preset.
- Edge overload is safer than fix2 in deadline terms and cache_update remains below the 0.05 target, but task success is still low and defer remains high. This suggests the bottleneck is feasible token/cache service availability under high edge load, not cache_update overuse.
- Low-SNR scenarios keep high semantic success without image overuse, but deadline violation remains substantial for token service. This should be handled by the separate low-SNR deadline/resource tuning line rather than more defer penalties.

## Prior Comparison Sources

- `outputs/rl/state_v2_fixed_medium_20260624/medium_summary.md` for B_state_v2_fixed_128x128 reference.
- `outputs/rl/semantic_path_cache_defer_short_20260624/scenario_comparison_summary.csv`.
- `outputs/rl/semantic_path_cache_defer_fix1_20260624/scenario_comparison_summary.csv`.
- `outputs/rl/semantic_path_cache_defer_fix2_20260624/scenario_comparison_summary.csv`.
