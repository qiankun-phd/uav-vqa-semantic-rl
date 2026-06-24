# Semantic Path Cache/Defer Fix4 Comparison

## Run

- output: `outputs/rl/semantic_path_cache_defer_fix4_20260624/`
- PPO variant: `semantic_path_two_timescale_ppo`
- train episodes: 300
- eval episodes: 50
- seeds: 0,1,2
- tasks per episode: 12
- device: cuda:0 (NVIDIA GeForce RTX 4060)
- Environment diagnosis commit: `448813c analysis(env): diagnose semantic path bottlenecks`

## Fix4 Summary

| scenario | semantic success | task success | deadline vio | UTM vio | oracle infeasible | UTM-safe service | mobility avoid/stay | bottleneck semantic/queue/tx/mobility | path cache/token/image/defer/update |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| normal_patrol | 0.275 | 0.275 | 0.000 | 0.000 | 0.675 | 1.000 | 0.000/1.000 | 0.536/0.000/0.000/0.415 | 0.389/0.611/0.000/0.000/0.000 |
| disaster_hotspot | 0.282 | 0.102 | 0.700 | 0.000 | 0.551 | 1.000 | 1.000/0.000 | 0.551/0.000/0.000/0.378 | 0.582/0.418/0.000/0.000/0.000 |
| low_snr_soft | 0.372 | 0.318 | 0.145 | 0.000 | 0.507 | 1.000 | 0.000/1.000 | 0.486/0.000/0.000/0.317 | 0.347/0.651/0.000/0.002/0.000 |
| low_snr_blockage | 0.822 | 0.276 | 0.610 | 0.000 | 0.604 | 1.000 | 0.005/0.995 | 0.383/0.002/0.000/0.568 | 0.427/0.573/0.000/0.000/0.000 |
| edge_overload | 0.440 | 0.103 | 0.517 | 0.000 | 0.439 | 1.000 | 0.543/0.457 | 0.297/0.001/0.000/0.528 | 0.344/0.656/0.000/0.000/0.000 |
| utm_conflict | 0.000 | 0.000 | 0.285 | 0.000 | 0.751 | 1.000 | 0.363/0.637 | 0.653/0.008/0.000/0.333 | 0.667/0.333/0.000/0.000/0.000 |

## Fix3 To Fix4 Delta

| scenario | task success delta | semantic success delta | deadline vio delta | UTM vio delta | defer ratio delta | cache update delta | deadline infeasible delta | oracle infeasible fix4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_patrol | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.047 | 0.675 |
| disaster_hotspot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.551 |
| low_snr_soft | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.507 |
| low_snr_blockage | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.604 |
| edge_overload | 0.021 | 0.119 | 0.221 | 0.000 | -0.426 | 0.000 | -0.259 | 0.439 |
| utm_conflict | 0.000 | 0.000 | -0.196 | 0.000 | 0.000 | 0.000 | -0.056 | 0.751 |

## Judgment Against Fix4 Criteria

- PASS: normal_patrol not below fix3
- PASS: low_snr_soft not below fix3
- MISS: edge_overload deadline lower than fix3 and task not lower
- PASS: utm_conflict UTM violation remains zero; zero success explained by infeasibility
- PASS: no always-defer/cache collapse

## Interpretation

- Fix4 adds bottleneck-aware path targets and mobility-aware expert supervision. The learned policy stays stable on `normal_patrol` and keeps UTM conflict violations at zero.
- `low_snr_soft` remains improved relative to the over-defer fix2 line and has a different profile from `low_snr_blockage`; the soft preset is easier, with lower deadline violation and higher task success.
- `edge_overload` remains a hard near-infeasible scenario. Fix4 removes cache-update overuse and selects token/cache, but deadline violation rises versus fix3 because the policy attempts more service while the oracle infeasible ratio is high and mobility/semantic bottlenecks dominate.
- `utm_conflict` task success remains zero while UTM violation stays zero. The environment diagnosis and fix4 oracle infeasible ratio show the UTM-safe service candidates are mostly semantically/deadline infeasible, so this is an infeasibility-aware failure rather than unsafe routing.
- Next algorithm step should not add more scalar penalties. It should either train on calibrated soft variants (`edge_overload_soft`, `utm_conflict_soft`) or add an explicit skip/fail action with value-aware rejection for hard infeasible tasks.

## Prior Comparison Sources

- `outputs/rl/state_v2_fixed_medium_20260624/medium_summary.md` for B_state_v2_fixed_128x128 reference.
- `outputs/rl/semantic_path_cache_defer_fix1_20260624/scenario_comparison_summary.csv`.
- `outputs/rl/semantic_path_cache_defer_fix2_20260624/scenario_comparison_summary.csv`.
- `outputs/rl/semantic_path_cache_defer_fix3_20260624/scenario_comparison_summary.csv`.
- `outputs/env/semantic_path_bottleneck_diagnosis_20260624/report.md` for oracle bottleneck diagnosis.
