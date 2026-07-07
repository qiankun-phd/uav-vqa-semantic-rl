# Scenario Benchmark Cache-Collapse Analysis

## Diagnosis

- Previous smoke PPO had high cache ratios because cache minimized delay/energy/payload while the semantic shortfall penalty was too small.
- The semantic-path controller persists `epsilon_k`, uses risk/staleness/UTM-aware cache shortfall penalties, exposes defer/cache_update ratios, and keeps a stronger semantic-token/cache-update prior.
- Compute-aware projection prefers semantic tokens over cache when token evidence reduces LCB shortfall under edge/deadline pressure; UTM conflicts are recorded as risk/queue costs instead of being hidden by cache fallback.
- Cache actions are projected to zero bandwidth/power/cpu/gpu; token/image actions retain service-dependent resource floors.

## Proposed PPO vs Always Cache

| scenario | proposed semantic success | cache semantic success | proposed cache mix | proposed token mix | proposed image mix | cache gap | proposed gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | 0.282 | 0.124 | 0.582 | 0.418 | 0.000 | 0.385 | 0.190 |
| edge_overload | 0.440 | 0.000 | 0.344 | 0.656 | 0.000 | 0.465 | 0.152 |
| low_snr_blockage | 0.822 | 0.344 | 0.427 | 0.573 | 0.000 | 0.272 | 0.071 |
| low_snr_soft | 0.372 | 0.079 | 0.349 | 0.651 | 0.000 | 0.215 | 0.184 |
| normal_patrol | 0.275 | 0.000 | 0.389 | 0.611 | 0.000 | 0.347 | 0.177 |
| utm_conflict | 0.000 | 0.000 | 0.667 | 0.333 | 0.000 | 0.433 | 0.224 |

- raw seed rows: 108
- `scenario_comparison_summary.csv` reports mean/std across seeds.
