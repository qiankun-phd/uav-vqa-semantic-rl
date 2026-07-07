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
| edge_overload | 0.056 | 0.000 | 0.021 | 0.161 | 0.000 | 0.465 | 0.538 |
| edge_overload_soft | 0.261 | 0.000 | 0.003 | 0.436 | 0.000 | 0.349 | 0.324 |
| low_snr_blockage | 0.833 | 0.344 | 0.510 | 0.490 | 0.000 | 0.272 | 0.056 |
| low_snr_soft | 0.372 | 0.079 | 0.349 | 0.651 | 0.000 | 0.215 | 0.184 |
| normal_patrol | 0.283 | 0.000 | 0.397 | 0.603 | 0.000 | 0.347 | 0.170 |
| utm_conflict | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.433 | 0.786 |
| utm_conflict_soft | 0.196 | 0.000 | 0.055 | 0.373 | 0.000 | 0.395 | 0.352 |

- raw seed rows: 144
- `scenario_comparison_summary.csv` reports mean/std across seeds.
