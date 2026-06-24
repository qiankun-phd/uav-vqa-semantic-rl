# Scenario Benchmark Cache-Collapse Analysis

## Diagnosis

- Previous smoke PPO had high cache ratios because cache minimized delay/energy/payload while the semantic shortfall penalty was too small.
- The semantic-path controller persists `epsilon_k`, uses risk/staleness/UTM-aware cache shortfall penalties, exposes defer/cache_update ratios, and keeps a stronger semantic-token/cache-update prior.
- Compute-aware projection prefers semantic tokens over cache when token evidence reduces LCB shortfall under edge/deadline pressure; UTM conflicts are recorded as risk/queue costs instead of being hidden by cache fallback.
- Cache actions are projected to zero bandwidth/power/cpu/gpu; token/image actions retain service-dependent resource floors.

## Proposed PPO vs Always Cache

| scenario | proposed semantic success | cache semantic success | proposed cache mix | proposed token mix | proposed image mix | cache gap | proposed gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | 0.280 | 0.124 | 0.418 | 0.431 | 0.151 | 0.385 | 0.200 |
| edge_overload | 0.384 | 0.000 | 0.266 | 0.734 | 0.000 | 0.465 | 0.175 |
| low_snr_blockage | 0.822 | 0.344 | 0.427 | 0.573 | 0.000 | 0.272 | 0.071 |
| low_snr_soft | 0.822 | 0.344 | 0.427 | 0.573 | 0.000 | 0.272 | 0.071 |
| normal_patrol | 0.275 | 0.000 | 0.434 | 0.566 | 0.000 | 0.347 | 0.177 |
| utm_conflict | 0.000 | 0.000 | 0.373 | 0.325 | 0.301 | 0.433 | 0.218 |

- raw seed rows: 108
- `scenario_comparison_summary.csv` reports mean/std across seeds.
