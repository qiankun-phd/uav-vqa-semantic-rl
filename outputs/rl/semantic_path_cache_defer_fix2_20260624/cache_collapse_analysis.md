# Scenario Benchmark Cache-Collapse Analysis

## Diagnosis

- Previous smoke PPO had high cache ratios because cache minimized delay/energy/payload while the semantic shortfall penalty was too small.
- The semantic-path controller persists `epsilon_k`, uses risk/staleness/UTM-aware cache shortfall penalties, exposes defer/cache_update ratios, and keeps a stronger semantic-token/cache-update prior.
- Compute-aware projection prefers semantic tokens over cache when token evidence reduces LCB shortfall under edge/deadline pressure; UTM conflicts are recorded as risk/queue costs instead of being hidden by cache fallback.
- Cache actions are projected to zero bandwidth/power/cpu/gpu; token/image actions retain service-dependent resource floors.

## Proposed PPO vs Always Cache

| scenario | proposed semantic success | cache semantic success | proposed cache mix | proposed token mix | proposed image mix | cache gap | proposed gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | 0.278 | 0.124 | 0.602 | 0.398 | 0.000 | 0.385 | 0.189 |
| edge_overload | 0.499 | 0.000 | 0.208 | 0.792 | 0.000 | 0.465 | 0.136 |
| low_snr_blockage | 0.825 | 0.344 | 0.429 | 0.571 | 0.000 | 0.272 | 0.072 |
| low_snr_soft | 0.287 | 0.079 | 0.710 | 0.290 | 0.000 | 0.215 | 0.343 |
| normal_patrol | 0.251 | 0.000 | 0.639 | 0.361 | 0.000 | 0.347 | 0.396 |
| utm_conflict | 0.000 | 0.000 | 0.641 | 0.359 | 0.000 | 0.433 | 0.541 |

- raw seed rows: 108
- `scenario_comparison_summary.csv` reports mean/std across seeds.
