# Scenario Benchmark Cache-Collapse Analysis

## Diagnosis

- Previous smoke PPO had high cache ratios because cache minimized delay/energy/payload while the semantic shortfall penalty was too small.
- The v3 controller persists `epsilon_k`, uses risk/staleness/UTM-aware cache shortfall penalties, distills semantic-greedy routing, and keeps a stronger semantic-token prior.
- Compute-aware projection prefers semantic tokens over cache when token evidence reduces LCB shortfall under edge/deadline pressure; UTM conflicts are recorded as risk/queue costs instead of being hidden by cache fallback.
- Cache actions are projected to zero bandwidth/power/cpu/gpu; token/image actions retain service-dependent resource floors.

## Proposed PPO vs Always Cache

| scenario | proposed semantic success | cache semantic success | proposed cache mix | proposed token mix | proposed image mix | cache gap | proposed gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | 0.228 | 0.015 | 0.089 | 0.911 | 0.000 | 0.576 | 0.274 |
| edge_overload | 0.010 | 0.000 | 0.000 | 1.000 | 0.000 | 0.468 | 0.222 |
| low_snr_blockage | 0.786 | 0.053 | 0.247 | 0.727 | 0.027 | 0.465 | 0.099 |
| nominal_patrol | 0.283 | 0.000 | 0.094 | 0.793 | 0.113 | 0.450 | 0.182 |
| utm_conflict | 0.000 | 0.000 | 0.087 | 0.913 | 0.000 | 0.507 | 0.195 |

- raw seed rows: 135
- `scenario_comparison_summary.csv` reports mean/std across seeds.
