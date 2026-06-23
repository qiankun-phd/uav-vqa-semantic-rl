# Scenario Benchmark Cache-Collapse Analysis

## Diagnosis

- Previous smoke PPO had high cache ratios because cache minimized delay/energy/payload while the semantic shortfall penalty was too small.
- The v2 controller uses semantic-only success bonus, stronger LCB/margin reward, explicit cache shortfall penalty, high-epsilon/high-risk cache penalty, oracle warm-start, service-level curriculum, and token exploration bonus.
- Cache actions are projected to zero bandwidth/power/cpu/gpu; token/image actions retain service-dependent resource floors.

## Proposed PPO vs Always Cache

| scenario | proposed semantic success | cache semantic success | proposed cache mix | proposed token mix | proposed image mix | cache gap | proposed gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | 0.238 | 0.015 | 0.185 | 0.799 | 0.017 | 0.576 | 0.267 |
| edge_overload | 0.000 | 0.000 | 0.309 | 0.691 | 0.000 | 0.545 | 0.121 |
| low_snr_blockage | 0.785 | 0.053 | 0.276 | 0.697 | 0.027 | 0.465 | 0.108 |
| nominal_patrol | 0.303 | 0.000 | 0.367 | 0.451 | 0.182 | 0.450 | 0.186 |
| utm_conflict | 0.000 | 0.000 | 0.420 | 0.580 | 0.000 | 0.507 | 0.239 |

- raw seed rows: 135
- `scenario_comparison_summary.csv` reports mean/std across seeds.
