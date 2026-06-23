# Scenario Benchmark Cache-Collapse Analysis

## Diagnosis

- Previous smoke PPO had high cache ratios because cache minimized delay/energy/payload while the semantic shortfall penalty was too small.
- The v3 controller persists `epsilon_k`, uses risk/staleness/UTM-aware cache shortfall penalties, distills semantic-greedy routing, and keeps a stronger semantic-token prior.
- Compute-aware projection prefers semantic tokens over cache when token evidence reduces LCB shortfall under edge/deadline pressure; UTM conflicts are recorded as risk/queue costs instead of being hidden by cache fallback.
- Cache actions are projected to zero bandwidth/power/cpu/gpu; token/image actions retain service-dependent resource floors.

## Proposed PPO vs Always Cache

| scenario | proposed semantic success | cache semantic success | proposed cache mix | proposed token mix | proposed image mix | cache gap | proposed gap |
|---|---:|---:|---:|---:|---:|---:|---:|

- raw seed rows: 90
- `scenario_comparison_summary.csv` reports mean/std across seeds.
