# Semantic Scenario Benchmark v3 vs v4 Delta

v4 reruns the same policy set after environment commit 8f903c2 (scenario feasibility calibration). No LUT or major algorithm changes were made in this v4 run.

## Proposed PPO Delta

| scenario | metric | v3 | v4 | delta |
|---|---:|---:|---:|---:|
| nominal_patrol | semantic success | 0.283 | 0.283 | +0.000 |
| nominal_patrol | task success | 0.098 | 0.098 | +0.000 |
| nominal_patrol | deadline violation | 0.450 | 0.450 | +0.000 |
| nominal_patrol | UTM conflict | 0.000 | 0.000 | +0.000 |
| nominal_patrol | cache | 0.094 | 0.094 | +0.000 |
| nominal_patrol | token | 0.793 | 0.793 | +0.000 |
| nominal_patrol | image | 0.113 | 0.113 | +0.000 |
| disaster_hotspot | semantic success | 0.228 | 0.228 | +0.000 |
| disaster_hotspot | task success | 0.130 | 0.130 | +0.000 |
| disaster_hotspot | deadline violation | 0.291 | 0.291 | +0.000 |
| disaster_hotspot | UTM conflict | 0.000 | 0.000 | +0.000 |
| disaster_hotspot | cache | 0.089 | 0.089 | +0.000 |
| disaster_hotspot | token | 0.911 | 0.911 | +0.000 |
| disaster_hotspot | image | 0.000 | 0.000 | +0.000 |
| low_snr_blockage | semantic success | 0.786 | 0.786 | +0.000 |
| low_snr_blockage | task success | 0.087 | 0.087 | +0.000 |
| low_snr_blockage | deadline violation | 0.729 | 0.729 | +0.000 |
| low_snr_blockage | UTM conflict | 0.000 | 0.000 | +0.000 |
| low_snr_blockage | cache | 0.247 | 0.247 | +0.000 |
| low_snr_blockage | token | 0.727 | 0.727 | +0.000 |
| low_snr_blockage | image | 0.027 | 0.027 | +0.000 |
| edge_overload | semantic success | 0.010 | 0.607 | +0.597 |
| edge_overload | task success | 0.000 | 0.385 | +0.385 |
| edge_overload | deadline violation | 0.902 | 0.381 | -0.521 |
| edge_overload | UTM conflict | 0.000 | 0.000 | +0.000 |
| edge_overload | cache | 0.000 | 0.041 | +0.041 |
| edge_overload | token | 1.000 | 0.959 | -0.041 |
| edge_overload | image | 0.000 | 0.000 | +0.000 |
| utm_conflict | semantic success | 0.000 | 0.000 | +0.000 |
| utm_conflict | task success | 0.000 | 0.000 | +0.000 |
| utm_conflict | deadline violation | 0.371 | 0.843 | +0.472 |
| utm_conflict | UTM conflict | 0.913 | 0.151 | -0.761 |
| utm_conflict | cache | 0.087 | 0.100 | +0.013 |
| utm_conflict | token | 0.913 | 0.900 | -0.013 |
| utm_conflict | image | 0.000 | 0.000 | +0.000 |

## Scenario-Level Interpretation

- nominal_patrol: semantic success 0.283 -> 0.283 (+0.000), task success 0.098 -> 0.098, deadline violation 0.450 -> 0.450, UTM conflict 0.000 -> 0.000; v4 service mix cache/token/image = 0.094/0.793/0.113.
- disaster_hotspot: semantic success 0.228 -> 0.228 (+0.000), task success 0.130 -> 0.130, deadline violation 0.291 -> 0.291, UTM conflict 0.000 -> 0.000; v4 service mix cache/token/image = 0.089/0.911/0.000.
- low_snr_blockage: semantic success 0.786 -> 0.786 (+0.000), task success 0.087 -> 0.087, deadline violation 0.729 -> 0.729, UTM conflict 0.000 -> 0.000; v4 service mix cache/token/image = 0.247/0.727/0.027.
- edge_overload: semantic success 0.010 -> 0.607 (+0.597), task success 0.000 -> 0.385, deadline violation 0.902 -> 0.381, UTM conflict 0.000 -> 0.000; v4 service mix cache/token/image = 0.041/0.959/0.000.
- utm_conflict: semantic success 0.000 -> 0.000 (+0.000), task success 0.000 -> 0.000, deadline violation 0.371 -> 0.843, UTM conflict 0.913 -> 0.151; v4 service mix cache/token/image = 0.100/0.900/0.000.

## Required Checks

- edge_overload proposed PPO semantic success improves from 0.010 to 0.607, and deadline violation drops from 0.902 to 0.381. The controller uses the calibrated token-feasible region rather than cache fallback.
- utm_conflict UTM conflict drops from 0.913 to 0.151, which is now a moderate-pressure regime. Proposed PPO still has 0.000 semantic success because LCB remains below epsilon and deadline pressure is high.
- low_snr_blockage proposed PPO remains stable at 0.786 semantic success versus semantic greedy 0.950; it is close but still below greedy, with much lower image usage.
- Cache collapse does not reappear: v4 proposed cache ratios are edge 0.041, UTM 0.100, low-SNR 0.247.

## Conclusion

The environment calibration fixes the main feasibility pathology for edge_overload: the proposed controller immediately moves into semantic-token routing and gains a large semantic-success improvement without major algorithm changes. For utm_conflict, calibration reduces conflicts to the intended moderate range, but semantic success remains zero; the next algorithm step should target conflict-aware evidence routing and deadline-aware UTM queue control rather than further cache-penalty tuning. The v4 result is therefore a useful benchmark checkpoint, not yet the final paper-scale controller.
