# Semantic Path Bottleneck Diagnosis

Environment-only diagnosis. PPO training and Semantic Utility LUT are not modified.

## Path Feasibility Summary

| scenario | path | joint | semantic | deadline | UTM | bottlenecks | avg tx | avg queue | avg infer | avg arrival | req. rate | req. bandwidth |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | cache | 0.000 | 0.000 | 1.000 | 1.000 | semantic_quality:1.000 | 0.000 | 0.226 | 0.020 | 0.000 | 0.000 | 50000.0 |
| disaster_hotspot | cache_update | 0.000 | 0.111 | 0.000 | 1.000 | mobility:0.111;semantic_quality:0.889 | 0.001 | 0.226 | 0.272 | 3.458 | 9411325.333 | 610663948457.1 |
| disaster_hotspot | defer | 0.000 | 1.000 | 0.000 | 1.000 | tx_delay:1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| disaster_hotspot | image | 0.000 | 0.111 | 0.000 | 1.000 | mobility:0.111;semantic_quality:0.889 | 0.103 | 0.226 | 2.255 | 3.458 | 1551905745.778 | 100685105932740.3 |
| disaster_hotspot | token | 0.000 | 0.111 | 0.000 | 1.000 | mobility:0.111;semantic_quality:0.889 | 0.001 | 0.226 | 0.272 | 3.458 | 9411325.333 | 610663948457.1 |
| edge_overload | cache | 0.000 | 0.000 | 1.000 | 1.000 | semantic_quality:1.000 | 0.000 | 0.284 | 0.020 | 0.000 | 0.000 | 50000.0 |
| edge_overload | cache_update | 0.000 | 0.056 | 0.278 | 1.000 | mobility:0.056;semantic_quality:0.944 | 0.001 | 0.284 | 0.302 | 3.960 | 6510486.223 | 498970846926.6 |
| edge_overload | defer | 0.000 | 1.000 | 0.000 | 1.000 | tx_delay:1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| edge_overload | image | 0.000 | 0.028 | 0.000 | 1.000 | resource:0.028;semantic_quality:0.972 | 0.097 | 0.284 | 2.291 | 3.960 | 1291752443.778 | 94354960216641.2 |
| edge_overload | token | 0.000 | 0.056 | 0.278 | 1.000 | mobility:0.056;semantic_quality:0.944 | 0.001 | 0.284 | 0.302 | 3.960 | 6510486.223 | 498970846926.6 |
| low_snr_blockage | cache | 0.111 | 0.111 | 1.000 | 1.000 | none:0.111;semantic_quality:0.889 | 0.000 | 0.066 | 0.020 | 0.000 | 0.000 | 32500.0 |
| low_snr_blockage | cache_update | 0.000 | 0.444 | 0.000 | 1.000 | mobility:0.444;semantic_quality:0.556 | 2.896 | 0.066 | 0.292 | 21.542 | 8133515.556 | 1821894270798209.0 |
| low_snr_blockage | defer | 0.000 | 1.000 | 0.000 | 1.000 | tx_delay:1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 650000.0 |
| low_snr_blockage | image | 0.000 | 0.444 | 0.000 | 1.000 | mobility:0.074;semantic_quality:0.556;tx_delay:0.370 | 104.737 | 0.066 | 2.279 | 21.542 | 297449762.963 | 66486974634468504.0 |
| low_snr_blockage | token | 0.000 | 0.444 | 0.000 | 1.000 | mobility:0.444;semantic_quality:0.556 | 2.896 | 0.066 | 0.292 | 21.542 | 8133515.556 | 1821894270798209.0 |
| low_snr_soft | cache | 0.167 | 0.167 | 1.000 | 1.000 | none:0.167;semantic_quality:0.833 | 0.000 | 0.094 | 0.020 | 0.000 | 0.000 | 50000.0 |
| low_snr_soft | cache_update | 0.083 | 0.417 | 0.306 | 1.000 | mobility:0.333;none:0.083;semantic_quality:0.583 | 0.009 | 0.094 | 0.272 | 6.924 | 5769213.497 | 7216690978350.8 |
| low_snr_soft | defer | 0.000 | 1.000 | 0.000 | 1.000 | tx_delay:1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| low_snr_soft | image | 0.000 | 0.417 | 0.000 | 1.000 | mobility:0.417;semantic_quality:0.583 | 0.473 | 0.094 | 2.255 | 6.924 | 636901591.778 | 462315063466709.4 |
| low_snr_soft | token | 0.083 | 0.417 | 0.306 | 1.000 | mobility:0.333;none:0.083;semantic_quality:0.583 | 0.009 | 0.094 | 0.272 | 6.924 | 5769213.497 | 7216690978350.8 |
| normal_patrol | cache | 0.000 | 0.000 | 1.000 | 1.000 | semantic_quality:1.000 | 0.000 | 0.112 | 0.020 | 0.000 | 0.000 | 50000.0 |
| normal_patrol | cache_update | 0.000 | 0.433 | 0.233 | 1.000 | mobility:0.433;semantic_quality:0.567 | 0.001 | 0.112 | 0.399 | 5.460 | 6498462.959 | 488330367691.2 |
| normal_patrol | defer | 0.000 | 1.000 | 0.000 | 1.000 | tx_delay:1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| normal_patrol | image | 0.000 | 0.367 | 0.000 | 1.000 | mobility:0.367;semantic_quality:0.633 | 0.111 | 0.112 | 2.405 | 5.460 | 1516291690.133 | 108839941299986.3 |
| normal_patrol | token | 0.000 | 0.433 | 0.233 | 1.000 | mobility:0.433;semantic_quality:0.567 | 0.001 | 0.112 | 0.399 | 5.460 | 6498462.959 | 488330367691.2 |
| utm_conflict | cache | 0.000 | 0.000 | 1.000 | 1.000 | semantic_quality:1.000 | 0.000 | 0.092 | 0.020 | 0.000 | 0.000 | 50000.0 |
| utm_conflict | cache_update | 0.000 | 0.000 | 0.000 | 0.733 | semantic_quality:1.000 | 0.001 | 0.092 | 0.272 | 9.910 | 9505928.000 | 954149936709.0 |
| utm_conflict | defer | 0.000 | 1.000 | 0.000 | 1.000 | tx_delay:1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| utm_conflict | image | 0.000 | 0.000 | 0.000 | 0.733 | semantic_quality:1.000 | 0.164 | 0.092 | 2.255 | 9.910 | 1595627524.267 | 160155904897169.8 |
| utm_conflict | token | 0.000 | 0.000 | 0.000 | 0.733 | semantic_quality:1.000 | 0.001 | 0.092 | 0.272 | 9.910 | 9505928.000 | 954149936709.0 |

## Scenario Diagnosis

### edge_overload

- Token joint feasible ratio is 0.000; deadline feasible ratio is 0.278.
- Token average queue/infer/load delays are 0.284/0.302/0.080 s; bottlenecks: mobility:0.056;semantic_quality:0.944.
- When token misses deadline, average required rate/bandwidth estimates are 6510486.223 Mbps and 498970846926.6 Hz.
- Cache-update uses token evidence but adds cache refresh semantics; its joint feasible ratio is 0.000, so it should not be selected only from semantic quality.
- Diagnosis: current edge-overload preset is nearly infeasible under the physical queue/model-load parameters.

### utm_conflict

- Stay/cache is UTM-safe by construction, but cache semantic feasibility is 0.000; low cache quality explains zero task success when cache dominates.
- Token avoid_conflict UTM risk 0.000 vs serve_task 0.018; avoid_conflict exposes a safer candidate if lower.
- Token with avoid_conflict deadline feasible ratio is 0.200, semantic feasible ratio is 0.000, joint feasible ratio is 0.000.
- Diagnosis: UTM-safe actions exist, but current service candidates are semantically or deadline infeasible; this is not only a UTM violation issue.


## Calibrated Scenario Suggestions

- If `edge_overload` oracle feasible ratios remain near zero, keep it as a hard stress scenario and add a separate `edge_overload_soft` preset instead of weakening the hard preset.
- If `utm_conflict` remains semantic-infeasible after UTM-safe mobility, add safer non-overlapping task subsets or lower cache/task epsilon for a soft UTM scenario; do not relax hard UTM buffers silently.
