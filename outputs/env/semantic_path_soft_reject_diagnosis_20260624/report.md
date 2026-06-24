# Semantic Path Soft Scenario and Reject Diagnosis

Environment-only diagnosis. PPO training and Semantic Utility LUT are not modified.

## Path Feasibility Summary

| scenario | path | joint | semantic | deadline | UTM | reject feasible | oracle success | bottlenecks/reasons | avg tx | avg queue | avg infer | avg arrival | req. rate | req. bandwidth |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| disaster_hotspot | _oracle_action | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | :1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| disaster_hotspot | cache | 0.000 | 0.000 | 0.200 | 1.000 | 0.000 | 0.000 | expired:0.800;semantic_quality:0.200 | 0.000 | 0.063 | 0.004 | 0.000 | 0.000 | 810000.0 |
| disaster_hotspot | cache_update | 0.000 | 0.033 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.800;mobility:0.033;semantic_quality:0.167 | 0.000 | 0.063 | 0.054 | 0.691 | 1880802.400 | 122974923935.3 |
| disaster_hotspot | defer | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.800;tx_delay:0.200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| disaster_hotspot | image | 0.000 | 0.033 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.800;mobility:0.033;semantic_quality:0.167 | 0.021 | 0.063 | 0.451 | 0.691 | 310351176.533 | 20293516945175.7 |
| disaster_hotspot | reject | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | expired:0.800;mixed:0.200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| disaster_hotspot | token | 0.000 | 0.033 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.800;mobility:0.033;semantic_quality:0.167 | 0.000 | 0.063 | 0.054 | 0.691 | 1880802.400 | 122974923935.3 |
| edge_overload | _oracle_action | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | :1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| edge_overload | cache | 0.000 | 0.000 | 0.528 | 1.000 | 0.000 | 0.000 | expired:0.389;semantic_quality:0.611 | 0.000 | 0.467 | 0.012 | 0.000 | 0.000 | 419444.4 |
| edge_overload | cache_update | 0.000 | 0.250 | 0.250 | 1.000 | 0.000 | 0.000 | expired:0.389;mobility:0.250;semantic_quality:0.361 | 0.001 | 0.467 | 0.317 | 2.854 | 2489586.668 | 192763633207.3 |
| edge_overload | defer | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.389;tx_delay:0.611 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| edge_overload | image | 0.000 | 0.111 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.389;resource:0.111;semantic_quality:0.500 | 0.061 | 0.467 | 1.557 | 2.854 | 776660371.111 | 59104353636139.4 |
| edge_overload | reject | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | expired:0.389;mixed:0.611 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| edge_overload | token | 0.000 | 0.250 | 0.250 | 1.000 | 0.000 | 0.000 | expired:0.389;mobility:0.250;semantic_quality:0.361 | 0.001 | 0.467 | 0.317 | 2.854 | 2489586.668 | 192763633207.3 |
| edge_overload_soft | _oracle_action | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.306 | :1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| edge_overload_soft | cache | 0.194 | 0.194 | 0.833 | 1.000 | 0.000 | 0.194 | expired:0.167;none:0.194;semantic_quality:0.639 | 0.000 | 0.189 | 0.017 | 0.000 | 0.000 | 208333.3 |
| edge_overload_soft | cache_update | 0.111 | 0.528 | 0.417 | 1.000 | 0.000 | 0.111 | expired:0.167;mobility:0.417;none:0.111;semantic_quality:0.306 | 0.001 | 0.189 | 0.499 | 3.042 | 2545691.113 | 177914777061.7 |
| edge_overload_soft | defer | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.167;tx_delay:0.833 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| edge_overload_soft | image | 0.028 | 0.444 | 0.083 | 1.000 | 0.000 | 0.028 | expired:0.167;mobility:0.333;none:0.028;resource:0.083;semantic_quality:0.389 | 0.080 | 0.189 | 2.201 | 3.042 | 928399699.917 | 66499049026947.8 |
| edge_overload_soft | reject | 0.694 | 0.000 | 1.000 | 1.000 | 0.694 | 0.000 | deadline:0.083;expired:0.167;mixed:0.667;semantic_quality:0.083 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| edge_overload_soft | token | 0.111 | 0.528 | 0.417 | 1.000 | 0.000 | 0.111 | expired:0.167;mobility:0.417;none:0.111;semantic_quality:0.306 | 0.001 | 0.189 | 0.499 | 3.042 | 2545691.113 | 177914777061.7 |
| low_snr_blockage | _oracle_action | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.111 | :1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| low_snr_blockage | cache | 0.111 | 0.111 | 0.444 | 1.000 | 0.000 | 0.111 | expired:0.556;none:0.111;semantic_quality:0.333 | 0.000 | 0.053 | 0.009 | 0.000 | 0.000 | 375555.6 |
| low_snr_blockage | cache_update | 0.000 | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.556;mobility:0.250;semantic_quality:0.194 | 1.015 | 0.053 | 0.136 | 9.654 | 3585284.444 | 630186129290756.1 |
| low_snr_blockage | defer | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.556;tx_delay:0.444 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 650000.0 |
| low_snr_blockage | image | 0.000 | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.556;mobility:0.056;semantic_quality:0.194;tx_delay:0.194 | 36.378 | 0.053 | 1.020 | 9.654 | 132193390.222 | 23092974470044424.0 |
| low_snr_blockage | reject | 0.889 | 0.000 | 1.000 | 1.000 | 0.889 | 0.000 | deadline:0.083;expired:0.556;mixed:0.361 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| low_snr_blockage | token | 0.000 | 0.250 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.556;mobility:0.250;semantic_quality:0.194 | 1.015 | 0.053 | 0.136 | 9.654 | 3585284.444 | 630186129290756.1 |
| low_snr_soft | _oracle_action | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 | :1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| low_snr_soft | cache | 0.333 | 0.333 | 0.833 | 1.000 | 0.000 | 0.333 | expired:0.167;none:0.333;semantic_quality:0.500 | 0.000 | 0.076 | 0.017 | 0.000 | 0.000 | 208333.3 |
| low_snr_soft | cache_update | 0.167 | 0.556 | 0.222 | 1.000 | 0.000 | 0.167 | expired:0.167;mobility:0.389;none:0.167;semantic_quality:0.278 | 0.007 | 0.076 | 0.453 | 5.179 | 3937763.718 | 3133555878976.6 |
| low_snr_soft | defer | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.167;tx_delay:0.833 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| low_snr_soft | image | 0.000 | 0.556 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.167;mobility:0.500;resource:0.056;semantic_quality:0.278 | 0.315 | 0.076 | 2.148 | 5.179 | 479732981.778 | 307768064246382.7 |
| low_snr_soft | reject | 0.667 | 0.000 | 1.000 | 1.000 | 0.667 | 0.000 | deadline:0.222;expired:0.167;mixed:0.611 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| low_snr_soft | token | 0.167 | 0.556 | 0.222 | 1.000 | 0.000 | 0.167 | expired:0.167;mobility:0.389;none:0.167;semantic_quality:0.278 | 0.007 | 0.076 | 0.453 | 5.179 | 3937763.718 | 3133555878976.6 |
| normal_patrol | _oracle_action | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | :1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| normal_patrol | cache | 0.000 | 0.000 | 0.472 | 1.000 | 0.000 | 0.000 | expired:0.528;semantic_quality:0.472 | 0.000 | 0.092 | 0.009 | 0.000 | 0.000 | 551388.9 |
| normal_patrol | cache_update | 0.000 | 0.167 | 0.028 | 1.000 | 0.000 | 0.000 | expired:0.528;mobility:0.167;semantic_quality:0.306 | 0.000 | 0.092 | 0.159 | 2.504 | 4039060.673 | 282124705915.6 |
| normal_patrol | defer | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.528;tx_delay:0.472 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| normal_patrol | image | 0.000 | 0.167 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.528;mobility:0.167;semantic_quality:0.306 | 0.053 | 0.092 | 1.101 | 2.504 | 735939573.111 | 51379271143823.7 |
| normal_patrol | reject | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | expired:0.528;mixed:0.472 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| normal_patrol | token | 0.000 | 0.167 | 0.028 | 1.000 | 0.000 | 0.000 | expired:0.528;mobility:0.167;semantic_quality:0.306 | 0.000 | 0.092 | 0.159 | 2.504 | 4039060.673 | 282124705915.6 |
| utm_conflict | _oracle_action | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | :1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| utm_conflict | cache | 0.000 | 0.000 | 0.367 | 1.000 | 0.000 | 0.000 | expired:0.600;semantic_quality:0.400 | 0.000 | 0.066 | 0.008 | 0.000 | 0.000 | 620000.0 |
| utm_conflict | cache_update | 0.000 | 0.000 | 0.000 | 0.833 | 0.000 | 0.000 | expired:0.600;semantic_quality:0.400 | 0.000 | 0.066 | 0.109 | 3.968 | 3802403.200 | 379734623972.5 |
| utm_conflict | defer | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.600;tx_delay:0.400 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| utm_conflict | image | 0.000 | 0.000 | 0.000 | 0.833 | 0.000 | 0.000 | expired:0.600;semantic_quality:0.400 | 0.065 | 0.066 | 0.902 | 3.968 | 636578966.400 | 63556599049635.0 |
| utm_conflict | reject | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | expired:0.600;mixed:0.400 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| utm_conflict | token | 0.000 | 0.000 | 0.000 | 0.833 | 0.000 | 0.000 | expired:0.600;semantic_quality:0.400 | 0.000 | 0.066 | 0.109 | 3.968 | 3802403.200 | 379734623972.5 |
| utm_conflict_soft | _oracle_action | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 | :1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| utm_conflict_soft | cache | 0.333 | 0.333 | 0.583 | 1.000 | 0.000 | 0.333 | expired:0.417;none:0.333;semantic_quality:0.250 | 0.000 | 0.049 | 0.012 | 0.000 | 0.000 | 445833.3 |
| utm_conflict_soft | cache_update | 0.000 | 0.167 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.417;mobility:0.167;semantic_quality:0.417 | 0.001 | 0.049 | 0.249 | 6.111 | 4881863.556 | 484530994504.4 |
| utm_conflict_soft | defer | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.417;tx_delay:0.583 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000000.0 |
| utm_conflict_soft | image | 0.000 | 0.139 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.417;mobility:0.139;semantic_quality:0.444 | 0.077 | 0.049 | 1.423 | 6.111 | 752669319.111 | 74812081350278.1 |
| utm_conflict_soft | reject | 0.667 | 0.000 | 1.000 | 1.000 | 0.667 | 0.000 | deadline:0.028;expired:0.417;mixed:0.556 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| utm_conflict_soft | token | 0.000 | 0.167 | 0.000 | 1.000 | 0.000 | 0.000 | expired:0.417;mobility:0.167;semantic_quality:0.417 | 0.001 | 0.049 | 0.249 | 6.111 | 4881863.556 | 484530994504.4 |

## Scenario Diagnosis

### edge_overload

- Token joint feasible ratio is 0.000; deadline feasible ratio is 0.250.
- Token average queue/infer/load delays are 0.467/0.317/0.049 s; bottlenecks: expired:0.389;mobility:0.250;semantic_quality:0.361.
- When token misses deadline, average required rate/bandwidth estimates are 2489586.668 Mbps and 192763633207.3 Hz.
- Cache-update uses token evidence but adds cache refresh semantics; its joint feasible ratio is 0.000, so it should not be selected only from semantic quality.
- Diagnosis: current edge-overload preset is nearly infeasible under the physical queue/model-load parameters.

### edge_overload_soft


### utm_conflict

- Stay/cache is UTM-safe by construction, but cache semantic feasibility is 0.000; low cache quality explains zero task success when cache dominates.
- Token avoid_conflict UTM risk 0.000 vs serve_task 0.009; avoid_conflict exposes a safer candidate if lower.
- Token with avoid_conflict deadline feasible ratio is 0.133, semantic feasible ratio is 0.000, joint feasible ratio is 0.000.
- Diagnosis: UTM-safe actions exist, but current service candidates are semantically or deadline infeasible; this is not only a UTM violation issue.

### utm_conflict_soft



## Soft vs Hard Comparison

- `edge_overload_soft` vs `edge_overload`: best service joint feasibility 0.194 vs 0.000; reject feasible ratio 0.694 vs 1.000.
- `utm_conflict_soft` vs `utm_conflict`: best service joint feasibility 0.333 vs 0.000; reject feasible ratio 0.667 vs 1.000.

## Calibrated Scenario Suggestions

- If `edge_overload` oracle feasible ratios remain near zero, keep it as a hard stress scenario and add a separate `edge_overload_soft` preset instead of weakening the hard preset.
- If `utm_conflict` remains semantic-infeasible after UTM-safe mobility, add safer non-overlapping task subsets or lower cache/task epsilon for a soft UTM scenario; do not relax hard UTM buffers silently.
