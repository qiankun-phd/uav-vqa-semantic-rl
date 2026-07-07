# Semantic Path PPO Fix1 Comparison

Compared controllers:
- `B_state_v2_fixed_128x128`: previous medium two-timescale PPO before semantic_path/cache_update/defer.
- `semantic_path_cache_defer_short_20260624`: first semantic-path PPO run, which overused cache_update in edge_overload and had UTM/deadline regressions.
- `semantic_path_cache_defer_fix1_20260624`: feasibility-gated semantic-path PPO in this run.

## Scenario Metrics

| scenario | controller | semantic success | task success | deadline vio | payload KB | cache | token | image | defer | cache_update | UTM vio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_patrol | B_state_v2_fixed_128x128 | 0.156 | 0.156 | 0.000 | 30.742 | n/a | n/a | n/a | n/a | n/a | n/a |
| normal_patrol | semantic_path_short | 0.378 | 0.230 | 0.251 | 0.751 | 0.257 | 0.688 | 0.000 | 0.004 | 0.050 | 0.000 |
| normal_patrol | fix1 | 0.275 | 0.275 | 0.000 | 0.548 | 0.434 | 0.566 | 0.000 | 0.000 | 0.000 | 0.000 |
| disaster_hotspot | B_state_v2_fixed_128x128 | 0.180 | 0.099 | 0.211 | 4.208 | n/a | n/a | n/a | n/a | n/a | n/a |
| disaster_hotspot | semantic_path_short | 0.316 | 0.278 | 0.144 | 0.641 | 0.451 | 0.504 | 0.000 | 0.000 | 0.044 | 0.000 |
| disaster_hotspot | fix1 | 0.280 | 0.196 | 0.336 | 29.512 | 0.418 | 0.431 | 0.151 | 0.000 | 0.000 | 0.000 |
| low_snr_soft | B_state_v2_fixed_128x128 | 0.948 | 0.128 | 0.854 | 0.918 | n/a | n/a | n/a | n/a | n/a | n/a |
| low_snr_soft | semantic_path_short | 0.931 | 0.271 | 0.699 | 0.672 | 0.305 | 0.450 | 0.000 | 0.010 | 0.235 | 0.000 |
| low_snr_soft | fix1 | 0.822 | 0.276 | 0.608 | 0.564 | 0.427 | 0.573 | 0.000 | 0.000 | 0.000 | 0.000 |
| low_snr_blockage | B_state_v2_fixed_128x128 | 0.948 | 0.128 | 0.854 | 0.918 | n/a | n/a | n/a | n/a | n/a | n/a |
| low_snr_blockage | semantic_path_short | 0.931 | 0.271 | 0.699 | 0.672 | 0.305 | 0.450 | 0.000 | 0.010 | 0.235 | 0.000 |
| low_snr_blockage | fix1 | 0.822 | 0.276 | 0.608 | 0.564 | 0.427 | 0.573 | 0.000 | 0.000 | 0.000 | 0.000 |
| edge_overload | B_state_v2_fixed_128x128 | 0.649 | 0.649 | 0.000 | 0.823 | n/a | n/a | n/a | n/a | n/a | n/a |
| edge_overload | semantic_path_short | 0.609 | 0.231 | 0.569 | 0.864 | 0.034 | 0.585 | 0.000 | 0.000 | 0.381 | 0.000 |
| edge_overload | fix1 | 0.384 | 0.104 | 0.557 | 0.697 | 0.265 | 0.712 | 0.000 | 0.001 | 0.022 | 0.000 |
| utm_conflict | B_state_v2_fixed_128x128 | 0.000 | 0.000 | 0.000 | 0.997 | n/a | n/a | n/a | n/a | n/a | n/a |
| utm_conflict | semantic_path_short | 0.000 | 0.000 | 0.541 | 0.592 | 0.501 | 0.499 | 0.000 | 0.000 | 0.000 | 0.091 |
| utm_conflict | fix1 | 0.000 | 0.000 | 0.679 | 60.770 | 0.373 | 0.325 | 0.301 | 0.000 | 0.000 | 0.179 |

## Findings

- `normal_patrol`: fix1 removes the previous deadline regression. Deadline violation is 0.000, with a stable cache/token mix around 0.434/0.566.
- `low_snr_soft` and `low_snr_blockage`: fix1 keeps semantic success high at 0.822 and improves task success over B_state_v2 (0.276 vs 0.128) while reducing deadline violation from 0.854 to 0.608. The current `low_snr_soft` remains an alias of `low_snr_blockage`.
- `edge_overload`: fix1 successfully removes cache_update overuse (0.381 -> 0.022) and lowers deadline violation slightly versus the first semantic-path run (0.569 -> 0.557), but task success remains far below B_state_v2 (0.104 vs 0.649). This is not paper-ready for edge overload.
- `utm_conflict`: fix1 does not solve UTM success. It keeps semantic/task success at 0 and still has UTM conflict around 0.179. Compared with first semantic-path run, deadline worsened (0.541 -> 0.679), so UTM routing needs a separate mobility/intent-aware fix rather than path gating alone.
- Cache collapse is avoided in normal/low-SNR settings, and cache_update no longer dominates edge_overload; the remaining blocker is selecting feasible low-delay token/cache routes under edge/UTM constraints.

## Recommendation

Do not use fix1 as final paper controller. Use it as a stabilized semantic-path baseline and run a fix2 focused on edge-overload resource floors/deadline slack and UTM mobility-mode/UAV assignment feasibility.
