# Deadline-aware Guard Mid-scale Validation

Settings: scenarios `low_snr_blockage, disaster_hotspot, edge_overload`; seeds `0,1,2`; `20` eval episodes per seed; `120` train episodes; `12` tasks per episode.

## low_snr_blockage

| policy | semantic success | task success | accuracy LCB | quality gap | delay | energy | deadline violation | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proposed_two_timescale_ppo | 1.000 | 0.028 | 0.922 | 0.000 | 666.307 | 944.401 | 0.972 | 0.000 | 0.028 | 0.000 | 0.972 |
| proposed_v2_deadline_guard | 0.933 | 0.058 | 0.844 | 0.010 | 26.674 | 139.601 | 0.938 | 0.000 | 0.037 | 0.963 | 0.000 |
| proposed_v2_no_image_under_low_snr | 0.933 | 0.058 | 0.844 | 0.010 | 26.674 | 139.601 | 0.938 | 0.000 | 0.037 | 0.963 | 0.000 |
| proposed_v2_nearest_uav_mobility | 0.696 | 0.106 | 0.757 | 0.091 | 15.394 | 1353.649 | 0.714 | 0.000 | 0.237 | 0.738 | 0.025 |

## disaster_hotspot

| policy | semantic success | task success | accuracy LCB | quality gap | delay | energy | deadline violation | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proposed_two_timescale_ppo | 0.220 | 0.110 | 0.572 | 0.278 | 1.560 | 158.416 | 0.290 | 0.000 | 0.107 | 0.893 | 0.000 |
| proposed_v2_deadline_guard | 0.220 | 0.110 | 0.572 | 0.278 | 1.560 | 158.408 | 0.290 | 0.000 | 0.107 | 0.893 | 0.000 |
| proposed_v2_no_image_under_low_snr | 0.220 | 0.110 | 0.572 | 0.278 | 1.560 | 158.408 | 0.290 | 0.000 | 0.107 | 0.893 | 0.000 |
| proposed_v2_nearest_uav_mobility | 0.220 | 0.110 | 0.572 | 0.278 | 1.560 | 158.408 | 0.290 | 0.000 | 0.107 | 0.893 | 0.000 |

## edge_overload

| policy | semantic success | task success | accuracy LCB | quality gap | delay | energy | deadline violation | UTM conflict | cache | token | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proposed_two_timescale_ppo | 0.738 | 0.738 | 0.682 | 0.045 | 1.283 | 144.300 | 0.000 | 0.000 | 0.057 | 0.943 | 0.000 |
| proposed_v2_deadline_guard | 0.738 | 0.738 | 0.682 | 0.045 | 1.283 | 144.300 | 0.000 | 0.000 | 0.057 | 0.943 | 0.000 |
| proposed_v2_no_image_under_low_snr | 0.738 | 0.738 | 0.682 | 0.045 | 1.283 | 144.300 | 0.000 | 0.000 | 0.057 | 0.943 | 0.000 |
| proposed_v2_nearest_uav_mobility | 0.706 | 0.626 | 0.668 | 0.056 | 2.561 | 384.350 | 0.121 | 0.000 | 0.046 | 0.954 | 0.000 |

## Judgement

- Low-SNR image overuse: fixed by the deadline/no-image guards. Old formal proposed image ratio was 0.917; mid-scale baseline proposed remains high at 0.972, while `proposed_v2_deadline_guard` and `proposed_v2_no_image_under_low_snr` reduce image ratio to 0.000. The nearest-UAV diagnostic also reduces image ratio to 0.025 but is less direct.
- Low-SNR deadline/task success: deadline violation is not solved. Old formal proposed deadline violation was 0.939; the deadline/no-image guards are 0.938, with task success 0.058. This reduces extreme delay/energy and payload, but does not yet improve task success enough for a final paper claim.
- Disaster hotspot: v2 stabilizes task success relative to the old formal proposed value 0.064. Mid-scale task success is 0.110 for the deadline guard and 0.110 for nearest-UAV mobility, with deadline violation 0.290; image ratio stays 0.000, so the policy remains token/cache dominated.
- Edge overload: preserved for deadline/no-image guards. Old formal proposed was semantic success 0.697, task success 0.681, deadline violation 0.046; mid-scale deadline/no-image guards reach semantic success 0.738, task success 0.738, deadline violation 0.000. The nearest-UAV diagnostic is weaker here: task success 0.626 and deadline violation 0.121.
- Recommendation: run a 300-episode formal v2 for `proposed_v2_deadline_guard` / `proposed_v2_no_image_under_low_snr` after one more low-SNR deadline tuning pass. The current guard clearly fixes image overuse and preserves edge-overload, but low-SNR task success remains too low for the final claim.
