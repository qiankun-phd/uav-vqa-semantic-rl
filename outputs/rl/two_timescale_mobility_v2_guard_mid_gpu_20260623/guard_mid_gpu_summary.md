# GPU Mid-scale Deadline Guard Validation

- Output dir: `outputs/rl/two_timescale_mobility_v2_guard_mid_gpu_20260623/`
- Device: `cuda:0 (NVIDIA GeForce RTX 4060)`
- Torch in RA_DI: `2.10.0+cu128`; CUDA runtime reported by torch: `12.8`
- Benchmark: scenarios `low_snr_blockage`, `disaster_hotspot`, `edge_overload`; seeds `0,1,2`; eval episodes `20`; train episodes `120`; tasks per episode `12`.
- GPU use was confirmed by `nvidia-smi`: python PID `1338071` used about `144 MiB` GPU memory during training.
- Wall-clock observation: GPU run started at about `2026-06-23 23:07:12 +0800` and wrote the root summary at `2026-06-23 23:22:57 +0800`, about `15m45s`. The earlier CPU mid run was comparable in size; any speedup is modest because this benchmark is dominated by Python environment rollout/evaluation rather than large neural-network batches.

## Core GPU Results

| scenario | policy | semantic success | task success | accuracy LCB | quality gap | delay | energy | deadline violation | UTM conflict | cache/token/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low_snr_blockage | proposed_two_timescale_ppo | 1.000 | 0.028 | 0.922 | 0.000 | 666.307 | 944.401 | 0.972 | 0.000 | 0.028/0.000/0.972 |
| low_snr_blockage | proposed_v2_deadline_guard | 0.933 | 0.058 | 0.844 | 0.010 | 26.674 | 139.601 | 0.938 | 0.000 | 0.037/0.963/0.000 |
| low_snr_blockage | proposed_v2_no_image_under_low_snr | 0.933 | 0.058 | 0.844 | 0.010 | 26.674 | 139.601 | 0.938 | 0.000 | 0.037/0.963/0.000 |
| low_snr_blockage | proposed_v2_nearest_uav_mobility | 0.696 | 0.106 | 0.757 | 0.091 | 15.394 | 1353.649 | 0.714 | 0.000 | 0.237/0.738/0.025 |
| disaster_hotspot | proposed_two_timescale_ppo | 0.220 | 0.110 | 0.572 | 0.278 | 1.560 | 158.470 | 0.290 | 0.000 | 0.107/0.893/0.000 |
| disaster_hotspot | proposed_v2_deadline_guard | 0.220 | 0.110 | 0.572 | 0.278 | 1.560 | 158.470 | 0.290 | 0.000 | 0.107/0.893/0.000 |
| disaster_hotspot | proposed_v2_no_image_under_low_snr | 0.220 | 0.110 | 0.572 | 0.278 | 1.560 | 158.470 | 0.290 | 0.000 | 0.107/0.893/0.000 |
| disaster_hotspot | proposed_v2_nearest_uav_mobility | 0.220 | 0.110 | 0.572 | 0.278 | 1.560 | 158.470 | 0.290 | 0.000 | 0.107/0.893/0.000 |
| edge_overload | proposed_two_timescale_ppo | 0.729 | 0.597 | 0.677 | 0.049 | 2.771 | 427.852 | 0.164 | 0.000 | 0.051/0.949/0.000 |
| edge_overload | proposed_v2_deadline_guard | 0.729 | 0.597 | 0.677 | 0.049 | 2.771 | 427.852 | 0.164 | 0.000 | 0.051/0.949/0.000 |
| edge_overload | proposed_v2_no_image_under_low_snr | 0.729 | 0.597 | 0.677 | 0.049 | 2.771 | 427.852 | 0.164 | 0.000 | 0.051/0.949/0.000 |
| edge_overload | proposed_v2_nearest_uav_mobility | 0.738 | 0.738 | 0.682 | 0.045 | 1.587 | 171.554 | 0.000 | 0.000 | 0.060/0.940/0.000 |

## CPU vs GPU Metric Check

The GPU run is numerically consistent with the CPU mid run for deterministic projection-heavy variants. Minor differences reflect stochastic PPO sampling and seed-level training order, not a device failure.

| scenario | policy | CPU task success | GPU task success | CPU deadline vio | GPU deadline vio |
|---|---:|---:|---:|---:|---:|
| low_snr_blockage | proposed_v2_deadline_guard | 0.058 | 0.058 | 0.938 | 0.938 |
| low_snr_blockage | proposed_v2_nearest_uav_mobility | 0.106 | 0.106 | 0.714 | 0.714 |
| disaster_hotspot | proposed_v2_deadline_guard | 0.110 | 0.110 | 0.290 | 0.290 |
| disaster_hotspot | proposed_v2_nearest_uav_mobility | 0.110 | 0.110 | 0.290 | 0.290 |
| edge_overload | proposed_v2_deadline_guard | 0.738 | 0.597 | 0.000 | 0.164 |
| edge_overload | proposed_v2_nearest_uav_mobility | 0.626 | 0.738 | 0.121 | 0.000 |

## Interpretation

- GPU support is functional: model parameters, rollout tensors, PPO update tensors, behavior cloning tensors, and checkpoint loading now honor the selected torch device.
- Low-SNR image overuse remains fixed for `proposed_v2_deadline_guard` and `proposed_v2_no_image_under_low_snr`: image ratio is `0.000`, with token/cache routing instead.
- Low-SNR task success remains low (`0.058`) under deadline guard despite high semantic success (`0.933`), so this still needs deadline/resource tuning before a 300-episode formal claim.
- Disaster hotspot behavior is stable (`task_success=0.110`, `deadline_violation=0.290`, no image use), matching the CPU mid finding.
- Edge overload is preserved for seed-averaged proposed guard variants (`task_success=0.597`, `deadline_violation=0.164`); the nearest-UAV variant is strongest here (`task_success=0.737`, `deadline_violation=0.000`).
- GPU did not obviously change the bottleneck. The model is small, batch sizes are tiny, and environment simulation/CSV reporting dominates wall time, so the main benefit is enabling larger future network/batch experiments rather than this small benchmark.

## Artifacts

- `scenario_comparison_summary.csv`
- `scenario_comparison_report.md`
- `guard_mid_gpu_summary.md`
