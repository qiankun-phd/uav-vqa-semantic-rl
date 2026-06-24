# State V2 Fixed Medium PPO Summary

Medium run after fixing `state_v2` to a canonical feature layout. Each run trained a two-timescale semantic PPO for 500 episodes with 20 tasks per episode and evaluated for 50 episodes on `cuda:0`.

## Completion

- Completed runs: 30/30
- Missing or failed runs: 0
- Output root: `outputs/rl/state_v2_fixed_medium_20260624`

## Evaluation: A vs B

| scenario | candidate | sem_success | task_success | acc_lcb | gap | delay | deadline_vio | energy | payload | cache | token | image | utm_conflict | reward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal_patrol | A_state_v1_128x128 | 0.156 | 0.156 | 0.752 | 0.249 | 1.146 | 0.000 | 143.695 | 30.742 | 0.107 | 0.744 | 0.149 | 0.000 | 0.368 |
| nominal_patrol | B_state_v2_fixed_128x128 | 0.156 | 0.156 | 0.752 | 0.249 | 1.146 | 0.000 | 143.695 | 30.742 | 0.107 | 0.744 | 0.149 | 0.000 | 0.368 |
| disaster_hotspot | A_state_v1_128x128 | 0.179 | 0.084 | 0.787 | 0.311 | 1.535 | 0.291 | 155.610 | 1.042 | 0.113 | 0.887 | 0.000 | 0.000 | -1.160 |
| disaster_hotspot | B_state_v2_fixed_128x128 | 0.180 | 0.099 | 0.790 | 0.321 | 1.318 | 0.211 | 144.118 | 4.208 | 0.112 | 0.871 | 0.017 | 0.000 | -0.819 |
| low_snr_blockage | A_state_v1_128x128 | 0.948 | 0.128 | 0.885 | 0.012 | 21.135 | 0.854 | 132.171 | 0.918 | 0.094 | 0.906 | 0.000 | 0.000 | -4.750 |
| low_snr_blockage | B_state_v2_fixed_128x128 | 0.948 | 0.128 | 0.885 | 0.012 | 21.135 | 0.854 | 132.171 | 0.918 | 0.094 | 0.906 | 0.000 | 0.000 | -4.750 |
| edge_overload | A_state_v1_128x128 | 0.628 | 0.626 | 0.862 | 0.066 | 1.627 | 0.029 | 211.240 | 0.836 | 0.046 | 0.954 | 0.000 | 0.000 | 1.661 |
| edge_overload | B_state_v2_fixed_128x128 | 0.649 | 0.649 | 0.862 | 0.061 | 1.253 | 0.000 | 141.170 | 0.823 | 0.051 | 0.949 | 0.000 | 0.000 | 1.838 |
| utm_conflict | A_state_v1_128x128 | 0.000 | 0.000 | 0.717 | 0.331 | 1.078 | 0.000 | 134.265 | 0.997 | 0.161 | 0.839 | 0.000 | 0.134 | -0.513 |
| utm_conflict | B_state_v2_fixed_128x128 | 0.000 | 0.000 | 0.717 | 0.331 | 1.078 | 0.000 | 134.265 | 0.997 | 0.161 | 0.839 | 0.000 | 0.134 | -0.513 |
| OVERALL | A_state_v1_128x128 | 0.382 | 0.199 | 0.801 | 0.194 | 5.304 | 0.235 | 155.396 | 6.907 | 0.104 | 0.866 | 0.030 | 0.027 | -0.879 |
| OVERALL | B_state_v2_fixed_128x128 | 0.387 | 0.206 | 0.801 | 0.195 | 5.186 | 0.213 | 139.084 | 7.538 | 0.105 | 0.862 | 0.033 | 0.027 | -0.775 |

## B minus A Delta

| scenario | delta_sem_success | delta_task_success | delta_gap | delta_delay | delta_deadline_vio | delta_payload | delta_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| disaster_hotspot | 0.001 | 0.015 | 0.009 | -0.217 | -0.080 | 3.166 | 0.341 |
| low_snr_blockage | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| edge_overload | 0.022 | 0.023 | -0.005 | -0.374 | -0.029 | -0.012 | 0.178 |
| utm_conflict | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| OVERALL | 0.004 | 0.008 | 0.001 | -0.118 | -0.022 | 0.631 | 0.104 |

## Training Convergence

| scenario | candidate | final50_reward | reward_delta | final50_success | success_delta | Q_deadline | Q_quality | non_cache | reward_stable | success_rising | Q_deadline_explode | Q_quality_explode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal_patrol | A_state_v1_128x128 | -29.746 | -2.458 | 0.136 | -0.021 | 13.808 | 1.244 | 0.890 | 0.667 | 0.000 | 0.000 | 0.000 |
| nominal_patrol | B_state_v2_fixed_128x128 | -30.656 | -2.445 | 0.136 | -0.021 | 13.808 | 1.244 | 0.890 | 0.667 | 0.000 | 0.000 | 0.000 |
| disaster_hotspot | A_state_v1_128x128 | -16.821 | -0.914 | 0.091 | 0.009 | 3.874 | 1.298 | 0.873 | 1.000 | 0.000 | 0.000 | 0.000 |
| disaster_hotspot | B_state_v2_fixed_128x128 | -18.344 | -2.052 | 0.095 | 0.009 | 3.409 | 1.357 | 0.879 | 0.667 | 0.000 | 0.000 | 0.000 |
| low_snr_blockage | A_state_v1_128x128 | -869.711 | -158.429 | 0.201 | -0.016 | 105.603 | 0.535 | 0.640 | 0.333 | 0.000 | 0.000 | 0.000 |
| low_snr_blockage | B_state_v2_fixed_128x128 | -869.773 | -158.445 | 0.201 | -0.016 | 105.603 | 0.535 | 0.640 | 0.333 | 0.000 | 0.000 | 0.000 |
| edge_overload | A_state_v1_128x128 | 5.037 | -0.800 | 0.402 | 0.002 | 7.143 | 0.719 | 0.964 | 0.000 | 0.333 | 0.000 | 0.000 |
| edge_overload | B_state_v2_fixed_128x128 | 8.699 | 0.327 | 0.508 | 0.022 | 4.804 | 0.661 | 0.961 | 1.000 | 0.667 | 0.000 | 0.000 |
| utm_conflict | A_state_v1_128x128 | -48.189 | 1.139 | 0.000 | 0.000 | 7.797 | 1.413 | 0.823 | 0.667 | 0.000 | 0.000 | 0.000 |
| utm_conflict | B_state_v2_fixed_128x128 | -48.256 | 1.128 | 0.000 | 0.000 | 7.797 | 1.413 | 0.823 | 0.667 | 0.000 | 0.000 | 0.000 |
| OVERALL | A_state_v1_128x128 | -191.886 | -32.292 | 0.166 | -0.005 | 27.645 | 1.042 | 0.838 | 0.533 | 0.067 | 0.000 | 0.000 |
| OVERALL | B_state_v2_fixed_128x128 | -191.666 | -32.297 | 0.188 | -0.001 | 27.084 | 1.042 | 0.839 | 0.667 | 0.133 | 0.000 | 0.000 |

## Interpretation

- Overall score heuristic selects `B_state_v2_fixed_128x128` for the next final run.
- Overall A task success / semantic success: 0.199 / 0.382.
- Overall B task success / semantic success: 0.206 / 0.387.
- Overall A deadline violation / quality gap: 0.235 / 0.194.
- Overall B deadline violation / quality gap: 0.213 / 0.195.
- Edge overload: B task success 0.649 vs A 0.626; B deadline violation 0.000 vs A 0.029.
- Low SNR: deadline violation remains high for A/B at 0.854 / 0.854; this remains the main blocker before a paper-scale final run.

Service-mix check:
- `A_state_v1_128x128` dominant service is token at 0.866; collapse=False.
- `B_state_v2_fixed_128x128` dominant service is token at 0.862; collapse=False.

## Convergence Decision

- The final-50 windows look mostly stable and queues do not explode, but this is still a 500-episode medium run. A 1000/2000-episode final run is recommended for paper results.
- Recommended next paper model: `B_state_v2_fixed_128x128`.
- If compute is limited, run the recommended model for 1000 episodes on all five scenarios first, then expand to 2000 episodes only after low-SNR deadline behavior is acceptable.

## Artifacts

- `outputs/rl/state_v2_fixed_medium_20260624/medium_eval_all_seed_results.csv`
- `outputs/rl/state_v2_fixed_medium_20260624/medium_eval_summary_by_scenario.csv`
- `outputs/rl/state_v2_fixed_medium_20260624/medium_training_all_seed_summary.csv`
- `outputs/rl/state_v2_fixed_medium_20260624/medium_training_summary_by_scenario.csv`
