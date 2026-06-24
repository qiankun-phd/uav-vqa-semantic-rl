# Low-SNR Deadline Diagnosis 2026-06-24

Scope: environment-side diagnosis only. No algorithm code, LUT, or environment dynamics were modified.

## Scenario Configuration

| item | value |
|---|---:|
| num_uavs | 3 |
| num_edges | 1 |
| num_areas | 5 |
| tasks_per_episode | 20 |
| episode_steps | 12 |
| area_spacing_m | 520.0 |
| area_radius_m | 80.0 |
| uav_altitude_m | 70.0 |
| bandwidth_hz | 650000.0 |
| semantic_cache_radius_m | 240.0 |
| tau_scale | 1.1 |
| risk_cycle | ['normal', 'critical', 'normal', 'normal'] |
| view_quality_cycle | ['medium', 'poor', 'medium', 'good'] |
| semantic_threshold_by_risk | {'normal': 0.56, 'critical': 0.78, 'high': 0.78} |

A2G stress parameters:

| parameter | value |
|---|---:|
| path_loss_exponent | 2.9 |
| noise_figure_db | 7.0 |
| excess_loss_db | 18.0 |
| los_excess_loss_db | 5.0 |
| nlos_excess_loss_db | 28.0 |
| fading_mode | slow_fading |
| slow_fading_std_db | 4.5 |
| fading_correlation | 0.7 |
| interference_overlap_scale | 0.06 |

## Candidate-Level Delay Decomposition

The table reports mean / p50 / p90 / p95 / max over all generated low_snr_blockage tasks from 60 deterministic seeds. Candidate actions use the environment default resource floors for each service level.

| service | deadline vio | deadline_s | SINR dB | rate Mbps | payload KB | total delay s | tx s | arrival s | queue s | infer s | sense+load+utm s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 4.811 / 5.500 / 5.500 / 5.500 / 5.500 | -21.965 / -22.723 / -13.987 / -11.914 / -5.157 | 0.001 / 0.000 / 0.002 / 0.003 / 0.012 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0.179 / 0.177 / 0.229 / 0.245 / 0.264 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 / 0.000 / 0.000 | 0.119 / 0.117 / 0.169 / 0.185 / 0.204 | 0.020 / 0.020 / 0.020 / 0.020 / 0.020 | 0.040 / 0.040 / 0.040 / 0.040 / 0.040 |
| 1 | 1.000 | 4.811 / 5.500 / 5.500 / 5.500 / 5.500 | -21.176 / -21.957 / -13.175 / -11.114 / -4.345 | 0.010 / 0.003 / 0.025 / 0.049 / 0.222 | 0.794 / 0.726 / 1.030 / 1.030 / 1.030 | 26.895 / 28.652 / 38.238 / 41.974 / 82.930 | 3.341 / 1.946 / 7.889 / 11.685 / 49.954 | 22.505 / 25.285 / 29.738 / 31.799 / 33.694 | 0.119 / 0.117 / 0.169 / 0.185 / 0.204 | 0.680 / 0.816 / 0.816 / 0.816 / 0.816 | 0.250 / 0.250 / 0.250 / 0.250 / 0.250 |
| 2 | 1.000 | 4.811 / 5.500 / 5.500 / 5.500 / 5.500 | -21.246 / -22.004 / -13.268 / -11.195 / -4.438 | 0.017 / 0.006 / 0.043 / 0.069 / 0.288 | 33.717 / 33.290 / 37.487 / 37.487 / 37.487 | 104.958 / 75.650 / 219.203 / 296.715 / 1135.313 | 78.805 / 46.272 / 185.959 / 263.286 / 1102.504 | 22.505 / 25.285 / 29.738 / 31.799 / 33.694 | 0.119 / 0.117 / 0.169 / 0.185 / 0.204 | 2.738 / 2.899 / 2.899 / 2.899 / 2.899 | 0.790 / 0.790 / 0.790 / 0.790 / 0.790 |

Mean component contribution by service:

| service | total s | arrival s | tx s | queue s | infer s | sense s | load s | utm s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.179 | 0.000 | 0.000 | 0.119 | 0.020 | 0.000 | 0.040 | 0.000 |
| 1 | 26.895 | 22.505 | 3.341 | 0.119 | 0.680 | 0.160 | 0.040 | 0.050 |
| 2 | 104.958 | 22.505 | 78.805 | 0.119 | 2.738 | 0.360 | 0.380 | 0.050 |

## Existing Formal Rollout Cross-Check

| policy | deadline vio | task success | semantic success | delay s | arrival s | payload KB | LCB | gap | s0/s1/s2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.000 | 0.053 | 0.053 | 0.173 | 0.000 | 0.000 | 0.343 | 0.465 | 1.000/0.000/0.000 |
| always_semantic_token | 0.955 | 0.018 | 0.661 | 13.674 | 12.394 | 1.047 | 0.634 | 0.210 | 0.000/1.000/0.000 |
| always_image | 0.988 | 0.002 | 0.673 | 35.056 | 11.964 | 53.872 | 0.623 | 0.265 | 0.000/0.000/1.000 |
| semantic_greedy | 0.913 | 0.087 | 0.950 | 15.411 | 12.202 | 2.557 | 0.856 | 0.005 | 0.063/0.887/0.050 |
| oracle_best_feasible_evidence | 0.916 | 0.082 | 0.950 | 34.820 | 12.097 | 41.572 | 0.911 | 0.005 | 0.064/0.053/0.883 |
| proposed_two_timescale_ppo | 0.939 | 0.053 | 0.955 | 509.965 | 0.000 | 34.288 | 0.912 | 0.004 | 0.053/0.030/0.917 |

## Diagnosis

1. Link rate is the primary physical stressor. The scenario combines 650 kHz bandwidth, area spacing 520 m, altitude 70 m, path-loss exponent 2.9, 18 dB extra blockage loss, and 28 dB NLoS loss. Candidate service rates are low enough that service-level 2/image has large tx delay even with max image resources.
2. Image evidence is intentionally near-impossible under this scenario. Candidate image delay is dominated by transmission plus travel; formal proposed rollouts that stay in place remove travel but still show image delay explosion under very low SNR. This is consistent with an image-impossible stress case rather than a bug.
3. UAV arrival delay is a major contributor for default serve_task candidates and heuristic baselines, but it is not the reason proposed PPO image rows fail in the formal rollout, where arrival delay is approximately zero. Therefore changing only UAV speed would not solve the proposed-policy low-SNR deadline violation.
4. Edge queue and inference are secondary. Queue delay and inference delay are small compared with tx and arrival delay in the low-SNR setting; edge load is not the bottleneck for this scenario.
5. Resource caps are internally reasonable for the stress design: token uses 45% bandwidth / 0.5 W / 0.25 CPU / 0.05 GPU by default, image uses 100% bandwidth / 1 W / 0.55 CPU / 0.35 GPU, and critical tasks get boosted resources. The problem persists even at image max bandwidth/power because the channel is deliberately weak.

## Recommended Config Patch (not applied)

Do not apply this to the default benchmark unless the paper needs a partially image-feasible low-SNR variant. Keep the current default as a stress scenario. A softer variant could be:

```yaml
multi_uav_env:
  scenarios:
    low_snr_blockage_soft:
      env:
        bandwidth_hz: 1000000.0        # current: 650000.0
        area_spacing_m: 380.0          # current: 520.0
        a2g:
          excess_loss_db: 12.0         # current: 18.0
          nlos_excess_loss_db: 22.0    # current: 28.0
      task_layout:
        tau_scale: 1.35                # current: 1.1
```

Use this only for a separate ablation such as `low_snr_blockage_soft`; do not overwrite the current stress preset.

## Artifact Notes

- Detailed sampled decomposition: `outputs/env/low_snr_deadline_diagnosis_20260624/candidate_delay_decomposition.csv`.
- No environment dynamics were changed.
