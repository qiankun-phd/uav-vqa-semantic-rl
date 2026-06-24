# Low-SNR Deadline Diagnosis

Source: eval-only replay of `B_state_v2_fixed_128x128` low_snr_blockage checkpoints into `outputs/rl/low_snr_deadline_tuning_20260624/baseline_B_state_v2_fixed/`, using the updated rollout schema with delay components.

## Headline

- Rows analyzed: 1800 tasks across 3 seeds.
- Deadline violation rate: 0.854.
- Semantic success rate: 0.948.
- Image ratio: 0.000; image is effectively suppressed.

## Delay Component Attribution on Deadline-Violating Tasks

| component | mean_s | p50_s | p90_s | primary_count | primary_share |
|---|---:|---:|---:|---:|---:|
| fly_delay_s | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| tx_delay_s | 23.796 | 14.529 | 54.259 | 1538 | 1.000 |
| queue_delay_s | 0.179 | 0.198 | 0.211 | 0 | 0.000 |
| infer_delay_s | 0.294 | 0.272 | 0.272 | 0 | 0.000 |
| arrival_delay_s | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| sense_delay_s | 0.160 | 0.160 | 0.160 | 0 | 0.000 |
| load_delay_s | 0.040 | 0.040 | 0.040 | 0 | 0.000 |

Interpretation: the dominant blocker is the largest primary-count component above. In this replay, `tx_delay_s` dominates essentially all deadline-violating tasks, while queue, inference, load, and mobility arrival delays are secondary.

## Service-Level Deadline Behavior

| service_level | count | deadline_violation_rate | mean_delay_s | semantic_success_rate |
|---:|---:|---:|---:|---:|
| 0 | 170 | 0.000 | 0.254 | 0.876 |
| 1 | 1630 | 0.944 | 23.312 | 0.955 |
| 2 | 0 | 0.000 | 0.000 | 0.000 |
| 3 | 0 | 0.000 | 0.000 | 0.000 |

## Token-Service Deadline Distribution

- `delay_s` among token deadline violations: mean 24.519, p50 15.251, p90 54.993.
- `fly_delay_s` among token deadline violations: mean 0.000, p50 0.000, p90 0.000.
- `tx_delay_s` among token deadline violations: mean 23.796, p50 14.529, p90 54.259.
- `queue_delay_s` among token deadline violations: mean 0.179, p50 0.198, p90 0.211.
- `infer_delay_s` among token deadline violations: mean 0.294, p50 0.272, p90 0.272.
- `arrival_delay_s` among token deadline violations: mean 0.000, p50 0.000, p90 0.000.

## Resource Saturation for Token Service

- `bandwidth_ge_99pct`: 0.000
- `power_ge_99pct`: 0.955
- `cpu_ge_99pct`: 0.000
- `gpu_ge_99pct`: 0.000
- `mean_bandwidth_hz`: 633989.264
- `mean_power_w`: 0.978
- `mean_cpu_share`: 0.393
- `mean_gpu_share`: 0.146

## Mobility Bottleneck Check

- Mean arrival/total delay ratio on violating tasks: 0.000.
- Arrival delay p90 on violating tasks: 0.000 s.
- This ratio is near zero, so mobility is not the low-SNR deadline bottleneck in this replay; the main blocker is token transmission delay under very low SNR.
