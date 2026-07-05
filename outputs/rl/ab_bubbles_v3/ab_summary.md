# BUBBLES A/B v3 summary

> Primary accuracy metric: `average_accuracy`. `semantic_success_rate` saturates
> to 0 for every arm under the all-critical `utm_conflict` condition
> (epsilon_critical=0.82 is unreachable there), so it is reported for
> completeness and only carries signal under `nominal`.

## Peak condition (scenario=utm_conflict)

| arm | runs | average_accuracy | airspace_conflict_rate | utm_conflict_violation_rate | semantic_success_rate | task_success_rate | service_level_0_ratio | service_level_1_ratio | service_level_2_ratio | deadline_violation_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Cgreedy | 1 | 0.3859 | 0.6540 | 0.6540 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Ccache | 1 | 0.3127 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| A2 | 3 | 0.5628 | 0.3027 | 0.3027 | 0.0000 | 0.0000 | 0.3907 | 0.6093 | 0.0000 | 0.7193 |
| A2warm | 3 | 0.5628 | 0.3027 | 0.3027 | 0.0000 | 0.0000 | 0.3907 | 0.6093 | 0.0000 | 0.7193 |
| B2 | 3 | 0.5628 | 0.3027 | 0.3027 | 0.0000 | 0.0000 | 0.3907 | 0.6093 | 0.0000 | 0.7193 |
| B1 | 3 | 0.5815 | 0.4707 | 0.4707 | 0.0000 | 0.0000 | 0.2173 | 0.7827 | 0.0000 | 1.0000 |
| A1 | 3 | 0.5332 | 0.1147 | 0.1147 | 0.0000 | 0.0000 | 0.2527 | 0.7473 | 0.0000 | 0.7527 |

## Nominal condition (bubbles_daily arrivals, no background intents)

| arm | runs | average_accuracy | airspace_conflict_rate | utm_conflict_violation_rate | semantic_success_rate | task_success_rate | service_level_0_ratio | service_level_1_ratio | service_level_2_ratio | deadline_violation_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Cgreedy | 1 | 0.6241 | 0.0000 | 0.0000 | 0.8960 | 0.3680 | 0.3260 | 0.5400 | 0.1340 | 0.6320 |
| Ccache | 1 | 0.2655 | 0.0000 | 0.0000 | 0.2970 | 0.2970 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| A2 | 3 | 0.6334 | 0.0000 | 0.0000 | 0.9463 | 0.2760 | 0.5923 | 0.3983 | 0.0093 | 0.7240 |
| A2warm | 3 | 0.6334 | 0.0000 | 0.0000 | 0.9463 | 0.2760 | 0.5923 | 0.3983 | 0.0093 | 0.7240 |
| B2 | 3 | 0.6334 | 0.0000 | 0.0000 | 0.9463 | 0.2787 | 0.5923 | 0.3983 | 0.0093 | 0.7213 |
| B1 | 3 | 0.6355 | 0.0000 | 0.0000 | 0.9453 | 0.3153 | 0.5683 | 0.4217 | 0.0100 | 0.6667 |
| A1 | 3 | 0.8577 | 0.0000 | 0.0000 | 0.9590 | 0.0467 | 0.5113 | 0.4837 | 0.0050 | 0.9470 |

## Final lambda_conflict per seed (end of training)

- A2: seed0=4.305 (ep0 0.004, min 0.004, max 4.452), seed1=4.691 (ep0 0.044, min 0.028, max 4.691), seed2=3.717 (ep0 0.004, min 0.004, max 4.006)
- A2warm: seed0=4.329 (ep0 3.964, min 2.926, max 4.549), seed1=4.714 (ep0 4.004, min 3.285, max 4.714), seed2=3.755 (ep0 3.964, min 3.580, max 4.498)
- B2: seed0=0.000 (ep0 0.000, min 0.000, max 0.000), seed1=0.000 (ep0 0.000, min 0.000, max 0.000), seed2=0.000 (ep0 0.000, min 0.000, max 0.000)
- B1: seed0=6.945 (ep0 0.064, min 0.064, max 7.241), seed1=7.103 (ep0 0.104, min 0.104, max 7.303), seed2=6.554 (ep0 0.064, min 0.064, max 7.216)
- A1: seed0=0.755 (ep0 0.044, min 0.018, max 0.958), seed1=1.063 (ep0 0.000, min 0.000, max 1.063), seed2=0.798 (ep0 0.004, min 0.004, max 0.853)

## Criteria

- [FAIL] PRIMARY: B2 conflict - A2 conflict >= 0.05 (dual channel load-bearing) (observed +0.0000)
- [FAIL] A2 conflict rate <= 0.15 (observed 0.3027)
- [FAIL] A2 cache ratio <= 0.35 (observed 0.3907)
- [PASS] A2 average_accuracy >= A1 average_accuracy - 0.02 (observed margin +0.0495)
- [PASS] B1 conflict - A2 conflict >= 0.05 (slow-head effect) (observed +0.1680)
- [PASS] nominal A2 semantic success >= 0.92 (observed 0.9463)
- [FAIL] nominal A2 task success >= 0.30 (observed 0.2760)
- [FAIL] WARM H1: A2warm conflict < A2 conflict - 0.05 (warm-started dual is load-bearing) (observed +0.0000)
- [FAIL] WARM H2: A2warm conflict rate <= 0.15 (pulled by limit 0.08) (observed 0.3027)
- [PASS] WARM guard: A2warm cache ratio <= 0.5 (no cache-collapse escape route) (observed 0.3907)
- [PASS] WARM guard: A2warm average_accuracy >= A2 average_accuracy - 0.05 (observed margin +0.0500)
- [PASS] WARM guard: nominal A2warm semantic success >= 0.92 (observed 0.9463)

