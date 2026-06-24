# Low-SNR Deadline Tuning Summary

Scenario: `low_snr_blockage`; candidate base: `B_state_v2_fixed_128x128`; seeds `0,1,2`; train episodes `500`; eval episodes `50`; tasks per episode `20`; device `cuda:0`.

## Results

| variant | semantic_success | task_success | deadline_violation | avg_delay | tx_delay | queue_delay | infer_delay | arrival_delay | payload | cache | token | image | cache_fallback | reward | cache_collapse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_B_state_v2_fixed | 0.948 | 0.128 | 0.854 | 21.135 | 20.456 | 0.170 | 0.278 | 0.000 | 0.918 | 0.094 | 0.906 | 0.000 | 0.000 | -4.750 | False |
| T1_deadline_slack | 0.948 | 0.128 | 0.854 | 21.135 | 20.456 | 0.170 | 0.278 | 0.000 | 0.918 | 0.094 | 0.906 | 0.000 | 0.000 | -4.750 | False |
| T2_token_fast | 0.948 | 0.126 | 0.846 | 20.836 | 20.101 | 0.249 | 0.256 | 0.000 | 0.918 | 0.094 | 0.906 | 0.000 | 0.000 | -4.688 | False |
| T3_cache_fallback | 0.948 | 0.128 | 0.851 | 20.809 | 20.134 | 0.170 | 0.275 | 0.000 | 0.915 | 0.098 | 0.902 | 0.000 | 0.098 | -4.681 | False |
| T1_T2_slack_token_fast | 0.948 | 0.126 | 0.846 | 20.836 | 20.101 | 0.249 | 0.256 | 0.000 | 0.918 | 0.094 | 0.906 | 0.000 | 0.000 | -4.688 | False |

## Delta vs Baseline

| variant | delta_semantic_success | delta_task_success | delta_deadline_violation | delta_delay | delta_reward |
|---|---:|---:|---:|---:|---:|
| T1_deadline_slack | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| T2_token_fast | 0.001 | -0.002 | -0.009 | -0.299 | 0.062 |
| T3_cache_fallback | 0.000 | 0.000 | -0.003 | -0.325 | 0.069 |
| T1_T2_slack_token_fast | 0.001 | -0.002 | -0.009 | -0.299 | 0.062 |

## Recommendation

- Recommended variant for the next 1000-episode pass: `T2_token_fast`.
- Selection criterion: prioritize lower deadline violation while preserving semantic success and avoiding cache collapse; the observed task-success change is negligible at this scale.
- `T2_token_fast` semantic success 0.948 vs baseline 0.948; deadline violation 0.846 vs baseline 0.854.
- T2/T1+T2 improves deadline only modestly because the diagnosis shows the remaining low-SNR blocker is token transmission delay under poor link capacity, not mobility arrival or CPU/GPU inference.

## Artifacts

- `outputs/rl/low_snr_deadline_tuning_20260624/diagnosis.md`
- `outputs/rl/low_snr_deadline_tuning_20260624/tuning_summary.csv`
- `outputs/rl/low_snr_deadline_tuning_20260624/tuning_all_seed_results.csv`
