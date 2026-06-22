# Semantic Utility API Examples

Utility CSV: `/home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_semantic_utility_with_ci.csv`

These examples call:

```python
U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
# -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

For RL/resource control, `accuracy_lcb` is the conservative QoS estimate. `accuracy_mean` remains useful for reporting expected answer correctness.

| task_type | service_level | snr_bin | view | freshness | risk | accuracy_mean | accuracy_lcb | payload_kb | uncertainty | sample_count |
|---|---|---|---|---|---|---|---|---|---|---|
| presence | 0 | -5dB | good | fresh | normal | 0.672 | 0.623 | 0.000 | 0.172 | 64 |
| presence | 1 | 20dB | good | fresh | normal | 0.891 | 0.791 | 0.740 | 0.202 | 64 |
| presence | 2 | 0dB | medium | fresh | critical | 1.000 | 0.772 | 61.810 | 0.391 | 13 |
| counting | 1 | 10dB | good | stale | normal | 0.867 | 0.703 | 0.785 | 0.294 | 30 |
| counting | 2 | 15dB | poor | expired | critical | 0.000 | 0.000 | 131.733 | 0.052 | 441 |

Interpretation:

- `s=0` cache has near-zero payload and does not depend on current SNR.
- `s=1` semantic tokens usually have much lower payload than image evidence.
- `s=2` image evidence may increase payload substantially, so it should be selected only when the utility gain justifies the cost.
- High `uncertainty` or low `sample_count` should make RL treat the estimate conservatively.
