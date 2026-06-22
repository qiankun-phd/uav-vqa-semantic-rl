# VQA-grounded Semantic Utility Model

Last updated: 2026-06-22 Asia/Macau

This module turns measured VQA outcomes into a stable control-facing semantic utility model. It is not an image-quality score and it is not only a raw lookup table. The model estimates how useful a selected evidence service is for answering a task under sensed SNR, view quality, cache freshness, and risk level.

## Definition

The semantic utility API is:

```python
U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
```

It returns:

```python
accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

The implementation is:

```text
src/vqa_semcom/semantic/utility.py
```

The current calibrated utility table is:

```text
outputs/lut/v1_9_semantic_utility_with_ci.csv
```

The source measurements are the V1.9 Qwen/VQA prediction rows. Each row records a task, evidence service level, sensed SNR bin, view/freshness/risk condition, transmitted payload, and whether the final answer was correct.

## Fields

`accuracy_mean` is the calibrated expected answer correctness for the task-conditioned cell.

`accuracy_lcb` is the lower confidence bound used by conservative control policies. It is the recommended QoS estimate for RL/resource allocation when the system must avoid over-trusting noisy or sparse cells.

`uncertainty` summarizes statistical confidence. It combines the Wilson confidence interval half width with a finite-sample penalty, then clips the value to `[0, 1]`.

`sample_count` is the number of measured VQA samples supporting the cell. Sparse cells remain visible instead of being silently treated as reliable.

`payload_kb` is the measured communication load for the selected evidence service:

- `s=0`: cache answer, approximately zero payload.
- `s=1`: detector semantic tokens.
- `s=2`: image evidence.

## Wilson CI

For each cell, measured answer correctness is treated as a Bernoulli outcome. If `n` is `sample_count` and `k` is the number of correct answers, the raw mean is:

```text
p_hat = k / n
```

The 95% Wilson interval is used instead of a plain normal interval because several cells have small sample counts:

```text
center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
margin = z / (1 + z^2 / n)
         * sqrt(p_hat(1-p_hat)/n + z^2/(4n^2))
```

where `z = 1.96`. The interval is clipped to `[0, 1]`.

## SNR Monotonic Sanity Check

Measured VQA accuracy can fluctuate across SNR bins because of finite samples and VLM randomness. For the same:

```text
task_type, service_level, view_quality_bin, freshness_bin, risk_level
```

the calibrated mean is constrained so that higher SNR does not reduce the semantic utility. The raw measurements are preserved in `raw_accuracy_mean`, `raw_accuracy_ci_low`, and `raw_accuracy_ci_high`.

The calibration note `snr_monotonic_adjusted` marks cells where this sanity check changed the control-facing value.

## Cache SNR-invariant Assumption

For `s=0` cache answer reuse, the system does not transmit visual evidence over the current channel. Therefore, the calibrated cache utility is forced to be SNR-invariant within each non-SNR task condition.

Rows affected by this assumption are marked with:

```text
cache_snr_invariant
```

This keeps the semantic model aligned with the communication model: cache quality depends on freshness and task/risk context, not on current sensed SNR.

## Sparse Cell Handling

Cells with low `sample_count` are marked with:

```text
sparse_cell
```

Sparse cells keep their measured/calibrated values, but their uncertainty is higher. RL policies should either use `accuracy_lcb` directly or penalize high `uncertainty` during action selection.

## Why RL Should Use `accuracy_lcb`

Using only `accuracy_mean` can make the policy over-select services whose estimated quality is high because of sampling noise. `accuracy_lcb` is a conservative utility estimate:

```text
quality_ok = accuracy_lcb >= epsilon_k
```

This is especially important for critical tasks and sparse SNR/view/freshness combinations. The mean remains useful for reporting expected performance, while the lower bound is safer for online service-level and resource decisions.

Recommended rollout info fields:

```python
info = {
    "semantic_accuracy_mean": u.accuracy_mean,
    "semantic_accuracy_lcb": u.accuracy_lcb,
    "semantic_uncertainty": u.uncertainty,
    "semantic_sample_count": u.sample_count,
    "payload_kb": u.payload_kb,
}
```

## Current V1.9 Calibration Snapshot

From `outputs/reports/semantic_utility_calibration.md`:

- utility cells: 648
- total measured samples: 160542
- SNR bins: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB
- sparse cells: 108
- SNR monotonic adjusted cells: 48
- cache SNR-invariant cells: 216

The API examples can be regenerated with:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/demo_semantic_utility_query.py
```

