# VQA-grounded Task-conditioned Semantic Utility Model

This report converts measured VQA correctness into a control-facing semantic utility interface.
The utility is task-conditioned and VQA-grounded: it estimates answer correctness, payload, and uncertainty, not image quality.

## Artifacts

- input predictions: `/home/qiankun/phd_research/vqa_semcom/outputs/vlm/v1_9_snr_predictions.csv`
- utility CSV: `/home/qiankun/phd_research/vqa_semcom/outputs/lut/v1_9_semantic_utility_with_ci.csv`
- API: `src/vqa_semcom/semantic/utility.py`

## Interface

```python
U_sem(task_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level)
# -> accuracy_mean, accuracy_lcb, payload_kb, uncertainty, sample_count
```

The RL/environment side should prefer `accuracy_lcb` when it needs conservative QoS decisions and use `uncertainty` to down-weight sparse cells.

## Calibration Summary

- utility cells: 648
- total measured samples: 160542
- SNR bins: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB
- sparse cells: 108
- SNR monotonic adjusted cells: 48
- cache SNR-invariant cells: 216
- confidence interval: Wilson 95% binomial interval from answer correctness
- uncertainty: CI half width plus finite-sample penalty, clipped to [0, 1]

## Service-level Summary

| service level | cells | mean accuracy | mean LCB | mean payload KB | mean uncertainty | samples |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 216 | 0.525 | 0.461 | 0.000 | 0.268 | 53514 |
| 1 | 216 | 0.720 | 0.550 | 0.920 | 0.319 | 53514 |
| 2 | 216 | 0.598 | 0.486 | 90.198 | 0.305 | 53514 |

## Accuracy Mean by SNR

| service level | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.525 | 0.525 | 0.525 | 0.525 | 0.525 | 0.525 |
| 1 | 0.636 | 0.675 | 0.728 | 0.746 | 0.761 | 0.771 |
| 2 | 0.582 | 0.594 | 0.596 | 0.601 | 0.606 | 0.607 |

## Notes for Paper Writing

- The original VLM prediction CSV is unchanged; this file is a calibrated semantic utility layer built on top of it.
- `s=0` cache is forced to be SNR-invariant because it does not transmit visual evidence.
- `s=1` semantic tokens and `s=2` image evidence are sanity-checked so higher sensed SNR does not reduce the calibrated mean utility for the same task condition.
- Sparse cells remain visible through `sample_count` and `uncertainty`; they are not silently treated as high-confidence measurements.
