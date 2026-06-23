# Semantic Service-Candidate Utility Interface

This report documents the RL/env-facing candidate-service utility interface built on top of the calibrated VQA-grounded semantic utility model. The LUT key is unchanged:

```text
question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level
```

The new helper evaluates every candidate service for the same task condition:

```python
candidates = SemanticUtilityModel.from_csv(...).get_service_candidates(obs)
```

Each candidate returns `accuracy_mean`, `accuracy_lcb`, `uncertainty`, `payload_kb`, `sample_count`, `semantic_quality_gap`, `semantic_efficiency`, `is_snr_sensitive`, `recommended_for_low_snr`, and `recommended_for_critical`.

## Service-Level Summary

| service | name | cells | mean accuracy LCB | mean payload KB | mean uncertainty | mean sample count | paper interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | cache_answer | 216 | 0.461 | 0.000 | 0.268 | 247.8 | cache answer: low payload, depends on freshness/cache hit, not SNR-sensitive |
| 1 | semantic_token | 216 | 0.550 | 0.920 | 0.319 | 247.8 | semantic token: lightweight semantic communication using compact detector evidence |
| 2 | image_evidence | 216 | 0.486 | 90.198 | 0.305 | 247.8 | image evidence: high-payload visual evidence, stronger link/delay/queue sensitivity |

## Candidate Examples

### low_snr_blockage_token_route

| service | name | acc mean | acc LCB | uncertainty | payload KB | gap | efficiency | SNR-sensitive | low-SNR rec | critical rec | samples |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 0 | cache_answer | 0.530 | 0.506 | 0.083 | 0.000 | 0.274 | 0.213 | False | False | False | 281 |
| 1 | semantic_token | 0.897 | 0.856 | 0.095 | 1.027 | 0.000 | 0.382 | True | True | True | 281 |
| 2 | image_evidence | 0.954 | 0.922 | 0.085 | 37.487 | 0.000 | 0.022 | True | False | True | 281 |

### edge_overload_compact_evidence

| service | name | acc mean | acc LCB | uncertainty | payload KB | gap | efficiency | SNR-sensitive | low-SNR rec | critical rec | samples |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 0 | cache_answer | 0.231 | 0.151 | 0.370 | 0.000 | 0.649 | 0.000 | False | False | False | 13 |
| 1 | semantic_token | 1.000 | 0.772 | 0.391 | 1.188 | 0.028 | 0.207 | True | False | False | 13 |
| 2 | image_evidence | 1.000 | 0.772 | 0.391 | 206.769 | 0.028 | 0.002 | True | False | False | 13 |

### critical_counting_conservative_gate

| service | name | acc mean | acc LCB | uncertainty | payload KB | gap | efficiency | SNR-sensitive | low-SNR rec | critical rec | samples |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 0 | cache_answer | 0.615 | 0.504 | 0.383 | 0.000 | 0.316 | 0.117 | False | False | False | 13 |
| 1 | semantic_token | 0.615 | 0.355 | 0.454 | 1.188 | 0.465 | 0.000 | True | False | False | 13 |
| 2 | image_evidence | 0.000 | 0.000 | 0.391 | 191.093 | 0.820 | 0.000 | True | False | False | 13 |

## Interpretation for Current Scenarios

- `low_snr_blockage`: the interface supports explaining why token/cache can dominate image evidence. Token candidates keep payload near 1 KB, while image evidence carries tens to hundreds of KB and increases transmission delay under low SNR.
- `edge_overload`: token candidates are the natural compact-evidence route because image evidence has high edge workload and payload. The candidate fields expose this through payload and efficiency without changing the LUT.
- `critical` risk: controllers should use `accuracy_lcb`, `semantic_quality_gap`, `uncertainty`, and `sample_count` instead of mean accuracy alone. The `recommended_for_critical` flag is conservative and only fires when LCB clears epsilon with acceptable uncertainty.

## Notes

- `is_snr_sensitive=False` for cache because cache reuse does not transmit fresh visual evidence over the current link.
- `recommended_for_low_snr` is a routing prior, not a hard constraint. It favors semantic tokens when their LCB clears epsilon and allows fresh cache only when it is already semantically sufficient.
- `semantic_efficiency` is derived for ranking candidates. It should guide exploration/projection together with the explicit quality gap, not replace the semantic QoS constraint.
