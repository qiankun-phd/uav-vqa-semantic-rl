# Semantic Service-Candidate Utility Interface

This report documents the RL/env-facing candidate-service utility interface built on top of the calibrated VQA-grounded semantic utility model. The LUT key is unchanged:

```text
question_type, service_level, snr_bin, view_quality_bin, freshness_bin, risk_level
```

The helper evaluates every candidate service for the same task condition:

```python
candidates = SemanticUtilityModel.from_csv(...).get_service_candidates(obs)
```

Each candidate returns semantic utility, payload, delay feasibility, and routing hints: `accuracy_mean`, `accuracy_lcb`, `uncertainty`, `payload_kb`, `sample_count`, `semantic_quality_gap`, `semantic_efficiency`, `estimated_delay_s`, `estimated_delay_feasible`, `semantic_feasible`, `deadline_feasible`, `joint_feasible`, `is_snr_sensitive`, `recommended_for_low_snr`, and `recommended_for_critical`.

## Service-Level Summary

| service | name | cells | mean accuracy LCB | mean payload KB | mean uncertainty | mean sample count | paper interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | cache_answer | 216 | 0.461 | 0.000 | 0.268 | 247.8 | cache answer: low payload, depends on freshness/cache hit, not SNR-sensitive |
| 1 | semantic_token | 216 | 0.550 | 0.920 | 0.319 | 247.8 | semantic token: lightweight semantic communication using compact detector evidence |
| 2 | image_evidence | 216 | 0.486 | 90.198 | 0.305 | 247.8 | image evidence: high-payload visual evidence, stronger link/delay/queue sensitivity |

## Feasibility Definitions

```text
semantic_feasible = accuracy_lcb >= epsilon_k
deadline_feasible = estimated_delay_s <= deadline_s
estimated_delay_feasible = deadline_feasible
joint_feasible = semantic_feasible and deadline_feasible
```

Delay can be provided by the environment through `estimated_delay_by_service`, `delay_by_service`, `service_delay_s`, `service_delay_by_level`, or `estimated_delay_s_by_service`. If no delay is provided, the helper uses a conservative service/payload fallback.

## Low-SNR Blockage Service Feasibility

| service | name | acc LCB | payload KB | delay s | semantic feasible | deadline feasible | joint feasible | gap | efficiency | low-SNR rec | critical rec |
|---:|---|---:|---:|---:|---|---|---|---:|---:|---|---|
| 0 | cache_answer | 0.506 | 0.000 | 0.050 | False | True | False | 0.274 | 0.213 | False | False |
| 1 | semantic_token | 0.856 | 1.027 | 1.200 | True | True | True | 0.000 | 0.382 | True | True |
| 2 | image_evidence | 0.922 | 37.487 | 8.500 | True | False | False | 0.000 | 0.022 | False | True |

Interpretation for `low_snr_blockage`: image evidence has strong semantic LCB in this example, but its estimated delay is above the deadline because low SNR makes large payload transmission expensive. Semantic tokens keep the payload near 1 KB and satisfy both semantic and deadline feasibility, which is the paper claim of lightweight semantic communication. Cache has the lowest payload and delay, but it fails the semantic LCB threshold in this critical stale-cache setting.

## Additional Candidate Examples

### edge_overload_compact_evidence

| service | name | acc LCB | payload KB | delay s | semantic feasible | deadline feasible | joint feasible | gap | efficiency | low-SNR rec | critical rec |
|---:|---|---:|---:|---:|---|---|---|---:|---:|---|---|
| 0 | cache_answer | 0.151 | 0.000 | 0.050 | False | True | False | 0.649 | 0.000 | False | False |
| 1 | semantic_token | 0.772 | 1.188 | 2.400 | False | True | False | 0.028 | 0.207 | False | False |
| 2 | image_evidence | 0.772 | 206.769 | 9.000 | False | False | False | 0.028 | 0.002 | False | False |

### critical_counting_conservative_gate

| service | name | acc LCB | payload KB | delay s | semantic feasible | deadline feasible | joint feasible | gap | efficiency | low-SNR rec | critical rec |
|---:|---|---:|---:|---:|---|---|---|---:|---:|---|---|
| 0 | cache_answer | 0.504 | 0.000 | 0.050 | False | True | False | 0.316 | 0.117 | False | False |
| 1 | semantic_token | 0.355 | 1.188 | 1.300 | False | True | False | 0.465 | 0.000 | False | False |
| 2 | image_evidence | 0.000 | 191.093 | 6.500 | False | False | False | 0.820 | 0.000 | False | False |

## Interpretation for Current Scenarios

- `low_snr_blockage`: image can be semantically strong but deadline-infeasible because payload is large under weak links; semantic tokens are the main lightweight semantic communication route when their LCB clears epsilon.
- `edge_overload`: token candidates are the compact-evidence route because image evidence has high payload and edge workload. Deadline feasibility makes this explicit instead of relying only on payload.
- `critical` risk: controllers should use `accuracy_lcb`, `semantic_quality_gap`, `uncertainty`, `sample_count`, and `joint_feasible` instead of mean accuracy alone.

## Notes

- `is_snr_sensitive=False` for cache because cache reuse does not transmit fresh visual evidence over the current link.
- `recommended_for_low_snr` is a routing prior, not a hard constraint. It favors semantic tokens when their LCB clears epsilon and allows fresh cache only when it is already semantically sufficient.
- `joint_feasible` is the hard semantic-plus-deadline feasibility signal for projection; `semantic_efficiency` should only rank candidates among feasible or nearly feasible options.
