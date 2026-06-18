# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.
It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.

- prediction rows: `53514`
- LUT rows: `324`
- overall measured accuracy: `0.617`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator`
- real VLM present: `yes`

## Latency and Payload Summary

| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |
|---|---:|---:|---:|---:|---:|
| cache | 0.000 | 0.000 | 0.000 | 0.000 | 17838 |
| image | 0.583 | 0.647 | 95.372 | 230.066 | 17838 |
| lightweight | 0.081 | 0.097 | 0.560 | 0.773 | 17838 |

## Payload by Service and Channel

| service | channel | mean payload (KB) | p95 payload (KB) | samples |
|---:|---|---:|---:|---:|
| 0 | bad | 0.000 | 0.000 | 5946 |
| 0 | medium | 0.000 | 0.000 | 5946 |
| 0 | good | 0.000 | 0.000 | 5946 |
| 1 | bad | 0.529 | 0.757 | 5946 |
| 1 | medium | 0.573 | 0.777 | 5946 |
| 1 | good | 0.579 | 0.782 | 5946 |
| 2 | bad | 35.825 | 53.705 | 5946 |
| 2 | medium | 72.199 | 111.280 | 5946 |
| 2 | good | 178.092 | 285.951 | 5946 |

## Semantic-token vs Raw-image Baseline

Here `s=1` is detector-generated semantic-token transmission: classes, boxes, counts, and confidence summaries. `s=2` is raw visual evidence transmission: degraded image bytes sent to Qwen-VL. The comparison therefore tests whether task-aware semantic communication can reduce payload while preserving answer correctness.

| service | evidence baseline | accuracy with 95% CI | mean payload (KB) | p95 payload (KB) | mean latency (s) | samples |
|---:|---|---:|---:|---:|---:|---:|
| 0 | cache answer | 0.549 [0.542, 0.556] | 0.000 | 0.000 | 0.000 | 17838 |
| 1 | detector semantic tokens | 0.698 [0.692, 0.705] | 0.560 | 0.773 | 0.081 | 17838 |
| 2 | raw visual evidence | 0.604 [0.597, 0.611] | 95.372 | 230.066 | 0.583 | 17838 |

### Accuracy by Service and Task Type

| service | evidence baseline | question type | accuracy with 95% CI | samples |
|---:|---|---|---:|---:|
| 0 | cache answer | counting | 0.619 [0.607, 0.632] | 5967 |
| 0 | cache answer | presence | 0.514 [0.505, 0.523] | 11871 |
| 1 | detector semantic tokens | counting | 0.247 [0.236, 0.258] | 5967 |
| 1 | detector semantic tokens | presence | 0.925 [0.920, 0.930] | 11871 |
| 2 | raw visual evidence | counting | 0.350 [0.339, 0.362] | 5967 |
| 2 | raw visual evidence | presence | 0.731 [0.723, 0.739] | 11871 |

## Prediction Distribution

| field | value | count |
|---|---|---:|
| question_type | counting | 17901 |
| question_type | presence | 35613 |
| service_level | 0 | 17838 |
| service_level | 1 | 17838 |
| service_level | 2 | 17838 |
| channel_bin | bad | 17838 |
| channel_bin | good | 17838 |
| channel_bin | medium | 17838 |
| freshness_bin | expired | 17838 |
| freshness_bin | fresh | 17838 |
| freshness_bin | stale | 17838 |
| view_quality_bin | good | 17712 |
| view_quality_bin | medium | 17928 |
| view_quality_bin | poor | 17874 |
| evidence_type | cache | 17838 |
| evidence_type | image | 17838 |
| evidence_type | lightweight | 17838 |

## Accuracy by Service, Channel, and View

### counting

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.559 | 0.668 | 0.621 |
| 0 | medium | 0.617 | 0.648 | 0.605 |
| 0 | good | 0.570 | 0.645 | 0.645 |
| 1 | bad | 0.229 | 0.164 | 0.269 |
| 1 | medium | 0.215 | 0.188 | 0.339 |
| 1 | good | 0.251 | 0.207 | 0.348 |
| 2 | bad | 0.309 | 0.324 | 0.383 |
| 2 | medium | 0.336 | 0.333 | 0.383 |
| 2 | good | 0.359 | 0.329 | 0.392 |

### presence

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.500 | 0.510 | 0.524 |
| 0 | medium | 0.518 | 0.505 | 0.523 |
| 0 | good | 0.532 | 0.492 | 0.524 |
| 1 | bad | 0.852 | 0.898 | 0.888 |
| 1 | medium | 0.929 | 0.953 | 0.956 |
| 1 | good | 0.932 | 0.956 | 0.963 |
| 2 | bad | 0.626 | 0.681 | 0.767 |
| 2 | medium | 0.667 | 0.723 | 0.797 |
| 2 | good | 0.724 | 0.756 | 0.844 |

## Paper Table: Resource Policies

| policy | success | accuracy | delay | energy | payload KB | payload reduction | cache | light | image | success 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.283 | 0.571 | 0.689 | 0.200 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | [0.273, 0.293] |
| always_light | 0.519 | 0.595 | 2.073 | 1.000 | 0.560 | 0.994 | 0.000 | 1.000 | 0.000 | [0.509, 0.530] |
| always_image | 0.100 | 0.540 | 3.902 | 2.500 | 94.680 | 0.000 | 0.000 | 0.000 | 1.000 | [0.093, 0.107] |
| greedy_min_sufficient_evidence | 0.692 | 0.778 | 2.251 | 1.239 | 28.680 | 0.697 | 0.279 | 0.412 | 0.308 | [0.682, 0.702] |
| oracle_best_feasible_evidence | 0.692 | 0.787 | 1.948 | 1.028 | 21.986 | 0.768 | 0.370 | 0.414 | 0.216 | [0.682, 0.702] |

## Failure Taxonomy

| failure type | count | share among failures |
|---|---:|---:|
| presence_false_negative | 5769 | 0.282 |
| image_vlm_counting_undercount | 3873 | 0.189 |
| image_vlm_presence_false_negative | 3192 | 0.156 |
| detector_semantic_counting_undercount | 2907 | 0.142 |
| counting_overcount | 2272 | 0.111 |
| detector_semantic_counting_invalid_answer | 1464 | 0.071 |
| detector_semantic_presence_false_negative | 888 | 0.043 |
| detector_semantic_counting_overcount | 123 | 0.006 |
| image_vlm_counting_overcount | 3 | 0.000 |

## Example Failures

- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=1 channel=bad payload=0.45KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `1`
- `0000001_02999_d_0000005` s=1 channel=medium payload=0.46KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `no`
- `0000001_02999_d_0000005` s=1 channel=good payload=0.46KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `no`
- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=good payload=570.06KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_03499_d_0000006` s=1 channel=bad payload=0.45KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `no`
- `0000001_03499_d_0000006` s=2 channel=bad payload=103.52KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_03499_d_0000006` s=2 channel=medium payload=215.95KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_03499_d_0000006` s=1 channel=medium payload=0.47KB Q: How many awning-tricycle objects are in this area? GT: `2` Pred: `no`
