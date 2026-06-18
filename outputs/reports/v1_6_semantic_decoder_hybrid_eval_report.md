# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.
It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.

- prediction rows: `53514`
- LUT rows: `324`
- overall measured accuracy: `0.667`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator, semantic-token-decoder`
- real VLM present: `yes`

## Latency and Payload Summary

| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |
|---|---:|---:|---:|---:|---:|
| cache | 0.000 | 0.000 | 0.000 | 0.000 | 17838 |
| image | 0.585 | 0.649 | 95.372 | 230.066 | 17838 |
| lightweight | 0.087 | 0.190 | 0.875 | 1.223 | 17838 |

## Payload by Service and Channel

| service | channel | mean payload (KB) | p95 payload (KB) | samples |
|---:|---|---:|---:|---:|
| 0 | bad | 0.000 | 0.000 | 5946 |
| 0 | medium | 0.000 | 0.000 | 5946 |
| 0 | good | 0.000 | 0.000 | 5946 |
| 1 | bad | 0.829 | 1.202 | 5946 |
| 1 | medium | 0.893 | 1.226 | 5946 |
| 1 | good | 0.902 | 1.230 | 5946 |
| 2 | bad | 35.825 | 53.705 | 5946 |
| 2 | medium | 72.199 | 111.280 | 5946 |
| 2 | good | 178.092 | 285.951 | 5946 |

## Semantic-token vs Raw-image Baseline

Here `s=1` is detector-generated semantic-token transmission: classes, boxes, counts, and confidence summaries. `s=2` is raw visual evidence transmission: degraded full-image bytes sent to Qwen-VL. `s=3` is detector-guided ROI/crop image transmission: zoomed target-region bytes sent to Qwen-VL. The comparison therefore tests whether task-aware semantic communication can reduce payload while preserving answer correctness.

| service | evidence baseline | accuracy with 95% CI | mean payload (KB) | p95 payload (KB) | mean latency (s) | samples |
|---:|---|---:|---:|---:|---:|---:|
| 0 | cache answer | 0.549 [0.542, 0.556] | 0.000 | 0.000 | 0.000 | 17838 |
| 1 | detector semantic tokens | 0.847 [0.841, 0.852] | 0.875 | 1.223 | 0.087 | 17838 |
| 2 | raw visual evidence | 0.604 [0.597, 0.611] | 95.372 | 230.066 | 0.585 | 17838 |

### Accuracy by Service and Task Type

| service | evidence baseline | question type | accuracy with 95% CI | samples |
|---:|---|---|---:|---:|
| 0 | cache answer | counting | 0.619 [0.607, 0.632] | 5967 |
| 0 | cache answer | presence | 0.514 [0.505, 0.523] | 11871 |
| 1 | detector semantic tokens | counting | 0.541 [0.529, 0.554] | 5967 |
| 1 | detector semantic tokens | presence | 1.000 [1.000, 1.000] | 11871 |
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
| 1 | bad | 0.471 | 0.535 | 0.617 |
| 1 | medium | 0.471 | 0.535 | 0.617 |
| 1 | good | 0.471 | 0.535 | 0.617 |
| 2 | bad | 0.309 | 0.324 | 0.383 |
| 2 | medium | 0.336 | 0.333 | 0.383 |
| 2 | good | 0.359 | 0.329 | 0.392 |

### presence

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.500 | 0.510 | 0.524 |
| 0 | medium | 0.518 | 0.505 | 0.523 |
| 0 | good | 0.532 | 0.492 | 0.524 |
| 1 | bad | 1.000 | 1.000 | 1.000 |
| 1 | medium | 1.000 | 1.000 | 1.000 |
| 1 | good | 1.000 | 1.000 | 1.000 |
| 2 | bad | 0.626 | 0.681 | 0.767 |
| 2 | medium | 0.667 | 0.723 | 0.797 |
| 2 | good | 0.724 | 0.756 | 0.844 |

## Paper Table: Resource Policies

| policy | success | accuracy | delay | energy | payload KB | payload reduction | cache | light | image | roi | success 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.277 | 0.566 | 0.689 | 0.200 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | [0.267, 0.286] |
| always_light | 0.625 | 0.781 | 2.069 | 1.000 | 0.874 | 0.991 | 0.000 | 1.000 | 0.000 | 0.000 | [0.614, 0.635] |
| always_image | 0.108 | 0.542 | 3.909 | 2.500 | 95.153 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | [0.101, 0.114] |
| greedy_min_sufficient_evidence | 0.794 | 0.816 | 2.076 | 1.087 | 19.412 | 0.796 | 0.277 | 0.517 | 0.206 | 0.000 | [0.785, 0.803] |
| no_cache_greedy | 0.625 | 0.698 | 2.759 | 1.563 | 35.516 | 0.627 | 0.000 | 0.625 | 0.376 | 0.000 | [0.614, 0.635] |
| no_semantic_tokens_greedy | 0.365 | 0.704 | 3.006 | 1.847 | 66.854 | 0.297 | 0.284 | 0.000 | 0.716 | 0.000 | [0.355, 0.376] |
| oracle_best_feasible_evidence | 0.794 | 0.834 | 1.634 | 0.742 | 0.567 | 0.994 | 0.323 | 0.677 | 0.000 | 0.000 | [0.785, 0.803] |

## Failure Taxonomy

| failure type | count | share among failures |
|---|---:|---:|
| presence_false_negative | 5769 | 0.323 |
| image_vlm_counting_undercount | 3873 | 0.217 |
| image_vlm_presence_false_negative | 3192 | 0.179 |
| counting_overcount | 2272 | 0.127 |
| detector_semantic_counting_undercount | 1926 | 0.108 |
| detector_semantic_counting_overcount | 810 | 0.045 |
| image_vlm_counting_overcount | 3 | 0.000 |

## Example Failures

- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=1 channel=bad payload=0.71KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=1 channel=medium payload=0.72KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=1 channel=good payload=0.72KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=good payload=570.06KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_03499_d_0000006` s=2 channel=bad payload=103.52KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_03499_d_0000006` s=2 channel=medium payload=215.95KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_03499_d_0000006` s=1 channel=bad payload=0.71KB Q: How many awning-tricycle objects are in this area? GT: `2` Pred: `0`
- `0000001_03499_d_0000006` s=1 channel=medium payload=0.73KB Q: How many awning-tricycle objects are in this area? GT: `2` Pred: `0`
