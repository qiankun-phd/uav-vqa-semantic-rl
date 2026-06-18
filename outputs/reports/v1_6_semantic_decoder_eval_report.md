# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.
It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.

- prediction rows: `10638`
- LUT rows: `324`
- overall measured accuracy: `0.615`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator, semantic-token-decoder`
- real VLM present: `yes`

## Latency and Payload Summary

| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |
|---|---:|---:|---:|---:|---:|
| cache | 0.000 | 0.000 | 0.000 | 0.000 | 3546 |
| image | 0.510 | 0.677 | 83.709 | 228.236 | 3546 |
| lightweight | 0.041 | 0.044 | 0.846 | 1.238 | 3546 |

## Payload by Service and Channel

| service | channel | mean payload (KB) | p95 payload (KB) | samples |
|---:|---|---:|---:|---:|
| 0 | bad | 0.000 | 0.000 | 1182 |
| 0 | medium | 0.000 | 0.000 | 1182 |
| 0 | good | 0.000 | 0.000 | 1182 |
| 1 | bad | 0.787 | 1.179 | 1182 |
| 1 | medium | 0.869 | 1.259 | 1182 |
| 1 | good | 0.884 | 1.266 | 1182 |
| 2 | bad | 31.175 | 94.894 | 1182 |
| 2 | medium | 63.662 | 190.285 | 1182 |
| 2 | good | 156.288 | 460.813 | 1182 |

## Semantic-token vs Raw-image Baseline

Here `s=1` is detector-generated semantic-token transmission: classes, boxes, counts, and confidence summaries. `s=2` is raw visual evidence transmission: degraded full-image bytes sent to Qwen-VL. `s=3` is detector-guided ROI/crop image transmission: zoomed target-region bytes sent to Qwen-VL. The comparison therefore tests whether task-aware semantic communication can reduce payload while preserving answer correctness.

| service | evidence baseline | accuracy with 95% CI | mean payload (KB) | p95 payload (KB) | mean latency (s) | samples |
|---:|---|---:|---:|---:|---:|---:|
| 0 | cache answer | 0.536 [0.521, 0.553] | 0.000 | 0.000 | 0.000 | 3546 |
| 1 | detector semantic tokens | 0.655 [0.639, 0.672] | 0.846 | 1.238 | 0.041 | 3546 |
| 2 | raw visual evidence | 0.654 [0.638, 0.670] | 83.709 | 228.236 | 0.510 | 3546 |

### Accuracy by Service and Task Type

| service | evidence baseline | question type | accuracy with 95% CI | samples |
|---:|---|---|---:|---:|
| 0 | cache answer | counting | 0.538 [0.512, 0.567] | 1323 |
| 0 | cache answer | presence | 0.535 [0.515, 0.554] | 2223 |
| 1 | detector semantic tokens | counting | 0.599 [0.571, 0.623] | 1323 |
| 1 | detector semantic tokens | presence | 0.688 [0.668, 0.707] | 2223 |
| 2 | raw visual evidence | counting | 0.420 [0.394, 0.444] | 1323 |
| 2 | raw visual evidence | presence | 0.794 [0.776, 0.810] | 2223 |

## Prediction Distribution

| field | value | count |
|---|---|---:|
| question_type | counting | 3969 |
| question_type | presence | 6669 |
| service_level | 0 | 3546 |
| service_level | 1 | 3546 |
| service_level | 2 | 3546 |
| channel_bin | bad | 3546 |
| channel_bin | good | 3546 |
| channel_bin | medium | 3546 |
| freshness_bin | expired | 3546 |
| freshness_bin | fresh | 3546 |
| freshness_bin | stale | 3546 |
| view_quality_bin | good | 4158 |
| view_quality_bin | medium | 2160 |
| view_quality_bin | poor | 4320 |
| evidence_type | cache | 3546 |
| evidence_type | image | 3546 |
| evidence_type | lightweight | 3546 |

## Accuracy by Service, Channel, and View

### counting

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.503 | 0.570 | 0.536 |
| 0 | medium | 0.521 | 0.602 | 0.541 |
| 0 | good | 0.473 | 0.624 | 0.552 |
| 1 | bad | 0.564 | 0.484 | 0.689 |
| 1 | medium | 0.564 | 0.484 | 0.689 |
| 1 | good | 0.564 | 0.484 | 0.689 |
| 2 | bad | 0.382 | 0.323 | 0.475 |
| 2 | medium | 0.400 | 0.323 | 0.492 |
| 2 | good | 0.418 | 0.323 | 0.492 |

### presence

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.530 | 0.429 | 0.566 |
| 0 | medium | 0.514 | 0.571 | 0.577 |
| 0 | good | 0.556 | 0.497 | 0.523 |
| 1 | bad | 0.638 | 0.673 | 0.753 |
| 1 | medium | 0.638 | 0.673 | 0.753 |
| 1 | good | 0.638 | 0.673 | 0.753 |
| 2 | bad | 0.686 | 0.714 | 0.796 |
| 2 | medium | 0.743 | 0.776 | 0.860 |
| 2 | good | 0.810 | 0.857 | 0.903 |

## Paper Table: Resource Policies

| policy | success | accuracy | delay | energy | payload KB | payload reduction | cache | light | image | roi | success 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.265 | 0.558 | 0.688 | 0.200 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | [0.251, 0.279] |
| always_light | 0.362 | 0.617 | 2.061 | 1.000 | 0.871 | 0.990 | 0.000 | 1.000 | 0.000 | 0.000 | [0.347, 0.377] |
| always_image | 0.324 | 0.574 | 3.916 | 2.500 | 85.446 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | [0.310, 0.339] |
| always_roi | 0.000 | 0.000 | 2.969 | 1.800 | 78.125 | 0.086 | 0.000 | 0.000 | 0.000 | 1.000 | [0.000, 0.000] |
| greedy_min_sufficient_evidence | 0.686 | 0.683 | 2.493 | 1.389 | 29.267 | 0.657 | 0.278 | 0.314 | 0.408 | 0.000 | [0.671, 0.700] |
| no_cache_greedy | 0.530 | 0.574 | 3.248 | 1.957 | 51.748 | 0.394 | 0.000 | 0.362 | 0.638 | 0.000 | [0.515, 0.546] |
| no_semantic_tokens_greedy | 0.492 | 0.681 | 3.105 | 1.890 | 60.608 | 0.291 | 0.265 | 0.000 | 0.735 | 0.000 | [0.477, 0.508] |
| no_roi_greedy | 0.693 | 0.684 | 2.468 | 1.388 | 29.927 | 0.650 | 0.276 | 0.319 | 0.406 | 0.000 | [0.678, 0.707] |
| oracle_best_feasible_evidence | 0.697 | 0.756 | 1.898 | 0.962 | 12.645 | 0.852 | 0.353 | 0.483 | 0.164 | 0.000 | [0.683, 0.711] |

## Failure Taxonomy

| failure type | count | share among failures |
|---|---:|---:|
| presence_false_negative | 1034 | 0.252 |
| image_vlm_counting_undercount | 765 | 0.187 |
| detector_semantic_presence_false_negative | 693 | 0.169 |
| counting_overcount | 611 | 0.149 |
| image_vlm_presence_false_negative | 459 | 0.112 |
| detector_semantic_counting_undercount | 405 | 0.099 |
| detector_semantic_counting_overcount | 126 | 0.031 |
| image_vlm_counting_overcount | 3 | 0.001 |

## Example Failures

- `0000001_02999_d_0000005` s=1 channel=bad payload=0.71KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `no`
- `0000001_02999_d_0000005` s=1 channel=medium payload=0.72KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `no`
- `0000001_02999_d_0000005` s=1 channel=good payload=0.72KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `no`
- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=1 channel=bad payload=0.71KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=1 channel=medium payload=0.72KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=1 channel=good payload=0.72KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=good payload=570.06KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=1 channel=bad payload=0.70KB Q: Are there bicycle objects in this area? GT: `yes` Pred: `no`
