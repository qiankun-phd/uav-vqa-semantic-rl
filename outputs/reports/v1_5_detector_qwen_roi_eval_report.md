# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.
It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.

- prediction rows: `14184`
- LUT rows: `432`
- overall measured accuracy: `0.636`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator`
- real VLM present: `yes`

## Latency and Payload Summary

| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |
|---|---:|---:|---:|---:|---:|
| cache | 0.000 | 0.000 | 0.000 | 0.000 | 3546 |
| image | 0.509 | 0.679 | 83.709 | 228.236 | 3546 |
| lightweight | 0.104 | 0.182 | 0.846 | 1.238 | 3546 |
| roi_image | 0.353 | 0.656 | 50.033 | 181.978 | 3546 |

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
| 3 | bad | 17.675 | 72.882 | 1182 |
| 3 | medium | 35.850 | 154.137 | 1182 |
| 3 | good | 96.573 | 404.587 | 1182 |

## Semantic-token vs Raw-image Baseline

Here `s=1` is detector-generated semantic-token transmission: classes, boxes, counts, and confidence summaries. `s=2` is raw visual evidence transmission: degraded full-image bytes sent to Qwen-VL. `s=3` is detector-guided ROI/crop image transmission: zoomed target-region bytes sent to Qwen-VL. The comparison therefore tests whether task-aware semantic communication can reduce payload while preserving answer correctness.

| service | evidence baseline | accuracy with 95% CI | mean payload (KB) | p95 payload (KB) | mean latency (s) | samples |
|---:|---|---:|---:|---:|---:|---:|
| 0 | cache answer | 0.536 [0.521, 0.553] | 0.000 | 0.000 | 0.000 | 3546 |
| 1 | detector semantic tokens | 0.747 [0.733, 0.762] | 0.846 | 1.238 | 0.104 | 3546 |
| 2 | raw visual evidence | 0.654 [0.638, 0.670] | 83.709 | 228.236 | 0.509 | 3546 |
| 3 | detector ROI crop image | 0.608 [0.593, 0.625] | 50.033 | 181.978 | 0.353 | 3546 |

### Accuracy by Service and Task Type

| service | evidence baseline | question type | accuracy with 95% CI | samples |
|---:|---|---|---:|---:|
| 0 | cache answer | counting | 0.538 [0.512, 0.567] | 1323 |
| 0 | cache answer | presence | 0.535 [0.515, 0.554] | 2223 |
| 1 | detector semantic tokens | counting | 0.322 [0.295, 0.348] | 1323 |
| 1 | detector semantic tokens | presence | 1.000 [1.000, 1.000] | 2223 |
| 2 | raw visual evidence | counting | 0.420 [0.394, 0.444] | 1323 |
| 2 | raw visual evidence | presence | 0.794 [0.776, 0.810] | 2223 |
| 3 | detector ROI crop image | counting | 0.395 [0.367, 0.420] | 1323 |
| 3 | detector ROI crop image | presence | 0.735 [0.717, 0.753] | 2223 |

## Prediction Distribution

| field | value | count |
|---|---|---:|
| question_type | counting | 5292 |
| question_type | presence | 8892 |
| service_level | 0 | 3546 |
| service_level | 1 | 3546 |
| service_level | 2 | 3546 |
| service_level | 3 | 3546 |
| channel_bin | bad | 4728 |
| channel_bin | good | 4728 |
| channel_bin | medium | 4728 |
| freshness_bin | expired | 4728 |
| freshness_bin | fresh | 4728 |
| freshness_bin | stale | 4728 |
| view_quality_bin | good | 5544 |
| view_quality_bin | medium | 2880 |
| view_quality_bin | poor | 5760 |
| evidence_type | cache | 3546 |
| evidence_type | image | 3546 |
| evidence_type | lightweight | 3546 |
| evidence_type | roi_image | 3546 |

## Accuracy by Service, Channel, and View

### counting

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.503 | 0.570 | 0.536 |
| 0 | medium | 0.521 | 0.602 | 0.541 |
| 0 | good | 0.473 | 0.624 | 0.552 |
| 1 | bad | 0.145 | 0.129 | 0.426 |
| 1 | medium | 0.291 | 0.258 | 0.492 |
| 1 | good | 0.255 | 0.226 | 0.475 |
| 2 | bad | 0.382 | 0.323 | 0.475 |
| 2 | medium | 0.400 | 0.323 | 0.492 |
| 2 | good | 0.418 | 0.323 | 0.492 |
| 3 | bad | 0.418 | 0.323 | 0.410 |
| 3 | medium | 0.418 | 0.323 | 0.410 |
| 3 | good | 0.418 | 0.290 | 0.426 |

### presence

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.530 | 0.429 | 0.566 |
| 0 | medium | 0.514 | 0.571 | 0.577 |
| 0 | good | 0.556 | 0.497 | 0.523 |
| 1 | bad | 1.000 | 1.000 | 1.000 |
| 1 | medium | 1.000 | 1.000 | 1.000 |
| 1 | good | 1.000 | 1.000 | 1.000 |
| 2 | bad | 0.686 | 0.714 | 0.796 |
| 2 | medium | 0.743 | 0.776 | 0.860 |
| 2 | good | 0.810 | 0.857 | 0.903 |
| 3 | bad | 0.629 | 0.694 | 0.742 |
| 3 | medium | 0.695 | 0.735 | 0.785 |
| 3 | good | 0.752 | 0.796 | 0.817 |

## Paper Table: Resource Policies

| policy | success | accuracy | delay | energy | payload KB | payload reduction | cache | light | image | roi | success 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.265 | 0.558 | 0.688 | 0.200 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | [0.251, 0.279] |
| always_light | 0.521 | 0.660 | 2.061 | 1.000 | 0.871 | 0.990 | 0.000 | 1.000 | 0.000 | 0.000 | [0.506, 0.536] |
| always_image | 0.324 | 0.574 | 3.916 | 2.500 | 85.446 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | [0.310, 0.339] |
| always_roi | 0.299 | 0.540 | 2.969 | 1.800 | 52.604 | 0.384 | 0.000 | 0.000 | 0.000 | 1.000 | [0.285, 0.314] |
| greedy_min_sufficient_evidence | 0.729 | 0.756 | 2.322 | 1.261 | 24.247 | 0.716 | 0.278 | 0.399 | 0.323 | 0.000 | [0.715, 0.743] |
| no_cache_greedy | 0.584 | 0.683 | 2.955 | 1.718 | 39.592 | 0.537 | 0.000 | 0.521 | 0.479 | 0.000 | [0.568, 0.599] |
| no_semantic_tokens_greedy | 0.597 | 0.668 | 2.853 | 1.694 | 49.491 | 0.421 | 0.265 | 0.000 | 0.455 | 0.280 | [0.582, 0.612] |
| no_roi_greedy | 0.730 | 0.757 | 2.305 | 1.265 | 25.220 | 0.705 | 0.276 | 0.401 | 0.324 | 0.000 | [0.716, 0.743] |
| oracle_best_feasible_evidence | 0.727 | 0.808 | 1.772 | 0.892 | 11.494 | 0.865 | 0.413 | 0.393 | 0.095 | 0.099 | [0.713, 0.740] |

## Failure Taxonomy

| failure type | count | share among failures |
|---|---:|---:|
| presence_false_negative | 1034 | 0.200 |
| roi_image_vlm_counting_undercount | 801 | 0.155 |
| image_vlm_counting_undercount | 765 | 0.148 |
| counting_overcount | 611 | 0.118 |
| roi_image_vlm_presence_false_negative | 588 | 0.114 |
| image_vlm_presence_false_negative | 459 | 0.089 |
| detector_semantic_counting_undercount | 435 | 0.084 |
| detector_semantic_counting_invalid_answer | 366 | 0.071 |
| detector_semantic_counting_overcount | 96 | 0.019 |
| image_vlm_counting_overcount | 3 | 0.001 |

## Example Failures

- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=3 channel=bad payload=106.18KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=3 channel=medium payload=228.42KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=1 channel=bad payload=0.71KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `yes`
- `0000001_02999_d_0000005` s=1 channel=medium payload=0.72KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `yes`
- `0000001_02999_d_0000005` s=1 channel=good payload=0.72KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `yes`
- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=good payload=570.06KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=3 channel=bad payload=106.18KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=3 channel=medium payload=228.42KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
