# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.
It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.

- prediction rows: `80271`
- LUT rows: `324`
- overall measured accuracy: `0.582`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator, semantic-token-decoder`
- real VLM present: `yes`

## Latency and Payload Summary

| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |
|---|---:|---:|---:|---:|---:|
| cache | 0.000 | 0.000 | 0.000 | 0.000 | 26757 |
| image | 0.583 | 0.646 | 95.372 | 230.066 | 26757 |
| lightweight | 0.037 | 0.040 | 0.833 | 1.218 | 26757 |

## Payload by Service and Channel

| service | channel | mean payload (KB) | p95 payload (KB) | samples |
|---:|---|---:|---:|---:|
| 0 | bad | 0.000 | 0.000 | 8919 |
| 0 | medium | 0.000 | 0.000 | 8919 |
| 0 | good | 0.000 | 0.000 | 8919 |
| 1 | bad | 0.797 | 1.198 | 8919 |
| 1 | medium | 0.847 | 1.223 | 8919 |
| 1 | good | 0.855 | 1.226 | 8919 |
| 2 | bad | 35.825 | 53.705 | 8919 |
| 2 | medium | 72.199 | 111.280 | 8919 |
| 2 | good | 178.092 | 285.951 | 8919 |

## Semantic-token vs Raw-image Baseline

Here `s=1` is detector-generated semantic-token transmission: classes, boxes, counts, and confidence summaries. `s=2` is raw visual evidence transmission: degraded full-image bytes sent to Qwen-VL. `s=3` is detector-guided ROI/crop image transmission: zoomed target-region bytes sent to Qwen-VL. The comparison therefore tests whether task-aware semantic communication can reduce payload while preserving answer correctness.

| service | evidence baseline | accuracy with 95% CI | mean payload (KB) | p95 payload (KB) | mean latency (s) | samples |
|---:|---|---:|---:|---:|---:|---:|
| 0 | cache answer | 0.562 [0.557, 0.568] | 0.000 | 0.000 | 0.000 | 26757 |
| 1 | detector semantic tokens | 0.599 [0.593, 0.604] | 0.833 | 1.218 | 0.037 | 26757 |
| 2 | raw visual evidence | 0.586 [0.580, 0.592] | 95.372 | 230.066 | 0.583 | 26757 |

### Accuracy by Service and Task Type

| service | evidence baseline | question type | accuracy with 95% CI | samples |
|---:|---|---|---:|---:|
| 0 | cache answer | counting | 0.654 [0.644, 0.664] | 9144 |
| 0 | cache answer | presence | 0.515 [0.507, 0.522] | 17613 |
| 1 | detector semantic tokens | counting | 0.432 [0.423, 0.442] | 9144 |
| 1 | detector semantic tokens | presence | 0.685 [0.678, 0.691] | 17613 |
| 2 | raw visual evidence | counting | 0.275 [0.265, 0.284] | 9144 |
| 2 | raw visual evidence | presence | 0.747 [0.741, 0.754] | 17613 |

## Prediction Distribution

| field | value | count |
|---|---|---:|
| question_type | counting | 27432 |
| question_type | presence | 52839 |
| service_level | 0 | 26757 |
| service_level | 1 | 26757 |
| service_level | 2 | 26757 |
| channel_bin | bad | 26757 |
| channel_bin | good | 26757 |
| channel_bin | medium | 26757 |
| freshness_bin | expired | 26757 |
| freshness_bin | fresh | 26757 |
| freshness_bin | stale | 26757 |
| view_quality_bin | good | 2754 |
| view_quality_bin | medium | 4050 |
| view_quality_bin | poor | 73467 |
| evidence_type | cache | 26757 |
| evidence_type | image | 26757 |
| evidence_type | lightweight | 26757 |

## Accuracy by Service, Channel, and View

### counting

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.665 | 0.520 | 0.431 |
| 0 | medium | 0.676 | 0.500 | 0.412 |
| 0 | good | 0.665 | 0.587 | 0.461 |
| 1 | bad | 0.319 | 0.420 | 0.647 |
| 1 | medium | 0.444 | 0.640 | 0.882 |
| 1 | good | 0.472 | 0.660 | 0.853 |
| 2 | bad | 0.245 | 0.440 | 0.618 |
| 2 | medium | 0.254 | 0.420 | 0.647 |
| 2 | good | 0.262 | 0.440 | 0.618 |

### presence

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.506 | 0.513 | 0.525 |
| 0 | medium | 0.515 | 0.513 | 0.544 |
| 0 | good | 0.521 | 0.540 | 0.480 |
| 1 | bad | 0.638 | 0.810 | 0.750 |
| 1 | medium | 0.684 | 0.880 | 0.868 |
| 1 | good | 0.686 | 0.870 | 0.897 |
| 2 | bad | 0.722 | 0.840 | 0.882 |
| 2 | medium | 0.735 | 0.840 | 0.941 |
| 2 | good | 0.750 | 0.850 | 0.941 |

## Paper Table: Resource Policies

| policy | success | accuracy | delay | energy | payload KB | payload reduction | cache | light | image | roi | success 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.262 | 0.548 | 0.688 | 0.200 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | [0.252, 0.272] |
| always_light | 0.202 | 0.607 | 2.075 | 1.000 | 0.841 | 0.991 | 0.000 | 1.000 | 0.000 | 0.000 | [0.193, 0.211] |
| always_image | 0.417 | 0.596 | 3.909 | 2.500 | 95.560 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | [0.406, 0.428] |
| greedy_min_sufficient_evidence | 0.697 | 0.700 | 2.743 | 1.618 | 50.613 | 0.470 | 0.258 | 0.193 | 0.549 | 0.000 | [0.687, 0.707] |
| no_cache_greedy | 0.571 | 0.592 | 3.533 | 2.196 | 76.595 | 0.198 | 0.000 | 0.203 | 0.797 | 0.000 | [0.560, 0.582] |
| no_semantic_tokens_greedy | 0.544 | 0.704 | 3.069 | 1.897 | 69.603 | 0.272 | 0.262 | 0.000 | 0.738 | 0.000 | [0.533, 0.555] |
| oracle_best_feasible_evidence | 0.697 | 0.715 | 2.266 | 1.252 | 30.782 | 0.678 | 0.351 | 0.294 | 0.355 | 0.000 | [0.687, 0.707] |

## Failure Taxonomy

| failure type | count | share among failures |
|---|---:|---:|
| image_vlm_counting_undercount | 6624 | 0.198 |
| detector_semantic_presence_false_negative | 4944 | 0.147 |
| presence_false_negative | 4761 | 0.142 |
| presence_false_positive | 3788 | 0.113 |
| detector_semantic_counting_undercount | 3723 | 0.111 |
| image_vlm_presence_false_negative | 3168 | 0.094 |
| counting_overcount | 3163 | 0.094 |
| detector_semantic_counting_overcount | 1467 | 0.044 |
| image_vlm_presence_false_positive | 1281 | 0.038 |
| detector_semantic_presence_false_positive | 606 | 0.018 |
| image_vlm_counting_overcount | 6 | 0.000 |

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
