# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.
It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.

- prediction rows: `80811`
- LUT rows: `360`
- overall measured accuracy: `0.582`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator, semantic-token-decoder`
- real VLM present: `yes`

## Latency and Payload Summary

| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |
|---|---:|---:|---:|---:|---:|
| cache | 0.000 | 0.000 | 0.000 | 0.000 | 26757 |
| image | 0.583 | 0.646 | 95.372 | 230.066 | 26757 |
| lightweight | 0.037 | 0.040 | 0.833 | 1.218 | 26757 |
| roi_image | 0.525 | 0.684 | 168.784 | 522.301 | 540 |

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
| 3 | bad | 60.635 | 106.180 | 180 |
| 3 | medium | 125.796 | 228.417 | 180 |
| 3 | good | 319.921 | 570.063 | 180 |

## Semantic-token vs Raw-image Baseline

Here `s=1` is detector-generated semantic-token transmission: classes, boxes, counts, and confidence summaries. `s=2` is raw visual evidence transmission: degraded full-image bytes sent to Qwen-VL. `s=3` is detector-guided ROI/crop image transmission: zoomed target-region bytes sent to Qwen-VL. The comparison therefore tests whether task-aware semantic communication can reduce payload while preserving answer correctness.

| service | evidence baseline | accuracy with 95% CI | mean payload (KB) | p95 payload (KB) | mean latency (s) | samples |
|---:|---|---:|---:|---:|---:|---:|
| 0 | cache answer | 0.562 [0.557, 0.568] | 0.000 | 0.000 | 0.000 | 26757 |
| 1 | detector semantic tokens | 0.599 [0.593, 0.604] | 0.833 | 1.218 | 0.037 | 26757 |
| 2 | raw visual evidence | 0.586 [0.580, 0.592] | 95.372 | 230.066 | 0.583 | 26757 |
| 3 | detector ROI crop image | 0.567 [0.528, 0.607] | 168.784 | 522.301 | 0.525 | 540 |

### Accuracy by Service and Task Type

| service | evidence baseline | question type | accuracy with 95% CI | samples |
|---:|---|---|---:|---:|
| 0 | cache answer | counting | 0.654 [0.644, 0.664] | 9144 |
| 0 | cache answer | presence | 0.515 [0.507, 0.522] | 17613 |
| 1 | detector semantic tokens | counting | 0.432 [0.423, 0.442] | 9144 |
| 1 | detector semantic tokens | presence | 0.685 [0.678, 0.691] | 17613 |
| 2 | raw visual evidence | counting | 0.275 [0.265, 0.284] | 9144 |
| 2 | raw visual evidence | presence | 0.747 [0.741, 0.754] | 17613 |
| 3 | detector ROI crop image | counting | 0.150 [0.100, 0.200] | 180 |
| 3 | detector ROI crop image | presence | 0.775 [0.733, 0.817] | 360 |

## Prediction Distribution

| field | value | count |
|---|---|---:|
| question_type | counting | 27612 |
| question_type | presence | 53199 |
| service_level | 0 | 26757 |
| service_level | 1 | 26757 |
| service_level | 2 | 26757 |
| service_level | 3 | 540 |
| channel_bin | bad | 26937 |
| channel_bin | good | 26937 |
| channel_bin | medium | 26937 |
| freshness_bin | expired | 26937 |
| freshness_bin | fresh | 26937 |
| freshness_bin | stale | 26937 |
| view_quality_bin | good | 2754 |
| view_quality_bin | medium | 4050 |
| view_quality_bin | poor | 74007 |
| evidence_type | cache | 26757 |
| evidence_type | image | 26757 |
| evidence_type | lightweight | 26757 |
| evidence_type | roi_image | 540 |

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
| 3 | bad | 0.150 | - | - |
| 3 | medium | 0.150 | - | - |
| 3 | good | 0.150 | - | - |

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
| 3 | bad | 0.800 | - | - |
| 3 | medium | 0.750 | - | - |
| 3 | good | 0.775 | - | - |

## Failure Taxonomy

| failure type | count | share among failures |
|---|---:|---:|
| image_vlm_counting_undercount | 6624 | 0.196 |
| detector_semantic_presence_false_negative | 4944 | 0.146 |
| presence_false_negative | 4761 | 0.141 |
| presence_false_positive | 3788 | 0.112 |
| detector_semantic_counting_undercount | 3723 | 0.110 |
| image_vlm_presence_false_negative | 3168 | 0.094 |
| counting_overcount | 3163 | 0.094 |
| detector_semantic_counting_overcount | 1467 | 0.043 |
| image_vlm_presence_false_positive | 1281 | 0.038 |
| detector_semantic_presence_false_positive | 606 | 0.018 |
| roi_image_vlm_counting_undercount | 153 | 0.005 |
| roi_image_vlm_presence_false_negative | 57 | 0.002 |
| roi_image_vlm_presence_false_positive | 24 | 0.001 |
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
