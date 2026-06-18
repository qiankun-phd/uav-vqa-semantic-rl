# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.

- prediction rows: `540`
- LUT rows: `270`
- overall measured accuracy: `0.761`
- model names: `cache-simulator, mock-vlm`
- real VLM present: `no`

## Latency Summary

| evidence type | mean latency (s) | p95 latency (s) | samples |
|---|---:|---:|---:|
| cache | 0.000 | 0.000 | 180 |
| image | 0.000 | 0.000 | 180 |
| lightweight | 0.000 | 0.000 | 180 |

## Prediction Distribution

| field | value | count |
|---|---|---:|
| question_type | counting | 162 |
| question_type | presence | 378 |
| service_level | 0 | 180 |
| service_level | 1 | 180 |
| service_level | 2 | 180 |
| channel_bin | bad | 180 |
| channel_bin | good | 180 |
| channel_bin | medium | 180 |
| freshness_bin | expired | 180 |
| freshness_bin | fresh | 180 |
| freshness_bin | stale | 180 |
| view_quality_bin | good | 216 |
| view_quality_bin | medium | 216 |
| view_quality_bin | poor | 108 |
| evidence_type | cache | 180 |
| evidence_type | image | 180 |
| evidence_type | lightweight | 180 |

## Accuracy by Service, Channel, and View

### counting

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.667 | 0.833 | 0.778 |
| 0 | medium | 0.333 | 0.667 | 0.778 |
| 0 | good | 0.667 | 1.000 | 0.778 |
| 1 | bad | 1.000 | 1.000 | 1.000 |
| 1 | medium | 1.000 | 1.000 | 1.000 |
| 1 | good | 0.000 | 0.500 | 0.667 |
| 2 | bad | 1.000 | 0.500 | 0.667 |
| 2 | medium | 1.000 | 1.000 | 1.000 |
| 2 | good | 1.000 | 1.000 | 1.000 |

### presence

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.444 | 0.611 | 0.533 |
| 0 | medium | 0.556 | 0.500 | 0.667 |
| 0 | good | 0.556 | 0.611 | 0.267 |
| 1 | bad | 0.667 | 0.667 | 0.800 |
| 1 | medium | 1.000 | 0.833 | 0.800 |
| 1 | good | 1.000 | 0.833 | 1.000 |
| 2 | bad | 1.000 | 0.500 | 0.800 |
| 2 | medium | 0.667 | 0.833 | 0.800 |
| 2 | good | 1.000 | 1.000 | 1.000 |

## Example Failures

- `0000001_02999_d_0000005` s=2 channel=medium Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `no`
- `0000001_02999_d_0000005` s=2 channel=bad Q: Are there car objects in this area? GT: `yes` Pred: `no`
- `0000001_03499_d_0000006` s=2 channel=medium Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `no`
- `0000001_03499_d_0000006` s=1 channel=good Q: How many awning-tricycle objects are in this area? GT: `2` Pred: `4`
- `0000001_03499_d_0000006` s=2 channel=bad Q: How many awning-tricycle objects are in this area? GT: `2` Pred: `4`
- `0000001_03499_d_0000006` s=1 channel=bad Q: Are there car objects in this area? GT: `yes` Pred: `no`
- `0000001_03999_d_0000007` s=1 channel=bad Q: Are there bicycle objects in this area? GT: `yes` Pred: `no`
- `0000001_03999_d_0000007` s=2 channel=bad Q: Are there bicycle objects in this area? GT: `yes` Pred: `no`
- `0000001_03999_d_0000007` s=1 channel=bad Q: Are there car objects in this area? GT: `yes` Pred: `no`
- `0000001_03999_d_0000007` s=1 channel=medium Q: Are there car objects in this area? GT: `yes` Pred: `no`
- `0000001_03999_d_0000007` s=1 channel=good Q: Are there car objects in this area? GT: `yes` Pred: `no`
- `0000001_03999_d_0000007` s=2 channel=bad Q: Are there car objects in this area? GT: `yes` Pred: `no`
