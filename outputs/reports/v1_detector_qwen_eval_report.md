# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.
It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.

- prediction rows: `10638`
- LUT rows: `324`
- overall measured accuracy: `0.627`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator`
- real VLM present: `yes`

## Latency and Payload Summary

| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |
|---|---:|---:|---:|---:|---:|
| cache | 0.000 | 0.000 | 0.000 | 0.000 | 3546 |
| image | 0.509 | 0.678 | 83.709 | 228.236 | 3546 |
| lightweight | 0.080 | 0.097 | 0.546 | 0.791 | 3546 |

## Payload by Service and Channel

| service | channel | mean payload (KB) | p95 payload (KB) | samples |
|---:|---|---:|---:|---:|
| 0 | bad | 0.000 | 0.000 | 1182 |
| 0 | medium | 0.000 | 0.000 | 1182 |
| 0 | good | 0.000 | 0.000 | 1182 |
| 1 | bad | 0.506 | 0.772 | 1182 |
| 1 | medium | 0.562 | 0.793 | 1182 |
| 1 | good | 0.571 | 0.793 | 1182 |
| 2 | bad | 31.175 | 94.894 | 1182 |
| 2 | medium | 63.662 | 190.285 | 1182 |
| 2 | good | 156.288 | 460.813 | 1182 |

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
| 1 | bad | 0.291 | 0.161 | 0.443 |
| 1 | medium | 0.255 | 0.194 | 0.492 |
| 1 | good | 0.345 | 0.226 | 0.508 |
| 2 | bad | 0.382 | 0.323 | 0.475 |
| 2 | medium | 0.400 | 0.323 | 0.492 |
| 2 | good | 0.418 | 0.323 | 0.492 |

### presence

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.530 | 0.429 | 0.566 |
| 0 | medium | 0.514 | 0.571 | 0.577 |
| 0 | good | 0.556 | 0.497 | 0.523 |
| 1 | bad | 0.867 | 0.837 | 0.785 |
| 1 | medium | 0.876 | 0.939 | 0.946 |
| 1 | good | 0.905 | 0.939 | 0.978 |
| 2 | bad | 0.686 | 0.714 | 0.796 |
| 2 | medium | 0.743 | 0.776 | 0.860 |
| 2 | good | 0.810 | 0.857 | 0.903 |

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
