# V1 VLM-measured Semantic Quality Report

This report measures task-conditioned VQA accuracy from prediction correctness, not from an image-quality score.
It also measures communication load as the bytes actually sent for cache/lightweight/image evidence.

- prediction rows: `21384`
- LUT rows: `324`
- overall measured accuracy: `0.630`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator`
- real VLM present: `yes`

## Latency and Payload Summary

| evidence type | mean latency (s) | p95 latency (s) | mean payload (KB) | p95 payload (KB) | samples |
|---|---:|---:|---:|---:|---:|
| cache | 0.000 | 0.000 | 0.000 | 0.000 | 7128 |
| image | 0.413 | 0.582 | 84.601 | 226.212 | 7128 |
| lightweight | 0.069 | 0.086 | 0.628 | 0.875 | 7128 |

## Payload by Service and Channel

| service | channel | mean payload (KB) | p95 payload (KB) | samples |
|---:|---|---:|---:|---:|
| 0 | bad | 0.000 | 0.000 | 2376 |
| 0 | medium | 0.000 | 0.000 | 2376 |
| 0 | good | 0.000 | 0.000 | 2376 |
| 1 | bad | 0.600 | 0.875 | 2376 |
| 1 | medium | 0.637 | 0.875 | 2376 |
| 1 | good | 0.648 | 0.879 | 2376 |
| 2 | bad | 31.089 | 84.209 | 2376 |
| 2 | medium | 63.818 | 179.503 | 2376 |
| 2 | good | 158.896 | 445.933 | 2376 |

## Prediction Distribution

| field | value | count |
|---|---|---:|
| question_type | counting | 7641 |
| question_type | presence | 13743 |
| service_level | 0 | 7128 |
| service_level | 1 | 7128 |
| service_level | 2 | 7128 |
| channel_bin | bad | 7128 |
| channel_bin | good | 7128 |
| channel_bin | medium | 7128 |
| freshness_bin | expired | 7128 |
| freshness_bin | fresh | 7128 |
| freshness_bin | stale | 7128 |
| view_quality_bin | good | 6372 |
| view_quality_bin | medium | 4536 |
| view_quality_bin | poor | 10476 |
| evidence_type | cache | 7128 |
| evidence_type | image | 7128 |
| evidence_type | lightweight | 7128 |

## Accuracy by Service, Channel, and View

### counting

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.533 | 0.621 | 0.541 |
| 0 | medium | 0.588 | 0.632 | 0.559 |
| 0 | good | 0.543 | 0.655 | 0.604 |
| 1 | bad | 0.363 | 0.172 | 0.378 |
| 1 | medium | 0.356 | 0.310 | 0.467 |
| 1 | good | 0.400 | 0.362 | 0.489 |
| 2 | bad | 0.304 | 0.379 | 0.456 |
| 2 | medium | 0.341 | 0.379 | 0.467 |
| 2 | good | 0.370 | 0.379 | 0.478 |

### presence

| service | channel | poor view | medium view | good view |
|---:|---|---:|---:|---:|
| 0 | bad | 0.516 | 0.482 | 0.550 |
| 0 | medium | 0.531 | 0.527 | 0.534 |
| 0 | good | 0.531 | 0.512 | 0.521 |
| 1 | bad | 0.866 | 0.891 | 0.870 |
| 1 | medium | 1.000 | 0.982 | 0.986 |
| 1 | good | 1.000 | 1.000 | 1.000 |
| 2 | bad | 0.601 | 0.645 | 0.753 |
| 2 | medium | 0.652 | 0.691 | 0.822 |
| 2 | good | 0.723 | 0.773 | 0.890 |

## Paper Table: Resource Policies

| policy | success | accuracy | delay | energy | payload KB | payload reduction | cache | light | image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.280 | 0.568 | 0.689 | 0.200 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| always_light | 0.519 | 0.664 | 2.073 | 1.000 | 0.644 | 0.993 | 0.000 | 1.000 | 0.000 |
| always_image | 0.174 | 0.554 | 3.902 | 2.500 | 87.352 | 0.000 | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.721 | 0.781 | 2.284 | 1.253 | 24.307 | 0.722 | 0.271 | 0.417 | 0.313 |
| oracle_best_feasible_evidence | 0.723 | 0.809 | 1.972 | 1.039 | 18.874 | 0.784 | 0.355 | 0.429 | 0.215 |

## Example Failures

- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_02999_d_0000005` s=1 channel=bad payload=0.97KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `2`
- `0000001_02999_d_0000005` s=1 channel=medium payload=0.98KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `yes`
- `0000001_02999_d_0000005` s=2 channel=bad payload=106.18KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=medium payload=228.42KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `0`
- `0000001_02999_d_0000005` s=2 channel=good payload=570.06KB Q: How many awning-tricycle objects are in this area? GT: `17` Pred: `2`
- `0000001_03499_d_0000006` s=2 channel=bad payload=103.52KB Q: Are there awning-tricycle objects in this area? GT: `yes` Pred: `No`
- `0000001_03499_d_0000006` s=2 channel=bad payload=103.52KB Q: How many awning-tricycle objects are in this area? GT: `2` Pred: `0`
- `0000001_03499_d_0000006` s=2 channel=medium payload=215.95KB Q: How many awning-tricycle objects are in this area? GT: `2` Pred: `0`
- `0000001_03499_d_0000006` s=2 channel=good payload=522.30KB Q: How many awning-tricycle objects are in this area? GT: `2` Pred: `0`
- `0000001_03499_d_0000006` s=1 channel=bad payload=0.44KB Q: Are there bicycle objects in this area? GT: `yes` Pred: `no`
