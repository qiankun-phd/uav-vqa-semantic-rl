# V0.5 LUT Distribution Report

This report treats the LUT as a task-conditioned semantic accuracy model, not as an image-quality score.

- task rows: `6988`
- LUT rows: `405`
- accuracy range: `0.214` to `0.920`
- accuracy mean: `0.507`
- nonzero CI cells: `405/405`

## Task Distribution: question_type

| value | count | ratio |
|---|---:|---:|
| counting | 3080 | 0.441 |
| presence | 3408 | 0.488 |
| risk | 500 | 0.072 |

## Task Distribution: view_quality_bin

| value | count | ratio |
|---|---:|---:|
| good | 2036 | 0.291 |
| medium | 2442 | 0.349 |
| poor | 2510 | 0.359 |

## Task Distribution: risk_level

| value | count | ratio |
|---|---:|---:|
| critical | 2702 | 0.387 |
| normal | 4286 | 0.613 |

## Task Distribution: target_class

| value | count | ratio |
|---|---:|---:|
| awning-tricycle | 394 | 0.056 |
| bicycle | 328 | 0.047 |
| bus | 246 | 0.035 |
| car | 934 | 0.134 |
| motor | 882 | 0.126 |
| pedestrian | 948 | 0.136 |
| people | 874 | 0.125 |
| scene | 500 | 0.072 |
| tricycle | 614 | 0.088 |
| truck | 500 | 0.072 |
| van | 768 | 0.110 |

## LUT Coverage: service_level

| value | count | ratio |
|---|---:|---:|
| 0 | 135 | 0.333 |
| 1 | 135 | 0.333 |
| 2 | 135 | 0.333 |

## LUT Coverage: channel_bin

| value | count | ratio |
|---|---:|---:|
| bad | 135 | 0.333 |
| good | 135 | 0.333 |
| medium | 135 | 0.333 |

## LUT Coverage: freshness_bin

| value | count | ratio |
|---|---:|---:|
| expired | 135 | 0.333 |
| fresh | 135 | 0.333 |
| stale | 135 | 0.333 |

## Simulation Results

| policy | success | accuracy | delay | energy | quality violation | deadline violation |
|---|---:|---:|---:|---:|---:|---:|
| always_cache | 0.102 | 0.478 | 0.691 | 0.200 | 0.897 | 0.000 |
| always_light | 0.111 | 0.472 | 2.078 | 1.000 | 0.889 | 0.000 |
| always_image | 0.237 | 0.593 | 3.918 | 2.500 | 0.748 | 0.369 |
| greedy_min_sufficient_evidence | 0.295 | 0.585 | 3.450 | 2.127 | 0.698 | 0.366 |
