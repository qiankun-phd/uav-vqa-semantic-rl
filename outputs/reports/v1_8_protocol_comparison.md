# V1.8 Protocol Comparison Report

V1.8 reports two evaluation protocols instead of treating all results as one task distribution.

- Protocol-A: operational positive-query setting. This corresponds to V1.6 and is used for the main semantic-communication claim: task-aware evidence selection reduces payload while keeping useful VQA service quality.
- Protocol-B: balanced presence calibration. This corresponds to V1.7 and adds negative presence questions, so a lower success rate should be interpreted as stricter calibration, not system regression.
- Protocol-B+ROI: optional fairness check that adds detector-guided ROI/crop image evidence while keeping full-image evidence as a baseline.

## Policy Tables

### Protocol-A

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality viol. | deadline viol. | s0 | s1 | s2 | s3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.277 | 0.566 | 0.689 | 0.200 | 0.000 | 1.000 | 0.723 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.625 | 0.781 | 2.069 | 1.000 | 0.874 | 0.991 | 0.376 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.108 | 0.542 | 3.909 | 2.500 | 95.153 | 0.000 | 0.723 | 0.339 | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.794 | 0.816 | 2.076 | 1.087 | 19.412 | 0.796 | 0.206 | 0.000 | 0.277 | 0.517 | 0.206 | 0.000 |
| no_cache_greedy | 0.625 | 0.698 | 2.759 | 1.563 | 35.516 | 0.627 | 0.376 | 0.170 | 0.000 | 0.625 | 0.376 | 0.000 |
| no_semantic_tokens_greedy | 0.365 | 0.704 | 3.006 | 1.847 | 66.854 | 0.297 | 0.466 | 0.169 | 0.284 | 0.000 | 0.716 | 0.000 |
| oracle_best_feasible_evidence | 0.794 | 0.834 | 1.634 | 0.742 | 0.567 | 0.994 | 0.206 | 0.000 | 0.323 | 0.677 | 0.000 | 0.000 |

### Protocol-B

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality viol. | deadline viol. | s0 | s1 | s2 | s3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.262 | 0.548 | 0.688 | 0.200 | 0.000 | 1.000 | 0.738 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.202 | 0.607 | 2.075 | 1.000 | 0.841 | 0.991 | 0.798 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.417 | 0.596 | 3.909 | 2.500 | 95.560 | 0.000 | 0.441 | 0.323 | 0.000 | 0.000 | 1.000 | 0.000 |
| greedy_min_sufficient_evidence | 0.697 | 0.700 | 2.743 | 1.618 | 50.613 | 0.470 | 0.302 | 0.053 | 0.258 | 0.193 | 0.549 | 0.000 |
| no_cache_greedy | 0.571 | 0.592 | 3.533 | 2.196 | 76.595 | 0.198 | 0.428 | 0.180 | 0.000 | 0.203 | 0.797 | 0.000 |
| no_semantic_tokens_greedy | 0.544 | 0.704 | 3.069 | 1.897 | 69.603 | 0.272 | 0.314 | 0.196 | 0.262 | 0.000 | 0.738 | 0.000 |
| oracle_best_feasible_evidence | 0.697 | 0.715 | 2.266 | 1.252 | 30.782 | 0.678 | 0.303 | 0.052 | 0.351 | 0.294 | 0.355 | 0.000 |

### Protocol-B+ROI

| policy | success | accuracy | delay | energy | payload KB | payload reduction | quality viol. | deadline viol. | s0 | s1 | s2 | s3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_cache | 0.267 | 0.551 | 0.691 | 0.200 | 0.000 | 1.000 | 0.734 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| always_light | 0.201 | 0.604 | 2.069 | 1.000 | 0.846 | 0.991 | 0.799 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| always_image | 0.421 | 0.589 | 3.931 | 2.500 | 94.177 | 0.000 | 0.437 | 0.327 | 0.000 | 0.000 | 1.000 | 0.000 |
| always_roi | 0.420 | 0.514 | 2.992 | 1.800 | 146.974 | -0.561 | 0.567 | 0.107 | 0.000 | 0.000 | 0.000 | 1.000 |
| greedy_min_sufficient_evidence | 0.713 | 0.729 | 2.452 | 1.412 | 77.068 | 0.182 | 0.273 | 0.047 | 0.266 | 0.191 | 0.273 | 0.270 |
| no_cache_greedy | 0.575 | 0.621 | 3.182 | 1.925 | 114.948 | -0.221 | 0.410 | 0.185 | 0.000 | 0.202 | 0.410 | 0.387 |
| no_semantic_tokens_greedy | 0.603 | 0.724 | 2.740 | 1.652 | 96.379 | -0.023 | 0.286 | 0.145 | 0.277 | 0.000 | 0.422 | 0.301 |
| no_roi_greedy | 0.713 | 0.703 | 2.672 | 1.581 | 50.746 | 0.461 | 0.286 | 0.047 | 0.274 | 0.192 | 0.534 | 0.000 |
| oracle_best_feasible_evidence | 0.713 | 0.723 | 2.203 | 1.208 | 33.887 | 0.640 | 0.271 | 0.018 | 0.353 | 0.299 | 0.304 | 0.043 |

## Main Reading

- Protocol-A greedy success is 0.794 with 19.412 KB/task and 0.796 payload reduction vs full-image.
- Protocol-B greedy success is 0.697; this includes negative presence and therefore tests detector false negatives/false positives more strictly.
- The comparison should be framed as operational efficiency vs calibration robustness, not V1.7 being worse than V1.6.

## Quality Breakdown

| protocol | group | value | samples | accuracy | payload KB |
|---|---|---|---:|---:|---:|
| Protocol-A | all | all | 53514 | 0.666536 | 32.082250 |
| Protocol-A | service_level | 0 | 17838 | 0.549221 | 0.000000 |
| Protocol-A | service_level | 1 | 17838 | 0.846620 | 0.874782 |
| Protocol-A | service_level | 2 | 17838 | 0.603767 | 95.371968 |
| Protocol-A | question_type | counting | 17901 | 0.503715 | 31.596295 |
| Protocol-A | question_type | presence | 35613 | 0.748378 | 32.326517 |
| Protocol-A | presence_polarity | unknown | 35613 | 0.748378 | 32.326517 |
| Protocol-A | gt_count_bin | 1 | 5103 | 0.817754 | 31.127444 |
| Protocol-A | gt_count_bin | 2-3 | 3402 | 0.398001 | 30.802327 |
| Protocol-A | gt_count_bin | 4-9 | 3240 | 0.240432 | 30.458644 |
| Protocol-A | gt_count_bin | >=10 | 6156 | 0.440383 | 33.022483 |
| Protocol-B | all | all | 80271 | 0.582278 | 32.068323 |
| Protocol-B | service_level | 0 | 26757 | 0.562283 | 0.000000 |
| Protocol-B | service_level | 1 | 26757 | 0.598610 | 0.833000 |
| Protocol-B | service_level | 2 | 26757 | 0.585940 | 95.371968 |
| Protocol-B | question_type | counting | 27432 | 0.453813 | 32.114411 |
| Protocol-B | question_type | presence | 52839 | 0.648971 | 32.044396 |
| Protocol-B | presence_polarity | positive | 29187 | 0.558947 | 32.060295 |
| Protocol-B | presence_polarity | negative | 23652 | 0.760063 | 32.024776 |
| Protocol-B | gt_count_bin | 1 | 5886 | 0.815494 | 30.909729 |
| Protocol-B | gt_count_bin | 2-3 | 4428 | 0.383921 | 31.400559 |
| Protocol-B | gt_count_bin | 4-9 | 4752 | 0.216540 | 31.133445 |
| Protocol-B | gt_count_bin | >=10 | 12366 | 0.397865 | 33.320398 |
| Protocol-B+ROI | all | all | 80811 | 0.582173 | 32.981894 |
| Protocol-B+ROI | service_level | 0 | 26757 | 0.562283 | 0.000000 |
| Protocol-B+ROI | service_level | 1 | 26757 | 0.598610 | 0.833000 |
| Protocol-B+ROI | service_level | 2 | 26757 | 0.585940 | 95.371968 |
| Protocol-B+ROI | service_level | 3 | 540 | 0.566667 | 168.784229 |
| Protocol-B+ROI | question_type | counting | 27612 | 0.451833 | 32.778188 |
| Protocol-B+ROI | question_type | presence | 53199 | 0.649824 | 33.087623 |
| Protocol-B+ROI | presence_polarity | positive | 29376 | 0.559845 | 32.931432 |
| Protocol-B+ROI | presence_polarity | negative | 23823 | 0.760777 | 33.280222 |
| Protocol-B+ROI | gt_count_bin | 1 | 5904 | 0.816057 | 30.893773 |
| Protocol-B+ROI | gt_count_bin | 2-3 | 4455 | 0.383614 | 32.305026 |
| Protocol-B+ROI | gt_count_bin | 4-9 | 4779 | 0.215317 | 31.100256 |
| Protocol-B+ROI | gt_count_bin | >=10 | 12474 | 0.394420 | 34.481921 |

## Detector Threshold Sweep

| threshold | overall | positive presence | negative presence | counting | payload KB |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.604216 | 0.542707 | 0.893075 | 0.420604 | 0.850760 |
| 0.15 | 0.603207 | 0.523281 | 0.906773 | 0.426509 | 0.843603 |
| 0.20 | 0.602758 | 0.507246 | 0.917428 | 0.433071 | 0.837584 |
| 0.25 | 0.598834 | 0.491829 | 0.923135 | 0.433071 | 0.833000 |
| 0.30 | 0.587846 | 0.471169 | 0.940259 | 0.408136 | 0.825598 |

## Source Files

- Protocol-A sim: `/home/qiankun/phd_research/vqa_semcom/outputs/sim/v1_6_semantic_decoder_hybrid_results.csv`
- Protocol-A predictions: `/home/qiankun/phd_research/vqa_semcom/outputs/vlm/v1_6_semantic_decoder_hybrid_predictions.csv`
- Protocol-B sim: `/home/qiankun/phd_research/vqa_semcom/outputs/sim/v1_7_quality_calibrated_results.csv`
- Protocol-B predictions: `/home/qiankun/phd_research/vqa_semcom/outputs/vlm/v1_7_quality_calibrated_predictions.csv`
- Protocol-B+ROI sim: `/home/qiankun/phd_research/vqa_semcom/outputs/sim/v1_8_roi_baseline_results.csv`
- Protocol-B+ROI predictions: `/home/qiankun/phd_research/vqa_semcom/outputs/vlm/v1_8_roi_baseline_predictions.csv`
