# V1.6 Success Breakdown

This report separates answer correctness, semantic quality satisfaction, deadline satisfaction, and final task success.
Quality satisfaction uses the measured LUT accuracy `A_k >= epsilon_k`; deadline satisfaction uses the configured delay proxy.

## by_service

| service_level | service_name | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|
| 0 | cache | 26757 | 0.562283 | 0.304855 | 1.000000 | 0.304855 |
| 1 | semantic-token decoder | 26757 | 0.598610 | 0.167508 | 1.000000 | 0.167508 |
| 2 | full image Qwen | 26757 | 0.585940 | 0.588070 | 0.662630 | 0.485032 |

## by_service_task

| service_level | service_name | question_type | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|---|
| 0 | cache | counting | 9144 | 0.654090 | 0.435477 | 1.000000 | 0.435477 |
| 0 | cache | presence | 17613 | 0.514620 | 0.237041 | 1.000000 | 0.237041 |
| 1 | semantic-token decoder | counting | 9144 | 0.432415 | 0.053806 | 1.000000 | 0.053806 |
| 1 | semantic-token decoder | presence | 17613 | 0.684892 | 0.226537 | 1.000000 | 0.226537 |
| 2 | full image Qwen | counting | 9144 | 0.274934 | 0.029528 | 0.549213 | 0.029528 |
| 2 | full image Qwen | presence | 17613 | 0.747402 | 0.878045 | 0.721513 | 0.721513 |

## by_service_risk

| service_level | service_name | risk_level | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|---|
| 0 | cache | critical | 6804 | 0.723986 | 0.587155 | 1.000000 | 0.587155 |
| 0 | cache | normal | 19953 | 0.507142 | 0.208590 | 1.000000 | 0.208590 |
| 1 | semantic-token decoder | critical | 6804 | 0.554233 | 0.394180 | 1.000000 | 0.394180 |
| 1 | semantic-token decoder | normal | 19953 | 0.613742 | 0.090212 | 1.000000 | 0.090212 |
| 2 | full image Qwen | critical | 6804 | 0.381393 | 0.394180 | 0.000000 | 0.000000 |
| 2 | full image Qwen | normal | 19953 | 0.655691 | 0.654187 | 0.888588 | 0.650429 |

## by_service_channel

| service_level | service_name | channel_bin | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|---|
| 0 | cache | bad | 8919 | 0.556116 | 0.306649 | 1.000000 | 0.306649 |
| 0 | cache | good | 8919 | 0.566319 | 0.308106 | 1.000000 | 0.308106 |
| 0 | cache | medium | 8919 | 0.564413 | 0.299809 | 1.000000 | 0.299809 |
| 1 | semantic-token decoder | bad | 8919 | 0.542886 | 0.155399 | 1.000000 | 0.155399 |
| 1 | semantic-token decoder | good | 8919 | 0.631685 | 0.173562 | 1.000000 | 0.173562 |
| 1 | semantic-token decoder | medium | 8919 | 0.621258 | 0.173562 | 1.000000 | 0.173562 |
| 2 | full image Qwen | bad | 8919 | 0.573831 | 0.586613 | 0.662630 | 0.485032 |
| 2 | full image Qwen | good | 8919 | 0.597713 | 0.590986 | 0.662630 | 0.485032 |
| 2 | full image Qwen | medium | 8919 | 0.586276 | 0.586613 | 0.662630 | 0.485032 |

## by_service_task_risk

| service_level | service_name | question_type | risk_level | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|---|---|
| 0 | cache | counting | critical | 4122 | 0.882824 | 0.966036 | 1.000000 | 0.966036 |
| 0 | cache | counting | normal | 5022 | 0.466348 | 0.000000 | 1.000000 | 0.000000 |
| 0 | cache | presence | critical | 2682 | 0.479866 | 0.004847 | 1.000000 | 0.004847 |
| 0 | cache | presence | normal | 14931 | 0.520863 | 0.278749 | 1.000000 | 0.278749 |
| 1 | semantic-token decoder | counting | critical | 4122 | 0.310771 | 0.000000 | 1.000000 | 0.000000 |
| 1 | semantic-token decoder | counting | normal | 5022 | 0.532258 | 0.097969 | 1.000000 | 0.097969 |
| 1 | semantic-token decoder | presence | critical | 2682 | 0.928412 | 1.000000 | 1.000000 | 1.000000 |
| 1 | semantic-token decoder | presence | normal | 14931 | 0.641149 | 0.087603 | 1.000000 | 0.087603 |
| 2 | full image Qwen | counting | critical | 4122 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | full image Qwen | counting | normal | 5022 | 0.500597 | 0.053763 | 1.000000 | 0.053763 |
| 2 | full image Qwen | presence | critical | 2682 | 0.967562 | 1.000000 | 0.000000 | 0.000000 |
| 2 | full image Qwen | presence | normal | 14931 | 0.707856 | 0.856138 | 0.851115 | 0.851115 |
