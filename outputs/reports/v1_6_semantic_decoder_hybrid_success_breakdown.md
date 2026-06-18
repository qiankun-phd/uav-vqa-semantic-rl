# V1.6 Success Breakdown

This report separates answer correctness, semantic quality satisfaction, deadline satisfaction, and final task success.
Quality satisfaction uses the measured LUT accuracy `A_k >= epsilon_k`; deadline satisfaction uses the configured delay proxy.

## by_service

| service_level | service_name | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|
| 0 | cache | 17838 | 0.549221 | 0.244478 | 1.000000 | 0.244478 |
| 1 | semantic-token decoder | 17838 | 0.846620 | 0.738143 | 1.000000 | 0.738143 |
| 2 | full image Qwen | 17838 | 0.603767 | 0.372351 | 0.655399 | 0.142785 |

## by_service_task

| service_level | service_name | question_type | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|---|
| 0 | cache | counting | 5967 | 0.619239 | 0.343891 | 1.000000 | 0.343891 |
| 0 | cache | presence | 11871 | 0.514026 | 0.194508 | 1.000000 | 0.194508 |
| 1 | semantic-token decoder | counting | 5967 | 0.541478 | 0.217195 | 1.000000 | 0.217195 |
| 1 | semantic-token decoder | presence | 11871 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | full image Qwen | counting | 5967 | 0.350427 | 0.000000 | 0.656109 | 0.000000 |
| 2 | full image Qwen | presence | 11871 | 0.731109 | 0.559515 | 0.655042 | 0.214556 |

## by_service_risk

| service_level | service_name | risk_level | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|---|
| 0 | cache | critical | 6147 | 0.624858 | 0.333821 | 1.000000 | 0.333821 |
| 0 | cache | normal | 11691 | 0.509452 | 0.197502 | 1.000000 | 0.197502 |
| 1 | semantic-token decoder | critical | 6147 | 0.808199 | 0.666179 | 1.000000 | 0.666179 |
| 1 | semantic-token decoder | normal | 11691 | 0.866821 | 0.775982 | 1.000000 | 0.775982 |
| 2 | full image Qwen | critical | 6147 | 0.651537 | 0.666179 | 0.000000 | 0.000000 |
| 2 | full image Qwen | normal | 11691 | 0.578650 | 0.217860 | 1.000000 | 0.217860 |

## by_service_channel

| service_level | service_name | channel_bin | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|---|
| 0 | cache | bad | 5946 | 0.546081 | 0.212748 | 1.000000 | 0.212748 |
| 0 | cache | good | 5946 | 0.550454 | 0.260343 | 1.000000 | 0.260343 |
| 0 | cache | medium | 5946 | 0.551127 | 0.260343 | 1.000000 | 0.260343 |
| 1 | semantic-token decoder | bad | 5946 | 0.846620 | 0.738143 | 1.000000 | 0.738143 |
| 1 | semantic-token decoder | good | 5946 | 0.846620 | 0.738143 | 1.000000 | 0.738143 |
| 1 | semantic-token decoder | medium | 5946 | 0.846620 | 0.738143 | 1.000000 | 0.738143 |
| 2 | full image Qwen | bad | 5946 | 0.573158 | 0.372351 | 0.655399 | 0.142785 |
| 2 | full image Qwen | good | 5946 | 0.635721 | 0.372351 | 0.655399 | 0.142785 |
| 2 | full image Qwen | medium | 5946 | 0.602422 | 0.372351 | 0.655399 | 0.142785 |

## by_service_task_risk

| service_level | service_name | question_type | risk_level | samples | answer_correctness | quality_satisfaction | deadline_satisfaction | final_success |
|---|---|---|---|---|---|---|---|---|
| 0 | cache | counting | critical | 2052 | 0.895712 | 1.000000 | 1.000000 | 1.000000 |
| 0 | cache | counting | normal | 3915 | 0.474330 | 0.000000 | 1.000000 | 0.000000 |
| 0 | cache | presence | critical | 4095 | 0.489133 | 0.000000 | 1.000000 | 0.000000 |
| 0 | cache | presence | normal | 7776 | 0.527135 | 0.296939 | 1.000000 | 0.296939 |
| 1 | semantic-token decoder | counting | critical | 2052 | 0.425439 | 0.000000 | 1.000000 | 0.000000 |
| 1 | semantic-token decoder | counting | normal | 3915 | 0.602299 | 0.331034 | 1.000000 | 0.331034 |
| 1 | semantic-token decoder | presence | critical | 4095 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1 | semantic-token decoder | presence | normal | 7776 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | full image Qwen | counting | critical | 2052 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | full image Qwen | counting | normal | 3915 | 0.534100 | 0.000000 | 1.000000 | 0.000000 |
| 2 | full image Qwen | presence | critical | 4095 | 0.978022 | 1.000000 | 0.000000 | 0.000000 |
| 2 | full image Qwen | presence | normal | 7776 | 0.601080 | 0.327546 | 1.000000 | 0.327546 |
