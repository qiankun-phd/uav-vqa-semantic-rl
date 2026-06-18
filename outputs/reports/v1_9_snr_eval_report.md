# V1.9 SNR-Calibrated Semantic Quality Report

This report calibrates task-conditioned answer correctness against sensed SNR bins. It does not assume a channel model such as AWGN or Rayleigh; SNR is used only as the sensed link-quality variable.

- prediction rows: `160542`
- LUT rows: `648`
- SNR bins: `-5dB, 0dB, 5dB, 10dB, 15dB, 20dB`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, cache-simulator, semantic-token-decoder`
- real VLM present: `yes`
- cache accuracy spread across SNR: `0.000000`

## Answer Accuracy vs Sensed SNR

| service | evidence | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | cache answer | 0.566 | 0.566 | 0.566 | 0.566 | 0.566 | 0.566 |
| 1 | detector semantic tokens | 0.546 | 0.565 | 0.594 | 0.608 | 0.617 | 0.632 |
| 2 | raw image evidence | 0.571 | 0.580 | 0.585 | 0.586 | 0.592 | 0.597 |

## Payload vs Sensed SNR

| service | evidence | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | cache answer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | detector semantic tokens | 0.791 | 0.816 | 0.839 | 0.850 | 0.854 | 0.855 |
| 2 | raw image evidence | 35.825 | 53.230 | 68.911 | 92.700 | 127.051 | 178.092 |

## Task Breakdown

| question type | service | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| counting | 0 | 0.654 | 0.654 | 0.654 | 0.654 | 0.654 | 0.654 |
| counting | 1 | 0.337 | 0.364 | 0.415 | 0.449 | 0.469 | 0.494 |
| counting | 2 | 0.263 | 0.273 | 0.274 | 0.278 | 0.276 | 0.280 |
| presence | 0 | 0.521 | 0.521 | 0.521 | 0.521 | 0.521 | 0.521 |
| presence | 1 | 0.655 | 0.669 | 0.687 | 0.691 | 0.694 | 0.703 |
| presence | 2 | 0.732 | 0.739 | 0.747 | 0.746 | 0.756 | 0.761 |

## Resource Simulation

| policy | success | accuracy | delay | energy | payload KB | payload reduction |
|---|---:|---:|---:|---:|---:|---:|
| always_cache | 0.286 | 0.560 | 0.050 | 0.200 | 0.000 | 1.000 |
| always_light | 0.203 | 0.599 | 0.355 | 1.000 | 0.850 | 0.991 |
| always_image | 0.560 | 0.588 | 1.546 | 2.500 | 93.044 | 0.000 |
| greedy_min_sufficient_evidence | 0.711 | 0.707 | 0.896 | 1.562 | 48.024 | 0.484 |
| no_cache_greedy | 0.567 | 0.583 | 1.305 | 2.195 | 73.879 | 0.206 |
| no_semantic_tokens_greedy | 0.704 | 0.712 | 1.119 | 1.843 | 65.779 | 0.293 |
| oracle_best_feasible_evidence | 0.711 | 0.721 | 0.664 | 1.219 | 29.122 | 0.687 |

## Generated Tables

- accuracy by SNR: `/home/qiankun/phd_research/vqa_semcom/outputs/reports/v1_9_snr_accuracy_by_snr.csv`
- payload by SNR: `/home/qiankun/phd_research/vqa_semcom/outputs/reports/v1_9_snr_payload_by_snr.csv`
- task accuracy by SNR: `/home/qiankun/phd_research/vqa_semcom/outputs/reports/v1_9_snr_task_accuracy_by_snr.csv`

Interpretation: online resource allocation should use continuous sensed SNR for transmission delay, then map it to the nearest `snr_bin` for LUT-based answer accuracy.
