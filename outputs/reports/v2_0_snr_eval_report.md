# V1.9 SNR-Calibrated Semantic Quality Report

This report calibrates task-conditioned answer correctness against sensed SNR bins. It does not assume a channel model such as AWGN or Rayleigh; SNR is used only as the sensed link-quality variable.

- prediction rows: `12744`
- LUT rows: `432`
- SNR bins: `-5dB, 0dB, 5dB, 10dB, 15dB, 20dB`
- model names: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c, semantic-token-decoder`
- real VLM present: `yes`
- cache accuracy spread across SNR: `0.000000`

## Answer Accuracy vs Sensed SNR

| service | evidence | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | detector semantic tokens | 0.681 | 0.681 | 0.681 | 0.681 | 0.681 | 0.681 |
| 2 | raw image evidence | 0.610 | 0.658 | 0.669 | 0.684 | 0.686 | 0.684 |

## Payload vs Sensed SNR

| service | evidence | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | detector semantic tokens | 0.822 | 0.822 | 0.822 | 0.822 | 0.822 | 0.822 |
| 2 | raw image evidence | 177.681 | 162.482 | 222.083 | 219.462 | 248.619 | 260.759 |

## Task Breakdown

| question type | service | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| counting | 1 | 0.534 | 0.534 | 0.534 | 0.534 | 0.534 | 0.534 |
| counting | 2 | 0.305 | 0.339 | 0.347 | 0.347 | 0.339 | 0.331 |
| presence | 1 | 0.754 | 0.754 | 0.754 | 0.754 | 0.754 | 0.754 |
| presence | 2 | 0.763 | 0.818 | 0.831 | 0.852 | 0.860 | 0.860 |

## Generated Tables

- accuracy by SNR: `/home/qiankun/phd_research/uav-vqa-semantic-rl/outputs/reports/v1_9_snr_accuracy_by_snr.csv`
- payload by SNR: `/home/qiankun/phd_research/uav-vqa-semantic-rl/outputs/reports/v1_9_snr_payload_by_snr.csv`
- task accuracy by SNR: `/home/qiankun/phd_research/uav-vqa-semantic-rl/outputs/reports/v1_9_snr_task_accuracy_by_snr.csv`

Interpretation: online resource allocation should use continuous sensed SNR for transmission delay, then map it to the nearest `snr_bin` for LUT-based answer accuracy.
