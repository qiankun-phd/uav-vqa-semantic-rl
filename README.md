# UAV-VQA Semantic Communication V0

This is a clean V0 implementation for task-conditioned UAV VQA semantic
communication quality modeling and resource simulation.

It does not import or reuse previous HPPO-VQA/UAV-MEC code. The V0 pipeline is:

```text
VisDrone annotations -> VQA-style tasks -> semantic quality LUT -> resource simulation
```

## Environment

Use the existing remote environment:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python
```

The implementation avoids OpenCV and uses only the Python standard library plus
NumPy/Pandas where available.

## Download VisDrone DET valset

```bash
python scripts/download_visdrone_det.py --config configs/v0.yaml --split val
```

The script downloads the official VisDrone2019-DET valset Google Drive file and
extracts it to:

```text
data/raw/visdrone/DET/val/
```

If Google Drive blocks command-line download, manually download the valset from:

```text
https://github.com/VisDrone/VisDrone-Dataset
```

and place/extract it so that this file layout exists:

```text
data/raw/visdrone/DET/val/annotations/*.txt
data/raw/visdrone/DET/val/images/*
```

## Build V0 LUT

```bash
python scripts/build_v0_lut.py --config configs/v0.yaml --limit-images 100
```

Outputs:

```text
outputs/tasks/v0_tasks.csv
outputs/lut/v0_semantic_quality_lut.csv
outputs/lut/v0_semantic_quality_summary.json
outputs/lut/v0_semantic_quality_summary.md
```

If VisDrone is not available, add `--demo` to run a tiny built-in fixture:

```bash
python scripts/build_v0_lut.py --config configs/v0.yaml --demo
```

## Run Resource Simulation

```bash
python scripts/run_v0_sim.py --config configs/v0.yaml --episodes 10
```

Outputs:

```text
outputs/sim/v0_results.csv
outputs/sim/v0_summary.md
```

Baselines:

- `always_cache`
- `always_light`
- `always_image`
- `greedy_min_sufficient_evidence`

## Run Tests

```bash
python -m unittest discover -s tests
```

## Build V1 VLM-measured LUT

V1 keeps the V0.5 rule-based LUT intact and adds a measured path:

```text
VisDrone image / lightweight evidence + question -> VQA/VLM answer -> correctness -> measured LUT
```

Run a dependency-free mock smoke test first:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_vlm_eval.py --config configs/v0.yaml --limit-images 5 --evaluator mock
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/build_v1_vlm_lut.py --config configs/v0.yaml
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/report_v1_vlm_lut.py --config configs/v0.yaml
```

Outputs:

```text
outputs/vlm/v1_predictions.csv
outputs/lut/v1_vlm_semantic_quality_lut.csv
outputs/reports/v1_vlm_eval_report.md
```

For real VLM inference, install the optional Qwen-VL dependencies and run the
2B model first. This is the recommended path for a 10GB RTX 3080:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python -m pip install "transformers>=4.49.0" accelerate qwen-vl-utils
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/prepare_qwen_model.py --config configs/v1_qwen.yaml
```

If the status report says the model is missing, try downloading through the
Hugging Face mirror:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/prepare_qwen_model.py \
  --config configs/v1_qwen.yaml \
  --download
```

If the model has been manually copied to the server, set `model_local_path` in
`configs/v1_qwen.yaml`, or pass it at setup time:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/prepare_qwen_model.py \
  --config configs/v1_qwen.yaml \
  --model-local-path /path/to/Qwen2-VL-2B-Instruct
```

Then run the real evaluator:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_vlm_eval.py \
  --config configs/v1_qwen.yaml \
  --limit-images 20 \
  --evaluator qwen \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --resume
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/build_v1_vlm_lut.py --config configs/v1_qwen.yaml
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/report_v1_vlm_lut.py --config configs/v1_qwen.yaml
```

V1 evaluates only `presence` and `counting` in the first measured LUT. `risk`
remains in the V0.5 rule-based path until the simpler VQA tasks are stable.

## Run V1 Resource Simulation

Once a measured LUT exists, run the resource allocation simulation against it:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_resource_sim.py \
  --config configs/v1_qwen.yaml \
  --episodes 20
```

You can also reuse the V0 script with an explicit LUT source:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v0_sim.py \
  --config configs/v1_qwen.yaml \
  --lut-source v1_qwen \
  --filter-supported-tasks \
  --episodes 20
```

V1 simulation policies:

- `always_cache`
- `always_light`
- `always_image`
- `greedy_min_sufficient_evidence`
- `oracle_best_feasible_evidence`

## Modeling Notes

The lookup table estimates expected VQA accuracy:

```text
A_k = LUT[question_type, service_level, channel_bin, view_quality_bin, freshness_bin, risk_level]
```

Quality and deadline constraints are intentionally separate:

```text
quality satisfied if A_k >= epsilon_k
deadline satisfied if T_k <= tau_k
```

Service levels are fixed:

- `0`: cache answer
- `1`: lightweight evidence, e.g. tags / boxes / semantic tokens
- `2`: high-fidelity evidence, e.g. crop / full image

`critical` is a risk level, not a service level.
