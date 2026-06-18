# Detector Migration Status

Generated at: 2026-06-16 11:06:06

## Server

- Active server: `qiankun@172.27.57.160`
- Project path: `/home/qiankun/phd_research/vqa_semcom`
- Python env: `/home/qiankun/.conda/envs/uav_semcom/bin/python`
- GPU: NVIDIA RTX 4060 8GB

## Completed

- Migrated the V0/V1 project from `172.28.23.182` to `172.27.57.160`.
- Installed/verified dependencies: `qwen-vl-utils`, `ultralytics`, `gdown`, `accelerate`.
- Downloaded a complete local Qwen2-VL-2B cache: `/home/qiankun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct`.
- Updated Qwen evaluator to use local model paths with `local_files_only=True`.
- Added detector pipeline code for VisDrone-to-YOLO conversion, YOLO training wrapper, detector lightweight evidence, and detector-Qwen evaluation.
- Unit tests pass: 19 tests.
- YOLO conversion smoke pass: 5 val images converted to YOLO labels.
- Real Qwen smoke pass for `s=1` lightweight evidence and `s=2` image evidence.
- Detector evidence smoke pass using temporary `yolov8n.pt`: detector outputs were converted to lightweight evidence and answered by Qwen.

## Current Blocker

- Official VisDrone2019-DET trainset did not start a normal transfer from Google Drive during the SSH run. The process stayed alive but no train zip was written after several minutes.
- Because trainset is missing, final VisDrone-trained detector weights are not available yet at `outputs/detector/visdrone_yolov8n/weights/best.pt`.

## Important Outputs

- Restored migrated Qwen predictions: `outputs/vlm/v1_qwen_predictions.csv`
- Restored Qwen LUT: `outputs/lut/v1_qwen_semantic_quality_lut.csv`
- Restored Qwen sim: `outputs/sim/v1_qwen_results.csv`
- Detector smoke predictions: `outputs/smoke/v1_detector_smoke_predictions.csv`
- Detector smoke detections: `outputs/smoke/v1_detector_smoke_detections.csv`

## Next Step

Once `VisDrone2019-DET-train.zip` is available under `data/raw/visdrone/`, run:

```bash
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/prepare_visdrone_yolo.py --config configs/v1_detector_qwen.yaml
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/train_visdrone_detector.py --config configs/v1_detector_qwen.yaml --epochs 50
/home/qiankun/.conda/envs/uav_semcom/bin/python scripts/run_v1_detector_eval.py --config configs/v1_detector_qwen.yaml --limit-images 100 --evaluator qwen --service-levels 1,2 --channels bad,medium,good --max-new-tokens 12
```
