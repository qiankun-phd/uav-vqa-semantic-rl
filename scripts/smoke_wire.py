import glob
from pathlib import Path
from vqa_semcom.config import load_config
from vqa_semcom.degradation.channel import degrade_image
from vqa_semcom.detector.visdrone_yolo import degrade_detections_for_channel, DetectionRecord
cfg = load_config("configs/v2_0_ldpc_channel.yaml")
print("channel_model=", cfg["vlm"]["channel_model"], "calib_blocks=", cfg["vlm"]["fading_link"]["calib_blocks"])
img = sorted(glob.glob("data/raw/visdrone/DET/val/images/*.jpg"))[0]
outdir = Path("outputs/vlm/wire_smoke")
recs = [DetectionRecord("car", 100,100,50,40,0.9) for _ in range(30)]
for snr in ["-5dB","5dB","20dB"]:
    p = degrade_image(Path(img), outdir, snr, cfg)
    d = degrade_detections_for_channel(recs, snr, cfg, "img0")
    print(f"{snr}: image -> {p.name} (exists={p.exists()}), detections kept {len(d)}/30")
print("calibration json:", Path("outputs/lut/link_calibration_v2_0.json").exists())
