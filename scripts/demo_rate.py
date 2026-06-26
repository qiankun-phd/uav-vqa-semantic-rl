import sys, glob
from pathlib import Path
import numpy as np, cv2
from vqa_semcom.degradation.digital_link import (
    LinkConfig, FadingConfig, ergodic_spectral_efficiency, rate_budget_bytes,
    outage_probability, transmit_image_rate_adaptive)
SNRS=[-5,0,5,10,15,20]
tau=float(sys.argv[1]) if len(sys.argv)>1 else 0.3
OUT=Path("outputs/vlm/rate_demo"); OUT.mkdir(parents=True, exist_ok=True)
lc=LinkConfig(channel_mode="rate_adaptive", fading=FadingConfig("rician",6.0),
              bandwidth_hz=1e6, tx_time_budget_s=tau, min_payload_bytes=1500)
img=cv2.imread(sorted(glob.glob("data/raw/visdrone/DET/val/images/*.jpg"))[0])
h,w=img.shape[:2]
ok,full=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,95]); 
print(f"image {w}x{h}, full q95={len(full)/1024:.0f}KB, B=1MHz tau={tau}s")
r_min=(1500*8/tau)/1e6
print("SNR | SE(b/s/Hz) | budget KB | p_outage | scale | quality | sent KB | outage")
for s in SNRS:
    se=ergodic_spectral_efficiency(s,lc.fading); budget=rate_budget_bytes(s,lc)
    rng=np.random.default_rng(1000+s)
    dec,m=transmit_image_rate_adaptive(img,s,lc,rng)
    cv2.imwrite(str(OUT/f"rate_{s}dB.png"),dec)
    print(f"{s:>3} | {se:9.2f} | {budget/1024:8.1f} | {m['p_outage']:.4f} | {m.get('scale','-')} | {m.get('quality','-')} | {m.get('bytes',0)/1024:6.1f} | {m['outage']}")
