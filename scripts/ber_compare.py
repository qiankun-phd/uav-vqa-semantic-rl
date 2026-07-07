from vqa_semcom.degradation.digital_link import LinkConfig, FadingConfig, LDPCConfig, calibrate_link
SNRS=[-5,0,5,10]
def show(label, lc, blocks):
    c=calibrate_link(SNRS, lc, seed=0)
    print(f"{label:38s} | " + "  ".join(f"{s}dB:{c[f'{s}dB']['fer']:.3f}" for s in SNRS))
print("配置(LDPC码长/信道/衰落)            | 各SNR的FER")
print("-"*86)
show("n=96  Rician K6 (当前,块衰落)", LinkConfig(ldpc=LDPCConfig(96,3,6,maxiter=50), fading=FadingConfig('rician',6.0), calib_blocks=400), 400)
show("n=648 Rician K6 (长码,块衰落)", LinkConfig(ldpc=LDPCConfig(648,3,6,maxiter=50), fading=FadingConfig('rician',6.0), calib_blocks=300), 300)
show("n=648 AWGN (长码,无衰落)", LinkConfig(ldpc=LDPCConfig(648,3,6,maxiter=50), fading=FadingConfig('awgn',0.0), calib_blocks=300), 300)
show("n=648 Rayleigh (长码,块衰落)", LinkConfig(ldpc=LDPCConfig(648,3,6,maxiter=50), fading=FadingConfig('rayleigh',0.0), calib_blocks=300), 300)
