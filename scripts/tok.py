from vqa_semcom.degradation.digital_link import (LinkConfig, FadingConfig, outage_probability, rate_budget_bytes, TOKEN_PAYLOAD_BYTES)
lc=LinkConfig(channel_mode="rate_adaptive", fading=FadingConfig("rician",6.0), bandwidth_hz=1e6, tx_time_budget_s=0.3, min_payload_bytes=1500)
B,tau=lc.bandwidth_hz,lc.tx_time_budget_s
r_tok=(TOKEN_PAYLOAD_BYTES*8/tau)/B
bytes_per_det=15  # class+box+conf
print(f"token payload={TOKEN_PAYLOAD_BYTES}B, r_target={r_tok:.4f} b/s/Hz")
print("SNR | token outage(=丢帧率) | image预算KB | 预算可容纳检测数(@15B)")
for s in [-5,0,5,10,15,20]:
    pout=outage_probability(s, lc.fading, r_tok)
    budget=rate_budget_bytes(s, lc)
    print(f"{s:>3} | {pout:.4f}               | {budget/1024:7.1f}  | {int(budget/bytes_per_det)}")
