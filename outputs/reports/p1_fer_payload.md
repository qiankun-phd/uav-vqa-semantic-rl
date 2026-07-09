# P1-5: residual frame loss + M1 payload vs SNR

token payload: design 256 B, measured mean 910 B

## M1 rate-adaptive payload (Rician)

| SNR | budget (KB) | fitted mean (KB) | outage |
|---|---|---|---|
| -5 dB | 14.3 | 14.0 | 0.0132 |
| 0 dB | 35.2 | 34.3 | 0.0032 |
| 5 dB | 71.6 | 69.3 | 0.0009 |
| 10 dB | 121.0 | 117.0 | 0.0004 |
| 15 dB | 178.0 | 170.0 | 0.0001 |
| 20 dB | 238.3 | 215.1 | 0.0001 |

## token outage at bins (measured payload)

- awgn: -5dB=0.00e+00, 0dB=0.00e+00, 5dB=0.00e+00, 10dB=0.00e+00, 15dB=0.00e+00, 20dB=0.00e+00
- rayleigh: -5dB=5.15e-02, 0dB=1.65e-02, 5dB=5.37e-03, 10dB=1.72e-03, 15dB=6.00e-04, 20dB=2.50e-04
- rician: -5dB=7.13e-03, 0dB=1.70e-03, 5dB=5.75e-04, 10dB=2.25e-04, 15dB=1.00e-04, 20dB=0.00e+00

## fixed-rate LDPC FER (measured calibration)

- awgn: -5dB=1.000, 0dB=0.883, 5dB=0.014, 10dB=0.001, 15dB=0.000, 20dB=0.000
- rayleigh: -5dB=0.979, 0dB=0.751, 5dB=0.362, 10dB=0.127, 15dB=0.041, 20dB=0.013
- rician: -5dB=0.998, 0dB=0.729, 5dB=0.171, 10dB=0.026, 15dB=0.004, 20dB=0.001
