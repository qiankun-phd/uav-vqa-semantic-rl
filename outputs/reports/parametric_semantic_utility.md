# Parametric Semantic-Utility Model -- Calibration Report

Bayesian logistic regression (IRLS MAP + Laplace covariance, quasi-binomial
dispersion + out-of-fold structural residual variance) fitted to the V1.9 LUT
binomial cells, replacing the discrete table + nearest-SNR fallback.

- LUT cells: `648`  |  total VQA samples: `160542`
- Features (14): `intercept, q_counting, svc_token, svc_image, snr_z, snr_z:token, snr_z:image, view_ord, fresh_ord, risk_critical, q_counting:token, q_counting:image, view_ord:image, snr_z2`
- Prior `w~N(0,25.0)` (intercept free); dispersion `phi=25.82`; structural residual logit-var `s2=4.4696`; z=`1.96`

## A. Random 5-fold (in-grid interpolation)

Table memorises the dense grid here -- reported for calibration, not as the win.

| metric | parametric | discrete LUT |
|---|---|---|
| Brier (mean) | 0.0275 | 0.0003 |
| log-loss / sample | 0.6372 | 0.5640 |
| LCB coverage (target 1.96sd ~ 0.975) | 0.948 | 0.996 |

## B. Leave-one-SNR-out (SNR extrapolation -- the intended regime)

Entire SNR level removed from training; table can only nearest-neighbour.

| held-out SNR | parametric Brier | table Brier | better | param LCB cov |
|---|---|---|---|---|
| -5dB | 0.0312 | 0.0004 | table | 0.923 |
| 0dB | 0.0318 | 0.0004 | table | 0.923 |
| 5dB | 0.0312 | 0.0010 | table | 0.923 |
| 10dB | 0.0300 | 0.0007 | table | 0.923 |
| 15dB | 0.0301 | 0.0001 | table | 0.923 |
| 20dB | 0.0305 | 0.0002 | table | 0.923 |
| **mean** | **0.0308** | **0.0005** | **table** | -- |

## C. Sparse vs. dense cells (n<=16 vs. n>16) -- the parametric win

On sparse cells the discrete Wilson bound is wide and noisy; the pooled model
borrows strength -> tighter LCB (smaller half-width) at matched coverage.

| regime | cells | param Brier | table Brier | param LCB half-width | table LCB half-width | param cov | table cov |
|---|---|---|---|---|---|---|---|
| sparse | 216 | 0.0937 | 0.0076 | 0.5714 | 0.1755 | 0.820 | 0.966 |
| dense | 432 | 0.0267 | 0.0002 | 0.5243 | 0.0283 | 0.950 | 0.997 |

## D. Reliability (predicted vs. observed, sample-weighted)

| pred bin | mean predicted | mean observed | samples |
|---|---|---|---|
| [0.2,0.3] | 0.288 | 0.476 | 1473 |
| [0.3,0.4] | 0.349 | 0.318 | 13837 |
| [0.4,0.5] | 0.457 | 0.358 | 29177 |
| [0.5,0.6] | 0.548 | 0.588 | 49893 |
| [0.6,0.7] | 0.639 | 0.723 | 62251 |
| [0.7,0.8] | 0.738 | 0.878 | 2879 |
| [0.8,0.9] | 0.826 | 0.926 | 1032 |

## E. Fitted coefficients (log-odds)

| feature | weight |
|---|---|
| `intercept` | +0.273 |
| `q_counting` | +0.494 |
| `svc_token` | +0.694 |
| `svc_image` | +1.504 |
| `snr_z` | +0.000 |
| `snr_z:token` | +0.131 |
| `snr_z:image` | +0.042 |
| `view_ord` | +0.242 |
| `fresh_ord` | +0.206 |
| `risk_critical` | +0.219 |
| `q_counting:token` | -1.659 |
| `q_counting:image` | -2.672 |
| `view_ord:image` | +0.538 |
| `snr_z2` | -0.009 |

## F. Off-grid SNR generalisation (the table cannot interpolate)

| question | service | SNR(dB) | mean | LCB | uncertainty |
|---|---|---|---|---|---|
| presence | s1 | -2 | 0.648 | 0.042 | 0.476 |
| presence | s1 | +7 | 0.668 | 0.048 | 0.473 |
| presence | s1 | +12 | 0.678 | 0.052 | 0.472 |
| presence | s1 | +18 | 0.689 | 0.056 | 0.470 |
| presence | s2 | -2 | 0.761 | 0.097 | 0.450 |
| presence | s2 | +7 | 0.767 | 0.102 | 0.448 |
| presence | s2 | +12 | 0.769 | 0.104 | 0.447 |
| presence | s2 | +18 | 0.771 | 0.106 | 0.446 |
| counting | s1 | -2 | 0.478 | 0.013 | 0.484 |
| counting | s1 | +7 | 0.500 | 0.016 | 0.484 |
| counting | s1 | +12 | 0.511 | 0.017 | 0.484 |
| counting | s1 | +18 | 0.523 | 0.018 | 0.484 |
| counting | s2 | -2 | 0.462 | 0.012 | 0.484 |
| counting | s2 | +7 | 0.470 | 0.013 | 0.484 |
| counting | s2 | +12 | 0.474 | 0.013 | 0.484 |
| counting | s2 | +18 | 0.476 | 0.013 | 0.484 |

Model written to `outputs/lut/v1_9_parametric_utility_model.json`.

