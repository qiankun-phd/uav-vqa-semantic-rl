# s3 ROI + W2 presence t-scan -- harvest analysis

Chain: `outputs/logs/chain_s3.log` -> "S3+W2 CHAIN ALL DONE" (2026-07-08).
Runs: `s3_rician_{main,cmp,extra}_predictions.csv` (~43.8k VLM rows, gitignored under `outputs/vlm/`).
Rician channel, 6 SNR points (-5..20 dB), test split.

## W1 / s3 : four ROI modes vs token(s1) / image(s2)

### Service ladder (qtype=all) -- payload step confirmed
| service | method | accuracy | payload (median over SNR) |
|---|---|---|---|
| s1 token | M3_token | 0.671 (flat, SNR-invariant) | ~0.92 KB |
| s3 ROI | M6_roi | 0.562-0.583 | ~11-69 KB (grows with SNR) |
| s2 image | M1_image | 0.577-0.607 | ~162-271 KB |

s3 sits at the predicted ~10 KB step (11.2 KB @ -5 dB), ~17x lighter than s2
image. Full table: `outputs/reports/s3_service_ladder.csv`.

### (a) presence x polarity x roi_mode  (`s3_presence_breakdown.csv`)
Core read -- the four ROI modes split hard by presence polarity:

**Negative presence (object absent)** @ -5 dB:
- s1 token 0.874, s3 ROI(all) 0.866, s2 image 0.747
- s3 by mode: thumbnail **0.934**, suspect 0.700, target_topk 0.688
- Token and ROI-thumbnail crush image: they do not hallucinate absent objects.

**Positive presence (object present)** @ -5 dB:
- s1 token 0.599, s2 image 0.552, s3 ROI(all) 0.531
- s3 by mode: target_topk **0.671**, suspect 0.524, thumbnail **0.061** (catastrophic)
- Thumbnail downsampling destroys small-object detection on positives
  (0.06-0.27 across all SNR); target_topk is the only mode that helps positives
  (rises to 0.850 @ 20 dB).

Takeaway: no single ROI mode dominates -- **thumbnail wins negatives,
target_topk wins positives, suspect is middling on both**. This is the
polarity-conditioned routing signal for the semantic controller.

### (b) s3 vs s2 by SNR
s3 ROI only reverses/beats image at the lowest SNR (-5 dB: 0.5833 vs 0.5769)
while spending ~17x fewer bytes; at >=0 dB image edges ahead on accuracy.
The per-bit advantage of ROI is real but marginal at the "all" level -- the
value is concentrated in the negative-presence / thumbnail cell, not the mean.

### (c) extended comparison (`comparison_s3.csv`, M4/M5 candidate sets {1,2,3})
- M4_adaptive12 == M4_adaptive123 (0.671, 0.92 KB): the adaptive policy almost
  never selects s3, so adding s3 to the candidate set does not move the
  learned/heuristic operating point.
- M5_oracle123 > M5_oracle12 (@ -5 dB 0.833 vs 0.789; @ 20 dB 0.853 vs 0.808)
  at ~23 KB payload -- an oracle that can pick s3 per-sample gains ~4 pts,
  confirming s3 carries complementary information that only an oracle exploits.

## W2 : presence token-budget t-scan

### presence accuracy(t, snr)  (`w2_presence_accuracy_t_snr.csv`)
- Saturates by t~=8-16 at ~0.65-0.68; SNR-insensitive above 0 dB.
- Non-monotone dip at t=2 (0.51-0.57) then recovers -- 1 token > 2 tokens.
- Diminishing returns beyond t=8; "full" only reaches 0.65-0.68.

### confidence reliability -- INVERTED  (`confidence_auc.csv`, `confidence_calibration.csv`)
- **Overall ROC-AUC = 0.406** (answer_confidence vs correctness) -- BELOW 0.5,
  i.e. self-reported confidence is *anti*-correlated with being right.
- Per budget: 0.331 (t=1) rising to 0.452 (full), never crossing 0.5.
- Calibration bins confirm: highest-confidence bin (0.8-0.9) has LOWEST accuracy
  (0.34), lowest-confidence bin (0.0-0.5) has HIGH accuracy (0.70).
- Negative result: the VLM's verbalized confidence must NOT be used for
  gating/admission -- it would systematically drop the correct answers.

## Products written
`s3_service_ladder.csv`, `w2_presence_accuracy_t_snr.csv`, `confidence_auc.csv`
(new); `comparison_s3.csv`, `s3_presence_breakdown.csv`, `token_budget_full.csv`,
`confidence_calibration.csv`, `presence_token_budget_raw.csv` (chain).
Analysis script: `scripts/s3w2_analysis.py`.
