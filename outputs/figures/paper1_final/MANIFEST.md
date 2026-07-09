# Paper-1 figure bundle -- MANIFEST

Staging dir: `outputs/figures/paper1_final/`
Regenerated: 2026-07-09 (server 172.27.57.160, repo `~/phd_research/vqa_semcom`, branch `main`)
Python: `~/.conda/envs/uav_semcom/bin/python`

All nine PDFs below are regenerated from the CURRENT prediction logs
(`outputs/vlm/v3_0_{awgn,rayleigh,rician}_predictions.csv`, dated 2026-07-04) via the
tidy report `outputs/reports/comparison_v3_5qt.csv` (rebuilt 2026-07-09 22:09).
Test split = crc32(image_id) % 100 < 20, with `image_id` always read as a STRING.
The target filenames match the paper's `\evfig{...}` macro exactly.

IEEE styling applied to every figure: all in-figure titles (`ax.set_title` /
`fig.suptitle`) removed; channel / operating-point kept only as a small in-axes
label; error bands / error bars added from the Wilson 95% `lcb`/`ucb` columns
(no fabricated uncertainty). Legend rename `M3 GO-SG token` -> `M3 fixed token`.

| # | target file | generating script | data source CSV | source prediction logs |
|---|-------------|-------------------|-----------------|------------------------|
| F1 | `F1_accuracy_snr.pdf` | `scripts/make_comparison_figures_v2.py` | `outputs/reports/comparison_v3_5qt.csv` | `outputs/vlm/v3_0_{awgn,rayleigh,rician}_predictions.csv` (+`m2_analog_*`, `v2_0_*_naive`, `v3_0_clean` via the report) |
| F2 | `F2_cliff.pdf` | `scripts/make_comparison_figures_v2.py` | `outputs/reports/comparison_v3_5qt.csv` | same as F1 |
| F3 | `F3_latency.pdf` | `scripts/build_latency_breakdown.py` | `outputs/reports/comparison_v3_5qt.csv` (channel uses) + `outputs/reports/latency_breakdown.csv` | `outputs/vlm/v2_0_rayleigh_predictions.csv`, `m2_analog_rayleigh_*`, `v2_0_rayleigh_naive_*` (measured latency_sec / detector_latency_sec) |
| F4 | `F4_bytype.pdf` | `scripts/make_comparison_figures_v2.py` | `outputs/reports/comparison_v3_5qt.csv` | same as F1 |
| F5 | `F5_pareto.pdf` | `scripts/make_comparison_figures_v2.py` | `outputs/reports/comparison_v3_5qt.csv` | same as F1 |
| F6 | `F6_complementarity.pdf` | `scripts/make_complementarity_fig.py` | (computed in-script) pooled `outputs/vlm/v3_0_{awgn,rayleigh,rician}_predictions.csv` | `outputs/vlm/v3_0_{awgn,rayleigh,rician}_predictions.csv` |
| F7 | `F7_mismatch.pdf` | `scripts/build_mismatch_matrix.py` | `outputs/reports/mismatch_matrix.csv` | `outputs/vlm/v3_0_{awgn,rayleigh,rician}_predictions.csv` |
| F8 | `F8_token_budget.pdf` | `scripts/build_token_budget_sweep.py` | `outputs/reports/token_budget_full.csv` | `outputs/detector/v2_0_snr_detections.csv` + task CSVs (`outputs/tasks/v1_7_tasks.csv`, `v2_comparison_tasks.csv`, `v2_extra_tasks.csv`); validated vs logged M3 in `comparison_v3_5qt.csv` |
| F9 | `F9_separation.pdf` | `scripts/build_separation_capacity.py` | `outputs/reports/latency_breakdown_3ch.csv` -> `outputs/reports/separation_capacity.csv` | derived from the 3-channel latency breakdown (F3 pipeline, all channels) |

## Exact regeneration commands (run on server 160, repo root)

```
PY=~/.conda/envs/uav_semcom/bin/python
# F1,F2,F4,F5
$PY scripts/make_comparison_figures_v2.py --csv outputs/reports/comparison_v3_5qt.csv --tag v3_final
# F6
$PY scripts/make_complementarity_fig.py
# F3 (rayleigh panel)
$PY scripts/build_latency_breakdown.py --comparison-csv outputs/reports/comparison_v3_5qt.csv \
    --prefix v2_0 --channel rayleigh --tag v3_final
# F7
$PY scripts/build_mismatch_matrix.py --prefix v3_0 --tag v3
# F8 (plot-only from the fresh full sweep csv; no CPU-heavy recompute)
$PY scripts/build_token_budget_sweep.py --from-csv outputs/reports/token_budget_full.csv \
    --fig-channel rician --tag v3
# latency_breakdown_3ch.csv = per-channel build_latency_breakdown.py (awgn+rayleigh+rician), concatenated
# F9
$PY scripts/build_separation_capacity.py --latency outputs/reports/latency_breakdown_3ch.csv \
    --out outputs/reports/separation_capacity.csv \
    --fig outputs/figures/comparison/F9_separation_capacity
```

Source-name -> target-name copy map (freshest v3/v3_final variants chosen):
`F1_acc_snr_3panel_v3_final.pdf`->F1_accuracy_snr, `F2_cliff_v3_final.pdf`->F2_cliff,
`F3_latency_v3_final.pdf`->F3_latency, `F4_qtype_v3_final.pdf`->F4_bytype,
`F5_pareto_v3_final.pdf`->F5_pareto, `F6_complementarity.pdf`->F6_complementarity,
`F7_mismatch_v3.pdf`->F7_mismatch, `F8_token_budget_v3.pdf`->F8_token_budget,
`F9_separation_capacity.pdf`->F9_separation.

## One-line description of each figure

- **F1 -- accuracy vs SNR (3-panel: AWGN / Rayleigh / Rician).** All seven methods;
  M4_adaptive is the best non-oracle at every SNR (~0.676-0.681 at 20 dB); Wilson-95%
  bands shaded for M4/M1/M3.
- **F2 -- cliff effect (Rayleigh).** Naive fixed-rate digital (M0_naive) collapses at
  low SNR (~0.409 at -5 dB) while the adaptive policy degrades gracefully.
- **F3 -- end-to-end latency breakdown (Rayleigh).** Stacked upload / tx-side detector /
  receiver inference per method x SNR; token path is orders of magnitude cheaper on airtime.
- **F4 -- per-question-type accuracy (Rician, SNR = 5 dB), asymmetric Wilson error bars.**
  Shows the routing rule: symbolic types (counting/comparison/co_presence) favour token,
  presence favours image; M4 tracks the winner per type.
- **F5 -- goal-oriented efficiency Pareto (Rician).** Accuracy vs mean complex channel
  uses per query (log x); M4 sits on the efficient frontier; M4 vertical Wilson bars shown.
- **F6 -- evidence<->question complementarity (pooled 3 channels, test).** token-gain
  Delta = acc(token) - acc(image) per question type, with Wilson-95% error bars.
- **F7 -- policy robustness to CSI error (assumed-vs-true SNR heatmaps, per channel).**
  Off-diagonal accuracy stays close to the matched diagonal -> scheduler robust to stale SNR.
- **F8 -- variable token-budget sweep (Rician).** Accuracy vs top-t truncation of the s1
  symbolic evidence, three panels (SNR -5 / +20 dB tokens axis, 5 dB bandwidth axis).
- **F9 -- evidence selection enters the BUBBLES Block-4 separation loop (Rician, class
  SAIL I-II).** (a) tactical-conflict separation d_TC vs evidence, (b) relative airspace
  capacity; self-test reproduces BUBBLES Tables G-4/G-5 to <0.2%.

## Authoritative-number confirmation

- **F6 token-gain (pooled 3ch, test), COMPUTED not hardcoded:**
  counting **+0.174**, comparison **+0.151**, co_presence **+0.119**,
  threshold **-0.014** (NEGATIVE bar, correct sign), presence **-0.087**.
- **F4 per-type (Rician, 5 dB):** presence M1_image = M4_adaptive = **0.7805** (> M3_token
  0.678) -> presence routes to image; counting token 0.453 > image 0.290, comparison token
  0.846 > image 0.702, co_presence token 0.664 > image 0.587, threshold ~tie.
- **F1/F4 main comparison:** M4_adaptive best non-oracle at every SNR; 20 dB M4 ~0.680.

## OLD -> NEW differences (the two fixed bugs)

- **F6 (`make_complementarity_fig.py`):**
  - OLD: hardcoded stale 182-server deltas `[0.194, 0.122, 0.111, 0.021, -0.068]` for
    order `[counting, co_presence, comparison, threshold, presence]` -- the **threshold
    sign was WRONG (+0.021)** and every value was off.
  - NEW: deltas are **computed in-script** from the pooled `v3_0_*` logs (dtype-str
    image_id, same test split as `build_evidence_complementarity.py`). Values are now
    counting +0.174, comparison +0.151, co_presence +0.119, **threshold -0.014 (negative)**,
    presence -0.087; Wilson-95% error bars added; in-figure title removed.
- **F4 (`make_comparison_figures_v2.py` per-type panel):**
  - OLD asset showed the **M4 presence bar LOWER than M1**, contradicting the routing
    rule (presence -> image).
  - NEW: F4 reads the fresh `comparison_v3_5qt.csv`; the M4 presence bar now equals the
    image accuracy (**0.7805**) and sits above the token bar, so presence correctly routes
    to image. Asymmetric Wilson error bars added; in-figure title removed.
