# State Vector / MLP Train-only PPO Ablation

- Output directory: `outputs/rl/state_mlp_ablation_trainonly_20260624/`
- Runs: 4 groups x 3 scenarios x 3 seeds = 36 train-only PPO runs.
- Train episodes per run: 120; evaluation episodes: 0; tasks per episode: 12.
- Policies: two-timescale PPO with proposed semantic RL, deadline-aware evidence guard, and payload-delay-aware projection.
- Device requested in every run: `cuda:0`; logs show `PPO training device: cuda:0 (NVIDIA GeForce RTX 4060)` and `nvidia-smi` observed four concurrent python GPU processes during training.
- No final evaluation rollout was run. Results below are from the final 20 rows of `ppo_training_trace.csv`.

## Run Completion

- Completed trace files: `36/36`.
- Failed or missing runs: `0`.

## Aggregated Final-20-Episode Metrics

| group | scenario | shaped reward | success | acc LCB | quality gap | Q deadline | Q quality | non-cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: state_v1 + 128,128 | low_snr_blockage | -2206.687 +/- 1415.565 | 0.165 +/- 0.015 | 0.723 +/- 0.040 | 0.119 +/- 0.036 | 146.208 +/- 13.200 | 0.533 +/- 0.192 | 0.689 +/- 0.010 |
| A: state_v1 + 128,128 | disaster_hotspot | -5.526 +/- 1.554 | 0.080 +/- 0.018 | 0.514 +/- 0.016 | 0.331 +/- 0.015 | 3.510 +/- 0.265 | 1.472 +/- 0.108 | 0.855 +/- 0.017 |
| A: state_v1 + 128,128 | edge_overload | 10.151 +/- 1.569 | 0.474 +/- 0.057 | 0.674 +/- 0.014 | 0.045 +/- 0.009 | 7.242 +/- 0.629 | 0.446 +/- 0.076 | 0.962 +/- 0.011 |
| B: state_v2 + 128,128 | low_snr_blockage | -2357.281 +/- 1291.761 | 0.108 +/- 0.051 | 0.754 +/- 0.066 | 0.092 +/- 0.059 | 174.761 +/- 34.995 | 0.419 +/- 0.293 | 0.621 +/- 0.055 |
| B: state_v2 + 128,128 | disaster_hotspot | -5.400 +/- 1.426 | 0.078 +/- 0.015 | 0.516 +/- 0.019 | 0.329 +/- 0.018 | 3.643 +/- 0.195 | 1.458 +/- 0.125 | 0.855 +/- 0.018 |
| B: state_v2 + 128,128 | edge_overload | 9.913 +/- 2.326 | 0.464 +/- 0.083 | 0.677 +/- 0.011 | 0.043 +/- 0.006 | 7.023 +/- 1.806 | 0.422 +/- 0.052 | 0.951 +/- 0.006 |
| C: state_v2 + 256,256 | low_snr_blockage | -2206.505 +/- 1415.722 | 0.165 +/- 0.015 | 0.723 +/- 0.040 | 0.119 +/- 0.036 | 146.208 +/- 13.200 | 0.533 +/- 0.192 | 0.689 +/- 0.010 |
| C: state_v2 + 256,256 | disaster_hotspot | -5.630 +/- 0.089 | 0.073 +/- 0.003 | 0.513 +/- 0.002 | 0.332 +/- 0.001 | 3.746 +/- 0.107 | 1.480 +/- 0.024 | 0.858 +/- 0.014 |
| C: state_v2 + 256,256 | edge_overload | 9.265 +/- 3.209 | 0.446 +/- 0.099 | 0.677 +/- 0.013 | 0.042 +/- 0.006 | 7.602 +/- 2.488 | 0.417 +/- 0.045 | 0.957 +/- 0.013 |
| D: state_v2 + 256,256,128 | low_snr_blockage | -2206.334 +/- 1415.755 | 0.165 +/- 0.015 | 0.723 +/- 0.040 | 0.119 +/- 0.036 | 146.208 +/- 13.200 | 0.533 +/- 0.192 | 0.689 +/- 0.010 |
| D: state_v2 + 256,256,128 | disaster_hotspot | -5.384 +/- 1.644 | 0.085 +/- 0.015 | 0.520 +/- 0.032 | 0.326 +/- 0.030 | 3.338 +/- 0.339 | 1.467 +/- 0.153 | 0.850 +/- 0.005 |
| D: state_v2 + 256,256,128 | edge_overload | 9.146 +/- 0.731 | 0.435 +/- 0.043 | 0.677 +/- 0.003 | 0.043 +/- 0.001 | 7.692 +/- 0.801 | 0.426 +/- 0.016 | 0.954 +/- 0.004 |

## Group Averages Across Scenarios

| group | shaped reward | success | acc LCB | quality gap | Q deadline | Q quality | non-cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: state_v1 + 128,128 | -734.021 +/- 1311.840 | 0.240 +/- 0.182 | 0.637 +/- 0.097 | 0.165 +/- 0.130 | 52.320 +/- 70.744 | 0.817 +/- 0.506 | 0.835 +/- 0.120 |
| B: state_v2 + 128,128 | -784.256 +/- 1345.013 | 0.217 +/- 0.192 | 0.649 +/- 0.111 | 0.155 +/- 0.136 | 61.809 +/- 86.519 | 0.766 +/- 0.543 | 0.809 +/- 0.150 |
| C: state_v2 + 256,256 | -734.290 +/- 1311.596 | 0.228 +/- 0.175 | 0.638 +/- 0.098 | 0.164 +/- 0.131 | 52.518 +/- 70.607 | 0.810 +/- 0.515 | 0.835 +/- 0.118 |
| D: state_v2 + 256,256,128 | -734.191 +/- 1311.558 | 0.228 +/- 0.160 | 0.640 +/- 0.096 | 0.163 +/- 0.129 | 52.412 +/- 70.682 | 0.809 +/- 0.511 | 0.831 +/- 0.116 |

## Interpretation

- Best overall final-20 success among the four groups: `A: state_v1 + 128,128`.
- Best overall final-20 shaped reward among the four groups: `A: state_v1 + 128,128`.
- Lowest overall final-20 semantic quality gap: `B: state_v2 + 128,128`.
- This is a training-trace-only diagnostic. It should guide which architecture to promote into a real evaluation run, not serve as a paper performance table.
- Because `--episodes 0` was used, rollout metrics in the empty result CSVs are intentionally zero and should be ignored.

## Files Kept for Commit

- `train_trace_summary.md`
- Code and tests for state-vector V2 and configurable MLP layers

Model checkpoints, per-run rollout files, and full training traces are intentionally left uncommitted.
