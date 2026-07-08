# E4 quality-backend A/B -- harvest analysis

Driver: `scripts/e4_quality_backend_ab.sh` (commit a5697f0). Arms: `lut`
(calibrated 3-D SNR x semantic x quality LUT) vs `persample` (per-sample
predictor), 3 seeds, two-timescale constrained PPO on `configs/v1_9_bubbles.yaml`,
each checkpoint evaluated under BOTH conditions: train-time `utm_conflict`
(peak) + `nominal` re-eval. Gate: persample average_accuracy >= lut with no
constraint-metric degradation.

## 500-episode group (COMPLETE, 3 seeds x 2 arms x 2 conditions)

| condition | arm | acc LCB | acc mean | quality_vio | payload KB | lambda_q |
|---|---|---|---|---|---|---|
| utm_conflict (PEAK) | lut       | 0.283 +- 0.012 | 0.346 +- 0.015 | 1.000 | 0.012 | 4.814 |
| utm_conflict (PEAK) | persample | 0.254 +- 0.039 | 0.307 +- 0.050 | 1.000 | 0.017 | 4.802 |
| nominal             | lut       | **0.510 +- 0.029** | 0.528 | **0.290** | 13.0 | - |
| nominal             | persample | 0.426 +- 0.041 | 0.444 | 0.507 | 32.5 | - |

## 1000-episode group (INCOMPLETE -- agent killed mid-run)
Only `lut` seed 0 + its nominal re-eval finished; `persample` seed 0 was killed
(empty `run.log`, 0 bytes). Not a usable A/B, but the lut arm confirms the trend:
- lut nominal acc LCB **0.548** (vs 0.510 @ 500ep), quality_vio 0.228 (vs 0.290).
- LUT keeps improving with more training on the informative (nominal) condition.

## VERDICT: persample FAILS the gate -- keep the calibrated LUT
- **PEAK (utm_conflict)**: both arms collapse to the same near-degenerate
  reject/defer policy (quality_vio saturated at 1.0, lambda_quality pinned ~4.8).
  The peak condition is uninformative for the A/B -- the quality constraint is
  unsatisfiable there, so the dual dominates and both backends converge. Prior
  single-seed read "0.2703 vs 0.2695" was within noise; over 3 seeds lut 0.283 >
  persample 0.254, but the honest statement is "tie at a degenerate policy".
- **NOMINAL (the differentiating condition)**: LUT wins cleanly on every axis --
  accuracy +0.084 LCB (0.510 vs 0.426), HALF the quality-violation rate
  (0.290 vs 0.507), and 2.5x lighter payload (13.0 vs 32.5 KB). The per-sample
  predictor is both less accurate and less constraint-compliant.
- Recommendation: retain the calibrated 3-D LUT as the quality backend; the
  per-sample predictor does not pay for itself.

## Diagnostic probes (original agent's anomaly hunt) -- artifact ruled out
The peak-condition two-arm collapse is a REAL scenario property, not numerical noise:
- `cpu_probe_lut_0`: CPU re-run reproduces acc LCB 0.2703 exactly (== 1000ep GPU
  lut seed0) -> GPU nondeterminism is NOT the cause.
- `hashseed_probe_0` vs `hashseed_probe_1`: bit-identical outputs (acc 0.506,
  payload 81.26) across two PYTHONHASHSEED values -> hash-seed nondeterminism is
  NOT the cause.
Conclusion: under `utm_conflict` the quality constraint is infeasible, lambda_quality
saturates, and any quality backend is driven to the same reject-heavy policy. The
A/B is only meaningful on `nominal`, where LUT is decisively better.

## Products
`outputs/rl/e4_quality_backend_ab_summary.csv`, `scripts/e4_aggregate.py`,
per-run `v1_9_resource_alloc_results.csv` / `_summary.md` / `ppo_lambda_trace.csv`
(rollout.csv + .pt gitignored).
