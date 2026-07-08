# EPSILON_RECAL_V2 -- attainability recalibration, iteration 2

Task #28 iteration 2.  v1 (`attainability_v1`, 0.615/0.166) fixed the
lambda ratchet and the nominal tier but left the PEAK tier essentially
infeasible (oracle satisfied only 65% of critical tasks -> learning arm
collapsed onto the cache-compliance shortcut).  v2 re-derives the two
constants by (1) quantile anchoring and (2) a cache-ceiling guardrail.

## Rule (1): quantile anchoring

Distribution = per-task best-*feasible*-service LCB (realised
`semantic_accuracy_lcb` of the oracle_best_feasible_evidence baseline,
whose accuracy model IS the 3D LUT `outputs/lut/v2_0_lut_3d.csv`, after
deadline / payload / freshness / view degradation).

* PEAK (utm_conflict, all critical) distribution: n=250 min=0.504 P10=0.504 P25=0.504 med=0.772 P90=0.772 max=0.772 mean=0.677
    -> eps_critical anchor = P10 = **0.504**  (target: oracle ~90%)
* NOMINAL normal-risk distribution:              n=281 min=0.297 P10=0.297 P25=0.297 med=0.360 P90=0.430 max=0.720 mean=0.361
    -> eps_normal anchor   = P25 = **0.297**  (target: oracle ~75%)

Pure-LUT best-service ceiling (condition-blind cross-check, max over
service of mean-over-SNR Wilson LCB):
    co_presence  0.691
    comparison   0.839
    counting     0.640
    presence     0.744
    threshold    0.598

## Rule (2): cache-ceiling guardrail

* PEAK bl_always_cache measured cache accuracy: n=250 min=0.151 P10=0.151 P25=0.151 med=0.504 P90=0.583 max=0.750 mean=0.383
* cache_accuracy_P90 = 0.583 ; margin = 0.05
* guardrail floor = P90 + margin = **0.633**
* eps_critical = max(P10 anchor 0.504, guardrail 0.633) = **0.633**
* guardrail BINDS (> anchor)

## ATTAINABILITY_V2_EPSILON

```python
ATTAINABILITY_V2_EPSILON = {
    "critical": 0.633,
    "normal": 0.297,
    "high": 0.633,
}
```

## Consistency note

WARNING: the cache guardrail floor (0.633) exceeds the
attainability P10 anchor (0.504).  The peak critical mix is
~44% `counting`, whose best-feasible path is the cache/local (service
0) because its semantic-transmit accuracy (0.27-0.48) is far below any
guardrail-compliant threshold, while service-2 full transmit is
deadline-blocked.  Forcing eps_critical >= cache_P90+margin therefore
makes counting-critical structurally infeasible (cannot cache, cannot
transmit to spec) -- i.e. the two v2 rules are mutually contradictory
for this workload.  Recorded for human decision; NOT auto-fixed.

## Validation rerun verdict (outputs/rl/eps_recal_v2, 2026-07-08)

Same arm matrix as v1 (proposed / no_lagrangian / fixed_penalty x3 seeds,
e4lut x2, non-learning baselines; peak=utm_conflict + nominal), all under
`--epsilon-calibration attainability_v2`.

Success criteria: **1 / 6 PASS** -- v2 FAILS.

| criterion | obs | verdict |
|---|---|---|
| PEAK oracle semSucc >= 0.85 (construction guarantee) | 0.652 | FAIL |
| PEAK proposed cache ratio < 0.40 | 0.545 | FAIL |
| PEAK proposed average_accuracy >= 0.45 | 0.276 | FAIL |
| PEAK proposed lambda_critical not pinned (>=2/3 seeds) | 0/3 | FAIL |
| NOMINAL proposed semSucc >= 0.90 | 0.846 | FAIL |
| NOMINAL proposed task success >= 0.75 | 0.773 | PASS |

Failure form: SAME collapse as v1, not a new pathology.  The binding cache
guardrail (0.633 > P10 anchor 0.504) keeps the peak tier infeasible by
construction for the ~44% counting-critical sub-mix (cache capped at 0.583
P90; semantic transmit 0.27-0.48; service-2 deadline-blocked), so the oracle
tops out at 0.652 and the learning arm has no compliant gradient to follow:
lambda_quality_critical ramps monotonically 0.06-0.10 -> ~8.4 (dual ceiling,
never turns around, all 3 seeds) and the policy stays on the cache shortcut
(cacheR 0.55, acc 0.28).  Nominal semSucc regressed 0.897 -> 0.846 because
eps_normal rose 0.166 -> 0.297 while the nominal-eval epsilon mix (0.376 avg)
now sits above what the cache-heavy nominal policy delivers.

Diagnosis for the human decision (NO third calibration round attempted):
rule (1) and rule (2) are mutually contradictory under the current peak task
mix and LUT -- any eps_critical that defeats the cache shortcut (>0.633)
is unattainable for counting-critical tasks, and any attainable one (<=0.504)
is below the cache ceiling.  Candidate resolutions (pick one upstream):
(a) make epsilon question-type-conditional (counting gets its own tier);
(b) change the peak mix / LUT so counting-critical has a compliant transmit
    path (e.g. service-2 deadline relief);
(c) drop rule (2) and instead block cache-only compliance structurally
    (e.g. forbid cache path for critical tasks) rather than via epsilon.
