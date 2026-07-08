# EPSILON_RECAL_V3 -- attainability recalibration, iteration 3 (structural cache-compliance ban)

Task #28 iteration 3.  Method **(c)** from the v2 human-decision menu:
*close the cache-only compliance shortcut structurally instead of via epsilon*,
then set epsilon back to the pure attainability anchor.

## Why v2 failed (recap)

v2 tried to defeat the cache shortcut with a **cache-ceiling guardrail**
(`eps_critical = max(P10 anchor 0.504, cache_P90+margin 0.633) = 0.633`).  The
guardrail (0.633) bound *above* the attainability anchor (0.504), and the two
rules were mutually contradictory for the counting-heavy peak mix: any
`eps_critical` that defeats the cache shortcut (>0.633) is unattainable, any
attainable one (<=0.504) sits below the cache ceiling.  v2 FAILED 1/6.

## v3 design: structural ban (method c) + pure attainability epsilon

Two coupled changes, both gated by a new env config so legacy/v1/v2 stay
bit-for-bit reproducible:

1. **Compliance-judgment layer (not just reward).**  New env config
   `critical_cache_compliance: allowed | forbidden` (default `allowed`).  Under
   `forbidden`, a **critical/high** task served by the s0 **cache-only** path is
   *never* counted as quality-compliant, even when the cached answer's LCB
   clears `epsilon_k`.  Implemented at BOTH judgment layers that exist in this
   codebase:
   * `MultiUAVVQAEnv.evaluate_action` (env truth: drives the env reward penalty,
     the oracle's env-side `_candidate_path_metric` feasibility search, and
     `candidate_mobility_metrics`);
   * `V19LUTResourceEnv._enrich_semantic_info` (the **authoritative RL record
     layer**, which recomputes `semantic_success`/`quality_violation` from the
     calibrated LUT/persample estimate and would otherwise *clobber* the env
     flag -- this is what feeds the CSV metrics, `info["success"]`, the binary
     Lagrangian quality cost, and the oracle feasibility search through
     `evaluate_action`).
   The Lagrangian quality cost is binary in `quality_violation`, so the ban is
   priced by lambda_quality_critical automatically.

2. **Safety-projection layer.**  Three cache-*downgrade* sites in `v19_ppo.py`
   were taught the ban via `_critical_cache_forbidden(env, obs)` (and the mirror
   `V19LUTResourceEnv._critical_cache_compliance_forbidden`):
   `_project_path_mobility_action`, `_deadline_token_cache_fallback_action`,
   `_projected_deadline_downgrade_action`.  For a critical/high task under the
   ban they never downgrade *onto* the s0 cache path (a guaranteed violation);
   they fall through to the cheapest feasible **transmission** (token) and
   otherwise leave lambda to price the residual infeasibility.

3. **Epsilon.**  Guardrail (rule 2 of v2) **dropped**.  `eps_critical` returns
   to the pure attainability anchor **P10 = 0.504** (peak all-critical
   best-feasible-service LCB distribution, `calibrate_epsilon_v2.py`);
   `eps_normal` restored to the v1 anchor **0.166** (v2's 0.297 had caused a
   pure nominal regression 0.897 -> 0.846).

```python
ATTAINABILITY_V3_EPSILON = {"critical": 0.504, "normal": 0.166, "high": 0.504}
```

### Design tradeoff: no hard action mask on cache

We deliberately did **not** hard-mask "critical -> cache" in the action space.
A hard mask would (i) kill exploration and (ii) break the Lagrangian-pricing
narrative and the ablations: the `no_lagrangian` arm must be *able* to pick
cache so it can be shown to collapse without the dual, and `fixed_penalty` must
see the same action set.  Instead the ban lives at the compliance-judgment
layer (so cache is *selectable* but *non-compliant*), and lambda prices it.  The
projection layer only refuses to *inject* a cache downgrade it knows is a
guaranteed violation.

## Four-generation comparison (peak = utm_conflict all-critical; proposed = 3-seed mean)

### PEAK

| arm | gen | eps | semSucc | qViol | task | acc | cacheR |
|---|---|---|---|---|---|---|---|
| oracle | legacy | 0.820 | 0.000 | 1.000 | 0.000 | 0.684 | 0.000 |
| oracle | v1 | 0.615 | 0.652 | 0.348 | 0.096 | 0.677 | 0.444 |
| oracle | v2 | 0.633 | 0.652 | 0.348 | 0.096 | 0.677 | 0.444 |
| **oracle** | **v3** | **0.504** | **0.556** | **0.444** | **0.000** | **0.677** | **0.444** |
| always_cache | v1 | 0.615 | 0.096 | 0.904 | 0.096 | 0.383 | 1.000 |
| always_cache | v2 | 0.633 | 0.096 | 0.904 | 0.096 | 0.383 | 1.000 |
| **always_cache** | **v3** | **0.504** | **0.000** | **1.000** | **0.000** | **0.383** | **1.000** |
| proposed | legacy | 0.820 | 0.000 | 1.000 | 0.000 | 0.563 | 0.000 |
| proposed | v1 | 0.608 | 0.100 | 0.900 | 0.097 | 0.261 | 0.491 |
| proposed | v2 | 0.629 | 0.100 | 0.900 | 0.099 | 0.276 | 0.545 |
| **proposed** | **v3** | **0.500** | **0.008** | **0.992** | **0.008** | **0.218** | **0.382** |
| no_lagrangian | v3 | 0.499 | 0.008 | 0.992 | 0.008 | 0.218 | 0.378 |
| fixed_penalty | v3 | 0.501 | 0.008 | 0.992 | 0.007 | 0.223 | 0.395 |
| e4lut | v3 | 0.500 | 0.006 | 0.994 | 0.006 | 0.217 | 0.380 |

### NOMINAL

| arm | gen | eps | semSucc | qViol | task | acc | cacheR |
|---|---|---|---|---|---|---|---|
| oracle | v1 | 0.330 | 0.995 | 0.005 | 0.780 | 0.510 | 0.529 |
| oracle | v2 | 0.376 | 0.995 | 0.005 | 0.764 | 0.515 | 0.513 |
| **oracle** | **v3** | **0.279** | **0.771** | **0.229** | **0.540** | **0.515** | **0.513** |
| proposed | v1 | 0.332 | 0.897 | 0.103 | 0.779 | 0.448 | 0.626 |
| proposed | v2 | 0.376 | 0.846 | 0.154 | 0.773 | 0.422 | 0.666 |
| **proposed** | **v3** | **0.280** | **0.642** | **0.358** | **0.538** | **0.444** | **0.639** |

## Lambda_quality_critical trajectory (v3 peak training)

All arms **pinned** (no interior turn-around):
`proposed_{0,1,2}` ramp monotonically 0.098 -> ~9.5 and stick at the dual
ceiling (cost_mean ~0.98); `no_lagrangian` stays 0 (dual disabled);
`fixed_penalty` frozen at 9.74.  0/3 proposed seeds turn around -- the classic
infeasible-constraint signature, unchanged from v2.

## Success criteria verdict: **1 / 7 PASS -- v3 FAILS**

| criterion | obs | verdict |
|---|---|---|
| PEAK oracle semSucc >= 0.85 (structural rule) | 0.556 | FAIL |
| PEAK proposed cache ratio < 0.40 | 0.382 | **PASS** |
| PEAK proposed average_accuracy >= 0.45 | 0.218 | FAIL |
| PEAK proposed semSucc off the 0.10 band (>= 0.25) | 0.008 | FAIL |
| PEAK proposed lambda_critical not pinned (>=2/3 seeds) | 0/3 | FAIL |
| NOMINAL proposed semSucc >= 0.89 | 0.642 | FAIL |
| NOMINAL proposed task success >= 0.75 | 0.538 | FAIL |

## Diagnosis (for human decision -- NO fourth round attempted)

**The structural ban is mechanically correct and does exactly what it should:**
it closes the cache shortcut cleanly (unlike v2's self-contradictory guardrail).
Evidence: `always_cache` peak semSucc collapses **0.096 (v1/v2) -> 0.000 (v3)**
-- a pure-cache policy now scores *zero* compliant critical tasks; and proposed
peak cache ratio falls 0.545 (v2) -> 0.382 (v3), clearing the <0.40 bar.

**But closing the shortcut does not create feasibility -- it exposes that the
0.504 anchor was itself cache-inflated.**  The P10=0.504 anchor was computed as
the P10 of the oracle's realised best-feasible-service LCB *with cache in the
feasible set*.  Remove cache and the oracle's own peak semSucc drops
0.652 -> 0.556: ~44% of critical tasks (the counting-heavy sub-mix) have **no
transmission path** that reaches 0.504 (semantic-transmit LCB 0.27-0.48;
service-2 image deadline-blocked; image ratio 0.00 for the oracle).  With the
cache exit sealed and those tasks now *unavoidably* quality-violating, the
learning arm no longer collapses onto cache (v2) -- it collapses onto **reject**
(proposed peak reject ~0.6, semSucc 0.008, acc 0.218), and
lambda_quality_critical ramps to the dual ceiling pinned in all 3 seeds because
the constraint has no feasible gradient.

**Collateral nominal damage.**  The nominal mix also contains critical tasks
that previously cleared via cache; the ban hits them too, and the peak-trained
reject-prone policy carries over, so nominal proposed regresses
semSucc 0.897 (v1) -> 0.642 (v3) and task 0.779 -> 0.538.  (Oracle nominal also
drops 0.995 -> 0.771 for the same reason.)

### Candidate resolutions (pick one upstream; do not stack another epsilon round)

1. **Re-anchor epsilon to the *transmission-only* attainability distribution.**
   Recompute the P10 anchor from best-feasible-service LCB with cache EXCLUDED
   from the feasible set (consistent with the ban).  This yields a strictly
   lower, genuinely attainable `eps_critical`; the current 0.504 is internally
   inconsistent with `forbidden` because it was measured with cache feasible.
2. **Question-type-conditional epsilon.**  Give `counting` its own lower tier
   (~0.45, the transmission ceiling) while other types keep 0.504.  Keeps the
   ban; makes the counting sub-mix feasible by transmission.
3. **Give counting-critical a compliant transmit path.**  Relax the service-2
   image deadline (or add a mid-tier) at peak so 0.504 becomes attainable
   without cache.
4. **Restrict the ban to non-counting critical tasks** (accept cache for
   counting where transmission genuinely cannot comply), i.e. combine (2)+ban.

The ban machinery (config-gated, tested, reproducible) is now in place and can
be reused as-is under any of these; only the epsilon derivation / task mix needs
the upstream fix.

## Provenance

* Constants: `ATTAINABILITY_V3_EPSILON` in `src/vqa_semcom/sim/multi_uav_env.py`.
* Anchor 0.504: `scripts/calibrate_epsilon_v2.py` (rule-1 P10, reused).
* Runs: `outputs/rl/eps_recal_v3/` (34-run matrix, `scripts/eps_recal_v3_runs.sh`).
* Summary/criteria: `scripts/summarize_eps_recal_v3.py`.
* Tests: `tests/test_epsilon_calibration.py` (20 total: v3 values + gate
  allowed/forbidden behaviour + critical cache-only violation + legacy/v1/v2
  regression guards).
