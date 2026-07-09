# EPSILON_RECAL_V4 -- attainability recalibration, iteration 4 (transmission-only anchor)

Task #28 iteration 4.  **VERDICT: FAIL (1/8 criteria)** -- see the criteria table
and the failure-mode analysis at the end.  The v4 anchor is correctly derived and
wired, but it exposes that the peak infeasibility is a **deadline** problem, not a
quality-threshold problem: no epsilon value can fix it.

## Why iteration 4

v3 (`attainability_v3`, 0.504/0.166, cache-compliance FORBIDDEN) proved that the
0.504 anchor was still derived from a distribution that CONTAINED the s0 cache in
the feasible service set (the v2 P10 of the oracle's realised best-*feasible*-
service LCB, oracle free to pick cache).  Once cache is banned from COMPLIANCE,
~44% of the peak all-critical mix (counting-heavy) has NO transmission path
reaching 0.504 -> the peak oracle collapsed to semSucc 0.556 (111/250 critical
tasks fell back to the non-compliant cache) and every learning arm collapsed
(semSucc 0.008, lambda pinned).

## v4 rule: pure-transmission feasible-set anchor

For each PEAK all-critical task we take the best TRANSMISSION-service quality
lower bound with s0 cache excluded (`scripts/calibrate_epsilon_v4.py`):

    best_tx_lcb(task) = max over tx levels {1 token, 2 image} of the candidate
                        semantic_accuracy_lcb at the task's realised sensed-SNR
                        obs (LUT/SNR quantity, deadline-independent)

    eps_critical(v4)  = floor_3dp( P10( best_tx_lcb : peak critical ) )

The max over tx LCB ignores the epsilon bar, so the anchor is anchor-independent
(no circularity).  eps_normal is held at the v1/v3 attainability anchor 0.166
(nominal normal tasks are unaffected by the cache ban).

### Measured distributions (oracle trajectory, 50 ep x 20 tasks, seed base 0)

| distribution | n | min | P05 | P10 | P25 | med | P90 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| PEAK critical best-tx LCB | 250 | 0.355 | 0.355 | **0.355** | 0.355 | 0.772 | 0.772 | 0.772 | 0.587 |
| NOMINAL critical best-tx LCB | 517 | 0.301 | 0.357 | 0.357 | 0.357 | 0.357 | 0.949 | 0.949 | 0.649 |
| NOMINAL normal best-tx LCB (record) | 466 | 0.489 | 0.489 | 0.489 | 0.489 | 0.684 | 0.744 | 0.850 | 0.633 |

Peak by qtype: counting mean 0.355 (n=111, single LUT value 0.355225), presence
mean 0.772 (n=139).  P10 = 0.355225 -> floor_3dp -> **eps_critical = 0.355** (rounded
DOWN so the P10 counting cluster stays >= threshold).

### ATTAINABILITY_V4_EPSILON

```python
ATTAINABILITY_V4_EPSILON = {
    "critical": 0.355,
    "normal": 0.166,
    "high": 0.355,
}
```

Selected via `--epsilon-calibration attainability_v4`, paired with
`--critical-cache-compliance forbidden`.

### Construction / collateral checks

* invariant  eps_critical(v4) 0.355 <= cache-inclusive v3 anchor 0.504  ->  **OK**
* PEAK semSucc-by-construction proxy (frac peak critical best_tx_lcb >= 0.355,
  LCB side only): **1.000**
* NOMINAL critical transmission reachability (frac nominal critical best_tx_lcb
  >= 0.355): **0.992**  (>= 0.90, no diagnostic)
* The construction guarantee holds on the QUALITY axis only; whether the best tx
  service is DEADLINE-feasible in the realised run is adjudicated by the matrix
  oracle (criterion 1) -- and that is exactly where v4 fails (below).

## 5-generation comparison (peak + nominal)

Generations: legacy = `matrix_v1` (0.82/0.65, cache allowed); v1 = `eps_recal_v1`
(0.615/0.166, allowed); v2 = `eps_recal_v2` (0.633/0.297, allowed); v3 =
`eps_recal_v3` (0.504/0.166, FORBIDDEN); v4 = `eps_recal_v4` (0.355/0.166,
FORBIDDEN, tx-only anchor).  proposed/no_lagrangian/fixed_penalty = 3-seed means,
e4lut = 2-seed.  Full table: `outputs/rl/eps_recal_v4/summary_5gen.txt`.

### PEAK (utm_conflict, all-critical)

| arm | gen | eps | semSucc | taskSucc | acc | cacheR | rejectR | ddlVio |
|---|---|---|---|---|---|---|---|---|
| oracle | v3 | 0.504 | 0.556 | 0.000 | 0.677 | 0.444 | 0.000 | 0.588 |
| oracle | **v4** | 0.355 | **0.556** | 0.000 | 0.677 | 0.444 | 0.000 | 0.588 |
| always_cache | v4 | 0.355 | 0.000 | 0.000 | 0.383 | 1.000 | 0.000 | 0.000 |
| random | v4 | 0.355 | 0.513 | 0.000 | 0.459 | 0.339 | 0.000 | 0.740 |
| proposed | v3 | 0.500 | 0.008 | 0.008 | 0.218 | 0.382 | 0.610 | 0.009 |
| proposed | **v4** | 0.354 | **0.002** | 0.002 | 0.329 | **0.798** | 0.200 | 0.006 |
| no_lagrangian | v4 | 0.352 | 0.008 | 0.007 | 0.217 | 0.375 | 0.618 | 0.003 |
| fixed_penalty | v4 | 0.352 | 0.008 | 0.008 | 0.217 | 0.380 | 0.613 | 0.008 |
| e4lut | v4 | 0.354 | 0.003 | 0.003 | 0.300 | 0.697 | 0.300 | 0.009 |

Per-seed peak proposed: seed0 cache 0.395 / reject 0.600; seeds 1-2 cache 1.000 /
reject 0.000 -- the collapse mode flipped from v3's reject-dominant to v4's
cache-dominant (cache is still reward-cheapest once nothing can be compliant).

### NOMINAL

| arm | gen | eps | semSucc | taskSucc | acc | cacheR | rejectR | ddlVio |
|---|---|---|---|---|---|---|---|---|
| oracle | v3 | 0.279 | 0.771 | 0.540 | 0.515 | 0.513 | 0.000 | 0.233 |
| oracle | **v4** | 0.211 | **0.771** | 0.540 | 0.515 | 0.513 | 0.000 | 0.233 |
| always_cache | v4 | 0.211 | 0.540 | 0.540 | 0.372 | 0.746 | 0.000 | 0.000 |
| proposed | v1 | 0.332 | 0.897 | 0.779 | 0.448 | 0.626 | 0.000 | 0.118 |
| proposed | v3 | 0.280 | 0.642 | 0.538 | 0.444 | 0.639 | 0.000 | 0.105 |
| proposed | **v4** | 0.211 | **0.577** | 0.537 | 0.400 | 0.706 | 0.000 | 0.040 |
| no_lagrangian | v4 | 0.211 | 0.680 | 0.536 | 0.469 | 0.601 | 0.000 | 0.144 |
| fixed_penalty | v4 | 0.211 | 0.693 | 0.537 | 0.478 | 0.588 | 0.000 | 0.156 |
| e4lut | v4 | 0.211 | 0.599 | 0.539 | 0.413 | 0.686 | 0.000 | 0.060 |

Nominal oracle/always_cache are bit-identical v3 -> v4 (their action streams do
not depend on epsilon below the old bar); the nominal-critical cache-compliance
loss (v1 0.897 -> v3/v4 <=0.64 for learning arms) is caused by the FORBIDDEN
gate, not by the anchor, and therefore does NOT recover under v4.

### Lambda_quality_critical trajectory (peak training)

All 3 proposed seeds: monotone ramp 0.098 -> ~9.6-9.7 at ep 499, zero drop,
quality-cost mean ~0.99 -> **pinned on 3/3 seeds** (same signature as v3; e4lut
identical; fixed_penalty frozen at 9.74 by design; no_lagrangian zero by design).

## V4 success criteria -- verdict

| # | criterion | observed | verdict |
|---|---|---|---|
| 1 | PEAK oracle semSucc >= 0.85 | 0.556 | **FAIL** |
| 2 | PEAK proposed cacheR < 0.40 | 0.798 | **FAIL** |
| 3 | PEAK proposed rejectR < 0.20 | 0.200 | PASS (boundary) |
| 4 | PEAK proposed semSucc >= 0.5 x oracle (0.278) | 0.002 | **FAIL** |
| 5 | PEAK proposed acc >= 0.7 x oracle acc (0.474) | 0.329 | **FAIL** |
| 6 | lambda_critical not pinned (>= 2/3 seeds) | 0/3 | **FAIL** |
| 7 | NOMINAL proposed semSucc >= 0.89 | 0.577 | **FAIL** |
| 8 | NOMINAL proposed task success >= 0.75 | 0.537 | **FAIL** |

**Overall: FAIL (1/8).**

## Failure mode (the important diagnostic)

Criterion 1 was expected to hold "by construction" -- it does not, and the gap is
informative.  The v4 anchor closes the QUALITY seam completely (peak
reach-by-LCB = 1.000 at 0.355), yet the peak oracle still serves exactly the same
action stream as v3: 139/250 presence-critical via token (all compliant), 111/250
counting-critical via cache (all non-compliant).  Rollout evidence
(`outputs/rl/eps_recal_v4/bl_oracle_best_feasible_evidence_0`): for every
counting-critical task the selected tx path is **deadline-infeasible**
(`selected_path_deadline_feasible = False` on all tx probes; the oracle's
feasibility search rejects token/image on the DEADLINE axis, then falls back to
cache).  In other words:

> The 44% counting-critical tranche is deadline-blocked for ALL transmission
> services under the peak load pattern.  The binding constraint is tau_k /
> queueing, not epsilon_k.  Lowering epsilon from 0.504 to 0.355 moves the
> quality bar below the counting tx LCB (0.355225) but cannot create a
> transmission that completes before the deadline.  There is NO epsilon in
> (0, 1] that makes the peak tier feasible while cache compliance is forbidden.

Consequences seen in the matrix:
* oracle semSucc invariant at 0.556 across v3 -> v4 (quality bar irrelevant);
* learning arms still constraint-infeasible -> lambda ramps to the ceiling and
  pins (3/3 seeds), the policy collapses to the cheapest non-compliant action
  (all-cache for 2 seeds, reject-heavy for 1);
* nominal learning arms stay depressed (0.58-0.69 vs v1's 0.90) because the
  nominal 50%-critical mix ALSO loses its cache-compliance path under the
  forbidden gate -- an anchor change cannot restore it.

## Candidate ways out (for human decision -- NOT auto-started)

1. **Deadline-side repair (attack the true binding constraint):** relax peak
   tau_scale / queue model for counting-critical, or add a deadline-aware
   admission rule so counting tasks are servable by token within tau_k.  The
   epsilon machinery (v4 anchor) is then already correct.
2. **Risk-conditional compliance channel:** allow cache compliance for
   critical tasks IF cache LCB >= epsilon AND freshness == fresh (a verified-
   cache path), keeping the blanket s0 shortcut closed.  Restores both the peak
   counting tranche and the nominal critical tier.
3. **Constraint semantics change:** treat the quality constraint per-*servable*
   task (exclude deadline-infeasible tasks from the quality-cost denominator),
   so lambda measures achievable violation and can turn; pair with an explicit
   deadline-violation dual.
4. **Mix redesign:** accept that all-critical + counting-heavy + peak load is
   outside the designed operating envelope and re-specify the peak scenario
   (fewer counting-critical or longer tau for image service).

## Five-generation constant history

| gen | eps_critical | eps_normal | cache gate | anchor rule |
|---|---|---|---|---|
| legacy | 0.82 | 0.65 | allowed | hand-set constants |
| v1 | 0.615 | 0.166 | allowed | 0.90 x oracle LCB ceiling / 0.75 x ceiling |
| v2 | 0.633 | 0.297 | allowed | P10 peak realised LCB + cache-P90 guardrail (guardrail binds -> contradiction) |
| v3 | 0.504 | 0.166 | **forbidden** | P10 peak realised LCB (cache-inclusive), guardrail dropped |
| v4 | **0.355** | 0.166 | **forbidden** | floor_3dp(P10) peak best-TRANSMISSION LCB, cache excluded |

Artifacts: `scripts/calibrate_epsilon_v4.py`, `scripts/eps_recal_v4_runs.sh`,
`scripts/summarize_eps_recal_v4.py`, `outputs/rl/eps_recal_v4/`,
`tests/test_epsilon_calibration.py` (20 tests, 4 new v4 guards incl. the
pure-tx <= cache-inclusive invariant).
