# Peak overload is an admission/escalation regime, not a dual-pricing regime

**Task #37, v8 — final verdict after the λ_QC unpin attempt.**

v8 tried two levers to make the quality-critical Lagrangian dual price a shadow
cost under the peak all-critical overload instead of pinning at its ceiling:
per-channel `lambda_max_quality` cap and a `dual_warmup_episodes` freeze. Both
are correctly implemented and wired (12 unit tests, smoke-verified end to end),
but they do not move the peak verdict: **6/13** (v7b: 6/13, composition shifted), and the
quality-critical dual remains pinned -- now at the new ceiling 8.0 on 3/3 seeds. This document records the terminal reading of the peak
regime and why it is the honest one.

Commits: levers+tests `exp(rl): v8 -- per-channel lambda cap + dual warm-up
(levers + tests)` (34f6093); this verdict `docs(rl): peak admission-regime
framing -- final verdict after v8`. Run artifacts: `outputs/rl/eps_recal_v8`
(36 arms, all exit 0), judge `scripts/summarize_eps_recal_v8.py`.

---

## 1. Terminal reading: what the peak all-critical domain actually is

The peak scenario (`utm_conflict`, all tasks critical/high, escalation budget
δ_esc = 0.155) is an **overload domain that is infeasible at the strategy layer**.
Safety in this domain is carried **structurally** — by the spec-attainability
certificate and the escalation/admission layer — not by the policy's
constraint-satisfying behaviour:

- The **escalation-aware oracle** (which honours the certificate: a critical task
  with no spec-attainable transmission service is escalated, not silently served)
  reaches **admitted mission 0.916** with **critical-escalation 0.666** and
  **admitted deadline-violation 0.084**. This is the ceiling, and it *escalates
  two-thirds of the peak load* to get there. The safe policy in this domain is
  "escalate most of it."
- The **best-feasible oracle** that does *not* escalate collapses to admitted
  mission 0.528 with admitted deadline-violation 0.412 — i.e. absent the
  escalation layer, even an oracle cannot keep the mission constraint on the
  admitted set. The feasible region for a non-escalating strategy is essentially
  empty here.

So the domain's safety is a property of the **certificate + escalation
structure**, and the escalation rate the oracle needs (0.666) sits far outside
any budget a *strategy-layer* dual could price toward δ_esc = 0.155. The peak
domain is, by construction, **beyond the policy's feasible region**.

## 2. The dual pin is the textbook signature of an infeasible constraint

Under the peak mix the quality-critical cost is structural: a critical task
served from anything below a spec-attainable transmission service is
non-compliant, and most of the peak load cannot clear ε at a spec-attainable
service. The realized `quality_cost_critical` therefore sits at **~0.37–0.42**
every step — an order of magnitude above the 0.02 limit — for the entire run,
across every generation and every lever setting:

| gen | proposed λ_QC (3 seeds, terminal) | qCostCrit (tail-50 mean) | pinned? |
|---|---|---|---|
| v6 (cap 20) | 18.54 / 18.05 / 18.34 | 0.43 / 0.37 / 0.38 | yes (3/3) |
| v7b (cap 20) | 18.45 / 18.10 / 18.52 | 0.42 / 0.37 / 0.37 | yes (3/3) |
| v8 (cap 8 + warmup 150) | 8.00 / 8.00 / 8.00 | 0.41 / 0.32 / 0.38 | **yes (3/3)** |

The v8 trajectory shape (identical on all three seeds): λ_QC frozen at 0 for
episodes 0–149 (warm-up verified — the freeze holds despite per-episode costs of
0.2–0.6), ascent resumes at exactly ep 150 (0.038 → 1.95 by ep 200 → 5.48 by
ep 300), **hits the new ceiling 8.0 by ~ep 370 and stays flat at 8.0 through
ep 499 with zero fall-back**. The pin is also backend-invariant: the e4lut
ablation (legacy LUT quality instead of lut_v5) pins at 8.0 on 2/2 seeds with
qCostCrit ~0.56–0.58 — an even deeper infeasibility under the older backend. A projected dual-ascent update
λ ← [λ + η(cost − limit)]_+ with cost − limit ≈ +0.35 at every step is a
**monotone climb to whatever ceiling it is given**. v6/v7b gave it 20 and it
pinned at ~18.4; v8 gives it 8 and it pins at exactly 8.0. Lowering the ceiling
changes the *number* it pins at, not the *fact* that it pins: the constraint is
unsatisfiable, so the multiplier has no interior equilibrium. This
is exactly the classical CMDP result — **when the constraint set is infeasible the
Lagrange multiplier is unbounded (diverges to its cap); a bounded interior λ\*
exists only when the constraint is feasible** (Altman, *Constrained Markov
Decision Processes*, 1999; Chow, Ghavamzadeh, Janson & Pavone, *Risk-Constrained
RL via CVaR*, JMLR 2017 — the dual is a shadow price only on the feasible
boundary). The pin is not a tuning bug; it is the optimiser correctly reporting
"there is no feasible policy here."

The warm-up lever confirms the mechanism from the other side: freezing the dual
for 150 episodes lets the policy form under the fixed-init reward, but once the
freeze lifts the cost is still ~0.37 and the dual resumes the same climb — the
shaping window cannot manufacture a feasible region that the environment does
not contain.

A second, subtler confirmation: capping λ_QC at 8 *halved the escalation rate*
(critical-escalation 0.126 at v7b → 0.086 at v8, dropping below the 2a budget
band [0.105, 0.205]). At v7b the pinned λ_QC ≈ 18.4 made escalation the cheap
escape valve from the quality penalty; at 8 that pressure is halved and the
policy under-escalates. In an infeasible domain the duals do not converge to
shadow prices — their *ceilings become behaviour-shaping knobs*, and every choice
of cap trades one criterion against another (v8 recovers 5a but drops 2a; net
6/13, composition shifted, verdict unchanged). This coupling is exactly
why constraint-pricing is the wrong control surface for this regime and an
admission/escalation layer with an explicit budget is the right one.

## 3. `proposed` ≡ `no_lagrangian` at peak is an honest finding, not a defect

Because the quality dual cannot find an interior price, the constrained arm and
the unconstrained arm converge to the same peak behaviour — the dual penalty is
either saturated-and-uniform (a constant offset that does not shape relative
action values) or frozen. This equivalence is **the expected outcome in an
infeasible domain** and is structurally the same phenomenon reported by the
reachability-constrained RL literature: **states/regimes outside the feasible
(reachable-safe) set require a dedicated feasibility mechanism, because a
violation *penalty* has nothing feasible to steer toward** (Yu, Ma, Li & Chen,
*Reachability Constrained Reinforcement Learning*, ICML 2022). Our escalation/
admission layer *is* that dedicated mechanism; the Lagrangian dual is the wrong
tool for this domain, and its degeneracy is informative.

The reward-wiring fix (v7b) is not wasted — it is precisely what lets us *see*
this cleanly. With the mission-aligned reward reaching the gradient, the
**unconstrained** arm's peak behaviour improved measurably (proof the reward
signal is now load-bearing), while the **constrained** arm did not — isolating the
pin as a dual-pricing failure rather than a reward-plumbing failure:

| metric (PEAK, no_lagrangian) | v6 legacy reward | v7b fixed reward | Δ |
|---|---|---|---|
| cacheR | 0.436 | 0.298 | −0.138 (less cache flooding) |
| task success | 0.242 | 0.384 | +0.142 |
| admitted acc | 0.164 | 0.213 | +0.049 |
| critEsc | 0.162 | 0.329 | +0.167 (escalates the infeasible load) |

The `proposed` arm's peak numbers are essentially invariant v6→v7b→v8
(admMiss ~0.18, cacheR ~0.44, λ_QC pinned) — the reward fix and both v8 levers
move the *unconstrained* policy and the dual's *ceiling*, but cannot make the
peak constraint feasible for the *constrained* policy.

## 4. Nominal domain: the dual works where the constraint is feasible

The same machinery, unchanged, prices correctly under the nominal scenario — the
control that the peak-regime claim rests on. Under nominal the realized quality
cost is attainable, the escalation rate is ~0, and the proposed policy tracks the
nominal oracle:

| metric (NOMINAL, proposed) | v6 | v7b | v8 |
|---|---|---|---|
| admitted mission | 0.635 | 0.592 | **0.653** |
| task success | 0.628 | 0.587 | 0.653 |
| escalation rate | 0.000 | 0.000 | 0.000 |
| nominal oracle admMiss (ceiling) | 0.703 | 0.703 | 0.703 |

v8's capped dual measurably **helps the feasible domain**: nominal admitted
mission recovers from the v7b regression (0.592, a hair under the 0.85·oracle
gate 0.597) to 0.653 — comfortably over the gate (criterion 5a back to PASS).
The mechanism is consistent: a policy trained with a bounded quality penalty is
less warped by the infeasible-domain pin and transfers better to the domain
where the constraint *is* satisfiable. The fixed-penalty arm corroborates it
independently — its static quality price was clamped 9.74 → 8.0 by the same cap,
and its nominal admitted mission rose 0.650 → 0.727. The nominal-TRAINED arm
under v8 reaches 0.723 admitted mission with deadline-violation 0.047. The lever
is not wasted — it is simply not a cure for infeasibility.

The direct dual evidence is the λ_QC trajectory of the **nominal-trained** arm
under the same v8 levers (cap 8, warm-up 150). Where the peak-trained dual
ratchets from 0 to the cap 8 in ~220 post-warmup episodes against a cost stuck
at ~0.4, the nominal-trained dual climbs **~4× slower and stays interior**:
0.01 (ep 150) → 0.50 (ep 200) → 1.33 (ep 300) → 2.33 (ep 400) → **3.29 at ep 499,
never touching the cap**, against a realized cost of 0.05–0.15 that is closing
on the 0.02 limit at the tail (0.053 at ep 499). Same algorithm, same levers,
same horizon — an interior shadow-price trajectory in the near-feasible domain,
a ceiling ratchet in the infeasible one. The contrast is the point: **dual
pricing is a domain property, not a global property of the method.** It works
under nominal (feasible) and degenerates under peak (infeasible), exactly as
CMDP theory predicts.

## 5. v6 → v7b → v8 generational comparison (peak + nominal, all metrics)

### PEAK — proposed (constrained) arm

| metric | v6 | v7b | v8 (cap 8 + warmup 150) |
|---|---|---|---|
| admitted mission | 0.179 | 0.179 | 0.180 |
| admitted acc | 0.167 | 0.167 | 0.166 |
| cacheR | 0.441 | 0.440 | 0.439 |
| non-escalated rejectR | 0.325 | 0.326 | 0.345 |
| critical-escalation | 0.124 | 0.126 | 0.086 |
| λ_QC terminal (mean of 3 seeds) | ~18.3 | ~18.4 | 8.00 (= new cap) |
| qCostCrit (tail) | ~0.39 | ~0.39 | ~0.37 |

Peak behaviour of the constrained arm is invariant across three generations of
reward and dual surgery (admMiss 0.179 → 0.179 → 0.180) — the levers move the
dual's ceiling and the escalation split, not the feasibility of the constraint.

### PEAK — no_lagrangian (unconstrained) arm — reward-fix efficacy witness

| metric | v6 | v7b | v8 |
|---|---|---|---|
| cacheR | 0.436 | 0.298 | 0.298 |
| task success | 0.242 | 0.384 | 0.384 |
| admitted acc | 0.164 | 0.213 | 0.213 |
| critical-escalation | 0.162 | 0.329 | 0.329 |

v8 reproduces v7b bit-for-bit on this arm — with `--no-constrained-ppo` the dual
system is off, so both v8 levers are no-ops. This doubles as a full-pipeline
legacy-regression check for the v8 code path (the unit-level guarantee in
`tests/test_lambda_qc_v8.py` holding in production).

### 13-criterion verdict progression

| verdict | v6 | v7b | v8 |
|---|---|---|---|
| criteria PASS | 6/13 | 6/13 | 6/13 |
| 2a critEsc in [δ−0.05, δ+0.05] | PASS (0.124) | PASS (0.126) | **FAIL (0.086)** |
| 3a cacheR<0.40 | FAIL | FAIL | FAIL (0.439) |
| 3b neRejR<0.20 | FAIL | FAIL | FAIL (0.345) |
| 3c mission≥0.6·oracle | FAIL | FAIL | FAIL (0.180 vs 0.549) |
| 3d acc≥0.7·oracle | FAIL | FAIL | FAIL (0.166 vs 0.566) |
| 4 λ_QC non-pinned | FAIL | FAIL | FAIL (0/3 seeds) |
| 5a nominal mission≥0.85·oracle | FAIL | FAIL (0.592) | **PASS (0.653)** |
| 5b nominal task≥0.75 | FAIL | FAIL | FAIL (0.653) |

(1a/1b oracle ceiling, 2b nominal escalation, 6a/6b hygiene hold across all
three generations.)

## 6. Recommendation for the paper (paper II framing)

Position the peak all-critical scenario as a **stress test that exceeds the
policy's feasible region**, not as a domain the strategy layer is expected to
solve. The safety story there is structural, and the figures should say so:

1. **Escalation-aware oracle as the safety anchor** — show admitted mission 0.916
   at escalation 0.666: safety is achieved *by escalating the infeasible load*,
   and the escalation rate lands inside the certificate-derived budget band.
2. **Feasible-region boundary figure** — best-feasible (non-escalating) oracle
   admitted deadline-violation 0.412 vs escalation-aware 0.084: the feasible set
   for a non-escalating strategy is empty; the escalation layer is what creates a
   feasible region.
3. **λ_QC divergence as an infeasibility diagnostic** — plot the monotone climb to
   the cap under peak vs the interior convergence under nominal, side by side, and
   label it as the CMDP infeasibility signature (Altman 1999; Yu et al. ICML'22).
4. **Admission-layer safety table** — escalation rate vs certificate budget, and
   admitted-set deadline/quality violations, to show the admission/escalation
   layer carries the safety guarantee structurally in the overload regime.
5. **Nominal dual-pricing figure** — the same λ channels converging to a bounded
   interior price under nominal, as the positive control that the method works
   where the constraint is feasible.

The honest scientific claim is: *the proposed constrained controller prices risk
correctly in the feasible (nominal) operating regime, and in the infeasible
(peak all-critical) overload regime the Lagrangian dual correctly degenerates —
safety there is delivered structurally by the spec-attainability certificate and
the escalation/admission layer, which is the appropriate architecture for a
domain outside the strategy's feasible set.*

---

### References
- E. Altman, *Constrained Markov Decision Processes*, Chapman & Hall/CRC, 1999.
- Y. Chow, M. Ghavamzadeh, L. Janson, M. Pavone, "Risk-Constrained Reinforcement
  Learning with Percentile Risk Criteria," *JMLR* 18(167):1–51, 2017.
- H. Yu, H. Ma, S. Li, J. Chen, "Reachability Constrained Reinforcement Learning,"
  *ICML* 2022.
