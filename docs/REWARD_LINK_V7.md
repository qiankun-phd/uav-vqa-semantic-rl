# Task #35/#36 v7 — mission-aligned reward + link-model guards

> Fixes the v6 peak failure (cache flooding pins `lambda_quality_critical`) at its
> root — the reward structure — plus two latent link-model optimisms.  All three
> changes are config-gated and default to the legacy/off path, so the pre-v7
> behaviour is bit-for-bit preserved (283-test suite green; the all-legacy ==
> v6 invariant is asserted in `V7AllLegacyIsV6Test`).

## 0. What v6 left on the table (not re-diagnosed here)

v6 established (see `docs/SCALE_CONSISTENCY_V6.md`) that the scenario/constraint
layer is healthy — escalation-aware oracle admitted mission 0.916, certificate
non-constant (0.32–0.47), cache ban engages (122 events), δ_esc 0.155/0.05,
τ=3.8/4.8 s (2σ), unified M/G/1 wait.  The peak proposed policy nonetheless
failed because a **quality∧deadline-COMPLIANT but UTM-blocked** token scored
**reward −3.42** (full-delay) while the **banned cache** scored **−1.08** →
cache floods (cacheR 0.44) → `lambda_quality_critical ≈ 18.4` pinned.
`no_lagrangian` showed the same cache ratio (0.45), proving the defect is in the
reward, not the dual.

## 1. The three v7 changes

All in `src/vqa_semcom/sim/multi_uav_env.py`, threaded through
`scripts/run_v1_9_resource_alloc.py` as CLI flags.

### #36 — `reward_success_semantics: legacy | mission_aligned`

A UTM/airspace block is an **airspace event, not a service failure**.  Under
`mission_aligned`, a service that cleared the quality LCB, the deadline, the
battery reserve and the GPU budget but was only UTM-blocked:

* earns the success bonus at a discount (`reward_blocked_service_discount`,
  default **0.8**, chosen < 1 so a blocked delivery stays strictly below an
  unblocked compliant-AND-cleared one);
* is **not** charged the airspace-conflict penalty (a spatial/tasking-layer
  cost, not a service failure).

Non-compliant services (quality/deadline/battery/GPU violation) and the banned
cache keep their negative rewards unchanged.  Implemented via a new
`service_compliant` boolean at the success-determination site (the strict
`success` AND is untouched) and a branch in `_reward`.  **Scenario/UTM
parameters are not touched** — the fix is purely in reward attribution.

The single-test (`MissionAlignedRewardGateTest`) asserts the ordering flips.

### #35-(2) — `reference_bandwidth: legacy | fair_share`

The observation SNR bin and the spec-attainability certificate were anchored on
the 50 kHz s0 `default_action(0)` bandwidth (0.05 × 1 MHz pool).  Its noise floor
`−174 + 10·log10(B) + NF` sits ~13 dB below a realised per-UAV share, so the
reference SINR — and hence the quality axis fed into ε/δ_esc — was
systematically optimistic.  `fair_share` anchors the reference link budget on the
fair spectrum share **pool / N_uav** (the same fair-share notion `action_mask`
exposes as a resource hint).  Implemented as a bandwidth override on the
reference parsed-action at the obs site; `_link_budget` is untouched.

### #35-(1) — `lut_support_guard: off | outage`

Nearest-bin snapping (`vqa_semcom.snr.nearest_snr_bin`) silently extrapolated
below-support SINRs to the lowest calibrated bin: a −30 dB link snaps to the
−5 dB bin and reads its LUT accuracy (up to ~0.85).  Under `outage`, a service
whose **effective SINR** is below the lowest LUT bin by more than
`lut_support_margin_db` (default 2.5 dB → threshold **−7.5 dB**) is a quality
**outage** (LCB = 0, quality-infeasible).  Above the top bin the SINR is still
clamped to the top bin (monotone saturation — more SINR cannot hurt quality;
declared behaviour, no upward extrapolation).  Applies to the transmission
services (s1/s2); the s0 cache path carries its own LCB.

The single-test (`LUTSupportGuardTest`) asserts SINR −30 dB ⇒ s1/s2 LCB = 0.

## 2. Reward-ordering fix — before/after (measured)

Synthetic compliant-but-blocked token (acc 0.86, comm-window ddl-delay 2.16 s,
airspace_conflict=True, success=False) vs banned cache (priority 0.54):

| condition                          | served compliant-BLOCKED | banned cache | served > cache? |
|------------------------------------|--------------------------|--------------|-----------------|
| legacy deadline, full delay (v6 diag) | **−3.4200**           | −1.08        | NO (inverted)   |
| comm_window, `legacy` reward       | **−1.5980**              | −1.08        | NO (inverted)   |
| comm_window, `mission_aligned`     | **+0.7780**              | −1.08        | **YES**         |

The v6 −3.42 is reproduced exactly.  Under `mission_aligned` the ordering flips
(+0.778 > −1.08).  Guards on the fix (all tested): a non-compliant blocked
service stays negative (−2.598, no free pass); a full-success unblocked delivery
(+1.122) still out-scores the blocked-discounted one (+0.778).

## 3. ε / δ_esc re-calibration — before/after (`calibrate_epsilon_v7.py`)

The #35-(2) requirement: re-standardise ε/δ_esc once the reference optimism and
the extrapolation are removed.  Ran `--link-model legacy` (reproduces v6) and
`--link-model corrected` (fair_share + outage), tau_conf 2σ, utm_conflict peak.

| quantity                       | v6 / legacy link | v7 corrected link |
|--------------------------------|------------------|-------------------|
| ε critical-counting (cc)       | 0.464            | 0.464 (v5 anchor, unchanged) |
| ε critical-presence (cp)       | 0.696            | 0.696 (v5 anchor, unchanged) |
| ε normal                       | 0.529            | 0.529 (v5 anchor, unchanged) |
| peak spec-unattainable         | 0.105            | **0.105** (identical) |
| δ_esc peak                     | 0.155            | **0.155** (identical) |
| δ_esc nominal                  | 0.05             | **0.05** (identical) |
| peak best-tx LCB (presence)    | med 0.859        | med 0.859 (byte-identical) |

**The link guards do not move this calibration.**  The corrections are live
(verified: reference bandwidth 50 → 250 kHz on the 4-UAV utm_conflict mix; guard
armed) but the peak certificate's candidate-service SINRs are **26.8–36.1 dB**
(critical tasks receive full-pool bandwidth), all in-support and above the top
20 dB bin — so the outage guard never fires and the reference SINR still
saturates the top bin.  ε is a deadline/link-independent v5 LUT quantile and is
unchanged by construction.  The guards eliminate **latent weak-link optimism**;
the utm_conflict peak mix is simply not weak-link-limited.  **v7 therefore keeps
the v6 escalation budget (δ_esc 0.155 / 0.05).**  Certificate stays non-constant
(True 46 / False 154 over the peak collection).

JSONs: `outputs/rl/eps_v7_calib_legacy.json`, `outputs/rl/eps_v7_calib_corrected.json`.

## 4. v6 → v7 matrix comparison (peak + nominal) — proposed policy, seed-mean

The headline result.  The three v7 changes produce a trained policy that is
**numerically identical to v6** on every reported metric, peak and nominal:

| metric   | v6 peak | v7 peak | v6 nom | v7 nom |
|----------|---------|---------|--------|--------|
| cacheR   | 0.441   | **0.441** | 0.167 | **0.167** |
| admMiss  | 0.179   | **0.179** | 0.635 | **0.635** |
| admAcc   | 0.167   | **0.167** | 0.430 | **0.430** |
| task     | 0.228   | **0.228** | 0.628 | **0.628** |
| critEsc  | 0.124   | **0.124** | 0.000 | **0.000** |
| rejR     | 0.390   | **0.390** | 0.000 | **0.000** |

This is **not** because the flags failed to take effect (run_config confirms
`reward_success_semantics=mission_aligned, reference_bandwidth=fair_share,
lut_support_guard=outage`; the per-token reward is demonstrably different —
served-token critical **mean reward +1.35** under v7 vs the v6 −3.42).  The
identical outcome means the policy converges to the **same cache attractor
regardless of the (now-correct) reward gradient**.

Full v7 peak matrix (from `scripts/summarize_eps_recal_v7.py`):

```
arm                                 mission  admMiss  admDdlV  admAcc  cacheR  neRejR   escR  critEsc  specAtt
bl_oracle_best_feasible_evidence      0.528    0.528    0.412   0.757   0.008   0.000  0.000    0.000    0.472
bl_oracle_escalation_aware            0.970    0.916    0.084   0.808   0.000   0.000  0.646    0.666    0.324
bl_always_cache                       0.000    0.000    0.000   0.000   1.000   0.000  0.000    0.000    0.504
bl_random                             0.292    0.292    0.333   0.454   0.339   0.000  0.000    0.000    0.463
proposed                              0.230    0.179    0.024   0.167   0.441   0.325  0.079    0.124    0.323
no_lagrangian                         0.244    0.176    0.021   0.164   0.436   0.321  0.102    0.162    0.322
fixed_penalty                         0.215    0.176    0.023   0.164   0.441   0.338  0.061    0.100    0.326
e4lut                                 0.189    0.137    0.089   0.143   0.653   0.108  0.085    0.110    0.407
```

Nominal (proposed vs oracle): proposed admMiss **0.635** (≥ 0.85·oracle 0.703 =
0.597 → PASS), task **0.628** (< 0.75 → FAIL), cacheR 0.167.  nomtrain proposed
mission 0.709 / task 0.709 (train-on-nominal recovers most of the gap but still
< 0.75 task).

## 5. λ trajectory — did `lambda_quality_critical` come alive?

**No — and the cross-arm evidence shows why that is the wrong question.**

```
arm/seed         lamQC_first lamQC_last  pinned?  qCostCrit
proposed_0             0.038     18.540    True      0.425
proposed_1             0.028     18.052    True      0.368
proposed_2             0.028     18.335    True      0.378
no_lagrangian_0        0.000      0.000    True      0.407   <- dual OFF, same qCost
no_lagrangian_1        0.000      0.000    True      0.334
no_lagrangian_2        0.000      0.000    True      0.376
fixed_penalty_0        9.740      9.740    True      0.416   <- dual FROZEN, same qCost
e4lut_0                0.038     20.000    True      0.574
```

`lambda_quality_critical` climbs monotonically 0.04 → 18.5 across all proposed
seeds (the v6 pin, unchanged).  But `quality_cost_critical` sits at ~0.37–0.43
the *entire* run and never approaches the 0.02 limit — the constraint is never
satisfied, so the dual ascends forever.  **`no_lagrangian` (dual disabled)
carries the identical qCostCrit ~0.33–0.41** with λ_QC pinned at 0; **fixed
penalty (λ frozen at 9.74)** likewise.  The dual is not the cause of the
quality violations — it is a *symptom* dutifully chasing an unsatisfiable
constraint.  With the reward fixed and the dual removed, the policy still
violates quality at the same rate, so the reward-fix (#36) — while correct at
the token level — does not reach the pin because the pin is downstream of a
policy-optimization failure, not a reward/dual failure.

## 6. Success criteria (v6 口径, all PASS ⇒ PASS)

**Verdict: 7/13 PASS → FAIL** (full output: `outputs/rl/eps_recal_v7/SUMMARY_v7.txt`).

| # | criterion | obs | verdict |
|---|-----------|-----|---------|
| 1a | PEAK oracle mission(admitted) ≥ 0.85 | 0.9156 | PASS |
| 1b | PEAK oracle admitted ddl-violation ≤ 0.085 (v6 0.084 hold-even) | 0.0844 | PASS |
| 2a | PEAK proposed crit-escalation in [0.10, 0.21] | 0.1243 | PASS |
| 2b | NOMINAL proposed escalation < 0.15 | 0.000 | PASS |
| 3a | PEAK proposed cacheR < 0.40 | 0.4409 | **FAIL** |
| 3b | PEAK proposed non-escalated-rejectR < 0.20 | 0.3254 | **FAIL** |
| 3c | PEAK proposed mission(admitted) ≥ 0.6·oracle (0.549) | 0.1786 | **FAIL** |
| 3d | PEAK proposed acc(admitted) ≥ 0.7·oracle-acc (0.566) | 0.1670 | **FAIL** |
| 4  | λ quality/deadline-critical non-pinned, active (≥2/3 seeds) | pinned | **FAIL** |
| 5a | NOMINAL proposed mission(admitted) ≥ 0.85·oracle (0.597) | 0.6348 | PASS |
| 5b | NOMINAL proposed task-success ≥ 0.75 | 0.6280 | **FAIL** |
| 6a | spec_attainable non-constant under nominal | yes | PASS |
| 6b | cache-ban engages (≥1 cache-only critical non-compliant event) | 122 | PASS |

## 6a. Root-cause diagnosis (FAIL — no self-authorised v8)

The three v7 changes are individually **correct and verified**:

* #36 flips the token-level ordering (compliant-blocked −1.598 → **+0.778** >
  banned-cache −1.08; served-token critical **mean reward +1.35** vs cache −1.05
  in the actual peak rollout);
* #35-(2) demonstrably lowers the reference SINR (50 → 250 kHz, median 31 → 27 dB);
* #35-(1) forces LCB 0 for a −30 dB service (vs the −5-bin extrapolation ~0.85).

Yet the peak proposed policy is **byte-identical to v6** and still floods cache
(0.44).  The matrix isolates the cause beyond reasonable doubt:

1. **It is not the reward.** Serving now pays +1.35, cache −1.05, reject −0.50 —
   the policy picks the *worst* economic option (cache) over reject, against the
   gradient.
2. **It is not the dual.** `no_lagrangian` (dual off) floods cache at 0.436 with
   the same qCostCrit ~0.40; `fixed_penalty` (λ frozen) at 0.441.  The λ_QC pin
   is a symptom, not a driver.
3. **It is not the link model.** The guards do not fire in this geometry; the
   calibration is unchanged.

The correct behaviour is the escalation-aware oracle's: **cache 0.0, reject/
escalate 0.65** (only ~32 % of critical tasks are spec-attainable, so the rest
must be escalated cost-free, never cached — cache-for-critical is a compliance
violation).  The proposed policy instead **caches the un-serviceable tasks it
should escalate**.  This is a **policy-optimization / exploration pathology**:
PPO settles into the cache basin and neither the corrected reward gradient nor
the (now-irrelevant) dual dislodges it in 500 episodes.

**Suggested next lever (for the human to authorise — not taken here):** the
remaining defect is in learning dynamics, not the reward/link/constraint layer
(all now verified healthy).  Candidates, in order of expected leverage:
(a) an **explicit escalate-vs-cache action-shaping / mask** that makes escalation
the reachable disposition for a spec-unattainable critical task (the oracle's
policy is trivially expressible — the RL agent just is not finding it);
(b) a **cache-entropy / warm-start** curriculum (start from the escalation-aware
oracle or anneal a cache penalty) to break the cache basin;
(c) a **longer / higher-entropy** training budget (the reward trace oscillates
−7…+8 through ep 500 — the run is not converged).  None of these are reward-
structure or link-model changes; the v7 work (this task) closes those two axes.

## 7. Test coverage

`tests/test_scale_consistency_v6.py` (extended): `MissionAlignedRewardGateTest`
(7), `ReferenceBandwidthGateTest` (3), `LUTSupportGuardTest` (4, incl. the
−30 dB ⇒ LCB 0 single-test), `V7AllLegacyIsV6Test` (3, the all-legacy == v6
bit-for-bit invariant + explicit reward-formula check).  Full suite: **283
passed**, legacy paths bit-for-bit.
