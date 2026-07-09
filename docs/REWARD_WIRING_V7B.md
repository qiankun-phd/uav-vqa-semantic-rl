# Reward-wiring fix (v7b): mission-aligned reward into the v19 training path

**Task #36, v7b.** This is a wiring fix for the v7 batch, not a new design. v7
implemented mission-aligned success attribution (#36) correctly in the *env*
reward, but the PPO learner never saw it. v7b routes the same signal into the
reward the learner actually optimises, and re-runs the v7 matrix unchanged
otherwise.

Commit: `exp(rl): v7b -- wire mission-aligned reward into the v19 training path`
(`c7d5993`, on top of v7 `0e94592`).

---

## 1. The bug (evidence, not re-diagnosis)

The v7 matrix ran `--reward-success-semantics mission_aligned` on every learning
arm. The env-layer log moved but the policy did not:

- `proposed` / `no_lagrangian` peak behaviour under v7 is **bit-for-bit identical
  to legacy v6** — same action statistics, same cache ratio:

  | metric (proposed, PEAK) | v6 legacy | v7 mission_aligned |
  |---|---|---|
  | admitted mission | 0.179 | 0.179 |
  | cacheR | 0.441 | 0.441 |
  | non-escalated rejectR | 0.325 | 0.325 |
  | admitted acc | 0.167 | 0.167 |
  | lambda_quality_critical (terminal) | pinned ~18.5 | pinned ~18.5 |

- Yet the training-trace *reward aggregation* columns changed between v6 and v7
  (`raw_return` = col 2, `return` = col 5, `mean_reward` = col 7), e.g. the
  compliant-blocked rescue is worth +60.8 reward-units per token.
- `ppo_lambda_trace.csv` is bit-for-bit identical v6↔v7.

Conclusion (given): the env `_reward()` mission_aligned branch is implemented and
correct, but it **only feeds the log**. The learner's gradient comes from a
different reward path that does not consume `reward_success_semantics` (the flag
appears only in `multi_uav_env.py` across the whole tree).

v7 verdict for reference: **7/13 criteria PASS**; the 6 FAILs are 3a cacheR<0.40,
3b non-escalated rejectR<0.20, 3c mission(admitted)>=0.6·oracle, 3d acc>=0.7·
oracle, 4 lambda_QC non-pinned, 5b nominal task>=0.75.

---

## 2. The fork point (file:line)

The learner does **not** optimise `env.step()`'s returned reward directly. Two
files diverge from `multi_uav_env._reward()`:

1. **`scripts/run_v1_9_resource_alloc.py:660-661`** — the default env reward mode
   is silently rewritten:
   ```python
   if args.semantic_reward_mode == "env":
       args.semantic_reward_mode = "semantic_utility"
   ```
   Every proposed/no_lagrangian/fixed_penalty/e4lut arm trains under
   `semantic_reward_mode = "semantic_utility"` (confirmed in each
   `run_config.json`).

2. **`src/vqa_semcom/rl/v19_ppo.py`** — the rollout collectors
   (`_collect_ppo_rollout` :957, `_collect_two_timescale_rollout` :1126) call
   ```python
   next_obs, raw_reward, done, info = env.step(action)          # carries #36
   semantic_reward = _semantic_controller_reward(obs, info, raw_reward, cfg)
   shaped_reward = semantic_reward - dual.penalty(info) if cfg.constrained else semantic_reward
   ```
   The **learner optimises `shaped_reward`** (→ `rollout["rewards"]` →
   `_discounted_returns` → advantages). But `_semantic_controller_reward`
   (:2777) returns `raw_reward` **only** when `mode == "env"` (:2778); under
   `semantic_utility` it self-composes a utility from `info` fields and
   **discards `raw_reward`**. That utility never reads `reward_success_semantics`.

So `mission_aligned` reached `raw_reward` (logged as `raw_return`/`return`/
`mean_reward`) but not `shaped_reward` (`shaped_return`/`mean_shaped_reward`),
which is what the gradient uses. Behaviour therefore matched legacy exactly.

### Why option (b): replicate in the controller

The env reward and the controller reward are **not isomorphic** — the controller
reward is a bespoke composition (12 success weight, 40 UTM-strong penalty,
per-path shaping, Lyapunov queues) with its own success/conflict terms, plus a
separate dual conflict penalty applied *outside* the function. Threading the env
reward through does nothing under `semantic_utility` (it is discarded). So the
fix replicates the mission_aligned rule in the controller path, kept in a single
helper, gated on the same flag, with a two-layer-consistency test.

---

## 3. The fix

All in `src/vqa_semcom/rl/v19_ppo.py`, gated on
`cfg.reward_success_semantics == "mission_aligned"` (default `"legacy"`).

- **`_mission_aligned_success_credit(info, cfg)`** — single truth source. Returns
  the success credit in [0,1], mirroring `_reward.success_bonus_scale`:
  legacy → `float(info["success"])`; mission_aligned + COMPLIANT (cleared
  quality/deadline/battery/resource — the same axes the wrapper folds into
  `info["success"]`, or the env's `service_compliant` flag when present) +
  BLOCKED (airspace_conflict or a UTM violation) + not already success →
  `reward_blocked_service_discount` (0.8). Non-compliant or unblocked → strict.

- **`_mission_aligned_reward_info(info, cfg)`** — returns `(info_view, credit)`.
  On a rescue the view **clears the airspace/UTM block flags** so both the
  shaped `safety_cost` terms (inside the controller) and the dual conflict
  penalty (outside it) drop the block charge — a UTM block is an airspace event,
  not a service failure. Legacy returns `info` unchanged.

- **`_semantic_controller_reward`** rebinds `info` to the mission view right after
  the `env` passthrough, and credits `task_success` at the discount (the SEMANTIC
  success axis keeps its true value — quality genuinely passed). This drops the
  40-unit UTM-strong penalty and pays the success/completion bonuses at 0.8.

- **the shaped-reward dual penalty** at both collectors feeds
  `dual.penalty(_mission_aligned_reward_info(info, cfg)[0])`, so the conflict
  lambda term (the only block-reading term in `DualState.penalty`) drops exactly
  when the env `_reward` drops `conflict_charged`.

- **`PPOTrainConfig`** gains `reward_success_semantics: str = "legacy"` and
  `reward_blocked_service_discount: float = 0.8`, threaded from the existing CLI
  flags in `run_v1_9_resource_alloc.py` (`None → "legacy"`). **No new flag** — the
  same `--reward-success-semantics mission_aligned` now feeds both layers.

---

## 4. Wiring proof (ran before the matrix)

### 4a. Unit-level: controller reward flips on the compliant-blocked token

A quality+deadline COMPLIANT, UTM-BLOCKED token (acc 0.86, s1), controller reward
under `semantic_utility`:

| | legacy | mission_aligned | Δ |
|---|---|---|---|
| controller reward | **−10.86** | **+49.94** | **+60.80** |
| credit | 0.0 | 0.8 | |
| view airspace_conflict | True | False (cleared) | |

- non-compliant blocked: credit 0.0 (not rescued).
- unblocked compliant success: legacy = mission = **53.14** (clean deliveries
  unchanged).

### 4b. Legacy is bit-for-bit unchanged

`_semantic_controller_reward` compared old (pre-patch backup) vs new module over
**8000 randomised infos** (4 semantic modes × constrained on/off): **0 differ**
(|Δ|<1e-12). The mission_aligned path is fully gated.

### 4c. Smoke: the LEARNER's reward diverges and behaviour begins to fork

Two training runs, same seed 0, `semantic_utility`, full v7 gate set, one legacy
one mission_aligned:

| | 20 episodes | 80 episodes |
|---|---|---|
| `mean_shaped_reward` mean(legacy → mission) | −27.16 → −26.86 | −55.94 → −55.82 |
| episodes where shaped reward differs | 3 / 20 | **15 / 80** |
| max per-episode `shaped_return` diff | **60.80** | **72.35** |
| first shaped-reward divergence | **ep 16** (Δ +6.08) | ep 65 |
| first `path_cache_ratio` action fork | — (too few updates) | **ep 67** |
| `lambda_quality_critical` episodes differ | 0 | **13 / 80** |

The learner's shaped reward now diverges from the legacy run (it did not under
the v7 code — `shaped_return` would have been byte-identical because the
controller discarded `raw_reward`). The action distribution (`path_cache_ratio`)
and the quality-critical dual begin to fork once enough compliant-blocked events
accumulate gradient (ep 65–67); this amplifies over the 500-episode matrix.

### 4d. Two-layer semantic consistency (unit test)

`tests/test_reward_wiring_v7b.py::TwoLayerConsistencyTest` asserts that switching
legacy→mission_aligned RAISES the reward of the same compliant-blocked event at
BOTH the env layer (`_reward`) and the controller layer, and FLIPS the
compliant-blocked vs cheap-shortcut ordering at both layers.

### 4e. Tests

283 baseline + 17 new (`test_reward_wiring_v7b.py`) = **300 pass**. New tests:
controller ordering (compliant-blocked beats the attractive s0 cache under
mission_aligned, loses under legacy), the monotone blocked-vs-cache gap, the dual
conflict-penalty drop on rescue, non-compliant/unblocked invariance, `env`-mode
passthrough, two-layer consistency, and a 1500-info legacy bit-for-bit regression.

---

## 5. v6 / v7 / v7b comparison

(PEAK = utm_conflict train; NOMINAL = nominal eval of the peak model. Oracle
ceiling = escalation-aware oracle. Metrics: admitted mission / task / admitted
acc / cacheR / non-escalated rejectR / critical-escalation.)

### PEAK (utm_conflict)

| arm | ver | admMiss | task | admAcc | cacheR | neRejR | critEsc |
|---|---|---|---|---|---|---|---|
| proposed | v6 | 0.179 | 0.228 | 0.167 | 0.441 | 0.325 | 0.124 |
| proposed | v7 | 0.179 | 0.228 | 0.167 | 0.441 | 0.325 | 0.124 |
| proposed | **v7b** | **0.179** | **0.229** | **0.167** | **0.440** | **0.326** | **0.126** |
| no_lagrangian | v6 | 0.176 | 0.242 | 0.164 | 0.436 | 0.321 | 0.162 |
| no_lagrangian | v7 | 0.176 | 0.242 | 0.164 | 0.436 | 0.321 | 0.162 |
| no_lagrangian | **v7b** | **0.164** | **0.384** | **0.213** | **0.298** | **0.287** | **0.329** |
| fixed_penalty | v6 | 0.176 | 0.214 | 0.164 | 0.441 | 0.338 | 0.100 |
| fixed_penalty | v7 | 0.176 | 0.214 | 0.164 | 0.441 | 0.338 | 0.100 |
| fixed_penalty | **v7b** | **0.173** | **0.206** | **0.164** | **0.422** | **0.366** | **0.087** |
| e4lut | v6 | 0.021 | 0.503 | 0.018 | 0.488 | 0.011 | 0.511 |
| e4lut | v7 | 0.137 | 0.189 | 0.143 | 0.653 | 0.108 | 0.110 |
| e4lut | **v7b** | **0.132** | **0.158** | **0.144** | **0.661** | **0.119** | **0.057** |
| oracle_esc_aware | v6/v7/v7b | 0.916 | 0.789 | 0.808 | 0.000 | 0.000 | 0.666 |

### NOMINAL (nominal eval of the peak model)

| arm | ver | admMiss | task | admAcc | cacheR | neRejR | critEsc |
|---|---|---|---|---|---|---|---|
| proposed | v6/v7 | 0.635 | 0.628 | 0.430 | 0.167 | 0.000 | 0.000 |
| proposed | **v7b** | **0.592** | **0.587** | **0.446** | **0.145** | **0.000** | **0.000** |
| no_lagrangian | v6/v7 | 0.693 | 0.689 | 0.401 | 0.232 | 0.000 | 0.000 |
| no_lagrangian | **v7b** | **0.593** | **0.593** | **0.438** | **0.151** | **0.000** | **0.000** |
| fixed_penalty | v6/v7 | 0.642 | 0.641 | 0.404 | 0.209 | 0.000 | 0.000 |
| fixed_penalty | **v7b** | **0.650** | **0.644** | **0.424** | **0.140** | **0.000** | **0.000** |
| oracle_esc_aware | v6/v7/v7b | 0.703 | 0.703 | 0.454 | 0.000 | 0.033 | 0.000 |

**Reading the table.**
- **proposed PEAK is unchanged v6→v7→v7b** (admMiss 0.179, cacheR 0.44). The
  wired signal has no eval-time effect on the constrained arm (see §7).
- **no_lagrangian PEAK moved materially in v7b**: cacheR 0.436→**0.298**, task
  0.242→**0.384**, admAcc 0.164→**0.213**. The unconstrained arm (no dual to
  fight the reward) *does* respond to the wired signal — this is direct evidence
  that the fix reaches the learner. The `proposed` arm's pinned λ_QC dual (§6)
  masks the same reward change.
- **all learning arms shift lower on cache under NOMINAL** (proposed 0.167→0.145,
  no_lag 0.232→0.151, fixed 0.209→0.140) — the wired mission_aligned reward
  consistently nudges the policy off cache, just not enough to pass the peak bar.
- **oracle / baselines are identical across versions** (they do not train and are
  reward-insensitive), as expected — re-evaluated fresh under v7b (not copied).

---

## 6. Criteria verdict

`scripts/summarize_eps_recal_v7b.py` (13 v6/v7 criteria):

| # | criterion | v7 | v7b | obs (v7b) |
|---|---|---|---|---|
| 1a | PEAK oracle mission(adm) ≥ 0.85 | PASS | PASS | 0.916 |
| 1b | PEAK oracle adm ddl-viol ≤ 0.085 | PASS | PASS | 0.084 |
| 2a | PEAK proposed crit-esc ∈ [0.10,0.21] | PASS | PASS | 0.126 |
| 2b | NOMINAL proposed escalation < 0.15 | PASS | PASS | 0.000 |
| 3a | PEAK proposed cacheR < 0.40 | FAIL | **FAIL** | 0.440 |
| 3b | PEAK proposed non-esc rejectR < 0.20 | FAIL | **FAIL** | 0.326 |
| 3c | PEAK proposed mission(adm) ≥ 0.6·oracle | FAIL | **FAIL** | 0.179 (need 0.549) |
| 3d | PEAK proposed acc(adm) ≥ 0.7·oracle | FAIL | **FAIL** | 0.167 (need 0.566) |
| 4 | λ_QC/deadline non-pinned ≥2/3 seeds | FAIL | **FAIL** | 0/3 (pinned ~18.5) |
| 5a | NOMINAL proposed mission ≥ 0.85·nom-oracle | PASS | **FAIL** | 0.592 (need 0.597) |
| 5b | NOMINAL proposed task ≥ 0.75 | FAIL | **FAIL** | 0.587 |
| 6a | spec_attainable non-constant (nominal) | PASS | PASS | yes |
| 6b | cache-ban engages (peak) | PASS | PASS | 122 events |

**v7b: 6/13 PASS → FAIL.** The 6 criteria that hung in v7 (3a, 3b, 3c, 3d, 4,
5b) all remain FAIL; 5a additionally regressed from PASS to FAIL (nominal
proposed admMiss 0.635→0.592, 5 thousandths below the 0.597 gate — the wired
reward shifted the nominal policy just under the ceiling-relative bar). No
non-training criterion (oracle / budget / certificate / ban / nominal-oracle)
regressed.

---

## 7. Diagnosis (FAIL — root cause, no self-continuation)

The wiring bug was **real and is fixed** — but fixing it does **not** resolve the
6 hung criteria, because mission_aligned's target event does not occur at
convergence.

**The rescue target is absent in the converged rollout.** In the `proposed` PEAK
eval rollout, blocked events = **0 / 343** and compliant-blocked rescue targets =
**0 / 343** — byte-identical to v7. The trained policy already routes around every
UTM/airspace block (0 airspace conflicts at eval), so the +60.8 compliant-blocked
success credit that mission_aligned pays is **never applied at eval**. The
premise of #36 (a compliant delivery under-rewarded because it is UTM-blocked)
does not bind for the converged policy.

**The signal does reach the learner — proven three ways.** (1) The smoke shaped
reward diverges (§4c: max `shaped_return` diff 72.35). (2) The unconstrained
`no_lagrangian` PEAK arm moved (cacheR 0.436→0.298, task 0.242→0.384) — a policy
change that can only come from the reward. (3) Every arm shifted lower on cache
under NOMINAL. So this is not a second wiring bug; the fix works, the event is
just rare/absent at convergence.

**Why `proposed` PEAK is unchanged while `no_lagrangian` moved.** `proposed` runs
the constrained dual; λ_quality_critical **pins at ~18.5** (all 3 seeds, terminal
window), with quality_cost_critical ~0.40 — 20× over the 0.02 limit and never
satisfied. That pinned dual dominates the shaped reward
(`shaped = semantic - λ·cost`), so a change in the positive success term (which
fires only on the rare, exploration-time compliant-blocked token) is swamped. The
unconstrained arm has no such dual, so the same reward change is visible.

**Therefore the cache-flood / λ_QC-pin is a *different* failure from #36.** It is
driven by the critical-cache quality-cost channel staying unsatisfiable under the
peak all-critical UTM mix (cache is quality-clobbered → quality_cost stays high →
λ_QC ramps to the ceiling), independent of the compliant-blocked reward ordering.
Candidate next steps (NOT run here, per the FAIL protocol):
- attack the λ_QC pin directly (the quality-cost channel calibration / limit, or
  the cache-ban's interaction with the quality cost), not the success attribution;
- or reframe the peak target so a quality-clobbered cache cannot be the local
  optimum the policy falls into (admission/reject shaping), since even the
  escalation-aware oracle only reaches mission 0.916 by escalating 0.67 of the
  critical mix.

The v7b wiring change is correct and should stay (it removes a genuine
train/log inconsistency and makes the unconstrained arm honest), but it is not
the lever for the 6 hung criteria.

---

## 8. Artefacts

- fix: `src/vqa_semcom/rl/v19_ppo.py` (`_mission_aligned_success_credit`,
  `_mission_aligned_reward_info`, `_semantic_controller_reward` rebind, both
  shaped-reward dual sites), `scripts/run_v1_9_resource_alloc.py` (config thread).
- tests: `tests/test_reward_wiring_v7b.py` (17; 283+17=300 green).
- matrix: `scripts/eps_recal_v7b_runs.sh` → `outputs/rl/eps_recal_v7b/`;
  `scripts/summarize_eps_recal_v7b.py` → `outputs/rl/eps_recal_v7b_summary.txt`.
- wiring smoke: `outputs/rl/_smoke_wiring_v7b/`, `outputs/rl/_smoke_wiring_v7b80/`.
