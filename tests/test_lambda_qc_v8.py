#!/usr/bin/env python
"""Task #37 v8: per-channel quality lambda cap + dual warm-up.

Two levers unpin the quality-critical dual under the peak all-critical overload:

  * lever 1 (lambda_max_quality): the quality_normal/quality_critical channels
    get a dedicated dual ceiling (default 20 = legacy; v8 uses 8), mirroring the
    conflict-channel precedent (lambda_max_conflict).  Caps the dual penalty at
    "a conflict step is net-negative" instead of "the whole return is swamped".
  * lever 2 (dual_warmup_episodes): freeze ALL dual variables for the first N
    training episodes (costs still logged by the caller), then resume normal
    projected sub-gradient ascent, so the policy forms under the fixed-init
    reward during the BC/service-prior shaping window.

Both default to the legacy value (20 / 0), so the pre-v8 path is bit-for-bit
preserved -- the legacy-regression tests below assert exactly that.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_semcom.rl.v19_ppo import (  # noqa: E402
    DualState,
    PPOTrainConfig,
    _dual_update,
    _init_dual_state,
    _update_duals,
)


def _risk_aware_cfg(**overrides: object) -> PPOTrainConfig:
    base = dict(
        two_timescale=True,
        semantic_reward_mode="semantic_utility",
        risk_aware_constraints=True,
        constrained=True,
        device="cpu",
    )
    base.update(overrides)
    return PPOTrainConfig(**base)


def _rollout(qc_cost: float = 0.4, qn_cost: float = 0.4) -> dict[str, list]:
    """A rollout whose quality-critical/normal costs sit far above their limits
    (the peak all-critical regime), everything else zero/absent."""
    return {
        "quality_costs": [qc_cost],
        "deadline_costs": [0.0],
        "quality_costs_normal": [qn_cost],
        "quality_costs_critical": [qc_cost],
        "deadline_costs_normal": [0.0],
        "deadline_costs_critical": [0.0],
        "conflict_costs": [0.0],
        "battery_costs": [0.0],
        "gpu_costs": [0.0],
        "escalation_costs": [0.0],
    }


class LambdaQCV8Test(unittest.TestCase):
    # --- lever 1: per-channel quality ceiling -----------------------------

    def test_default_quality_cap_is_legacy_global(self) -> None:
        """Default lambda_max_quality equals the global lambda_max (20)."""
        cfg = PPOTrainConfig()
        self.assertEqual(cfg.lambda_max_quality, 20.0)
        self.assertEqual(cfg.lambda_max_quality, cfg.lambda_max)

    def test_quality_dual_saturates_at_channel_cap(self) -> None:
        """Under persistent over-limit cost the quality_critical dual saturates
        at lambda_max_quality (8), NOT the global 20."""
        cfg = _risk_aware_cfg(lambda_max_quality=8.0)
        dual = DualState()
        for ep in range(1000):
            _update_duals(dual, _rollout(qc_cost=0.4, qn_cost=0.4), cfg, episode=ep)
        self.assertLessEqual(dual.quality_critical, 8.0 + 1e-9)
        self.assertLessEqual(dual.quality_normal, 8.0 + 1e-9)
        # It genuinely reaches the (lowered) ceiling -- the constraint is
        # unsatisfiable at this cost, so the cap binds.
        self.assertGreater(dual.quality_critical, 0.9 * 8.0)

    def test_lowered_cap_binds_below_global(self) -> None:
        """The 8-cap holds the quality dual strictly below where the legacy
        20-cap would let it climb, at the same over-limit pressure."""
        capped = DualState()
        legacy = DualState()
        cfg_capped = _risk_aware_cfg(lambda_max_quality=8.0)
        cfg_legacy = _risk_aware_cfg(lambda_max_quality=20.0)
        for ep in range(1000):
            _update_duals(capped, _rollout(), cfg_capped, episode=ep)
            _update_duals(legacy, _rollout(), cfg_legacy, episode=ep)
        self.assertLessEqual(capped.quality_critical, 8.0 + 1e-9)
        self.assertGreater(legacy.quality_critical, 8.0)
        self.assertLess(capped.quality_critical, legacy.quality_critical)

    def test_quality_cap_applied_in_non_risk_aware_path(self) -> None:
        """The shared-quality (non-risk-aware) branch also honours the cap."""
        cfg = _risk_aware_cfg(lambda_max_quality=8.0, risk_aware_constraints=False)
        dual = DualState()
        for ep in range(1000):
            _update_duals(dual, _rollout(), cfg, episode=ep)
        self.assertLessEqual(dual.quality_critical, 8.0 + 1e-9)
        self.assertLessEqual(dual.quality_normal, 8.0 + 1e-9)

    def test_init_dual_state_clamps_quality_to_channel_cap(self) -> None:
        """A warm-started quality init above the cap is clamped to it (fixed-
        penalty arm sets quality_critical=9.74 > 8)."""
        cfg = PPOTrainConfig(lambda_max_quality=8.0, lambda_init_quality_critical=9.74,
                             lambda_init_quality_normal=12.0)
        dual = _init_dual_state(cfg)
        self.assertEqual(dual.quality_critical, 8.0)
        self.assertEqual(dual.quality_normal, 8.0)
        # A below-cap init is untouched.
        cfg2 = PPOTrainConfig(lambda_max_quality=8.0, lambda_init_quality_critical=3.0)
        self.assertEqual(_init_dual_state(cfg2).quality_critical, 3.0)

    # --- lever 2: dual warm-up --------------------------------------------

    def test_default_warmup_is_zero(self) -> None:
        self.assertEqual(PPOTrainConfig().dual_warmup_episodes, 0)

    def test_warmup_freezes_then_resumes(self) -> None:
        """During warm-up the quality dual stays at its init; the first episode
        at/after the boundary moves it."""
        cfg = _risk_aware_cfg(dual_warmup_episodes=150, lambda_init_quality_critical=2.0)
        dual = _init_dual_state(cfg)
        start = dual.quality_critical
        self.assertEqual(start, 2.0)
        # Inside the window: frozen for every episode 0..149.
        for ep in range(150):
            _update_duals(dual, _rollout(), cfg, episode=ep)
            self.assertEqual(dual.quality_critical, start)
        # First post-window episode: the over-limit cost drives lambda up.
        _update_duals(dual, _rollout(), cfg, episode=150)
        self.assertGreater(dual.quality_critical, start)

    def test_warmup_none_episode_bypasses_gate(self) -> None:
        """episode=None (non-training callers) always updates, even with a
        non-zero warm-up configured -- preserves the legacy call contract."""
        cfg = _risk_aware_cfg(dual_warmup_episodes=150)
        dual = DualState()
        _update_duals(dual, _rollout(), cfg)  # no episode kwarg
        self.assertGreater(dual.quality_critical, 0.0)

    def test_warmup_logs_costs_but_not_lambda(self) -> None:
        """Warm-up pauses lambda updates only; the caller still logs rollout
        costs (this test asserts the freeze does not depend on cost being 0)."""
        cfg = _risk_aware_cfg(dual_warmup_episodes=10)
        dual = DualState()
        for ep in range(10):
            _update_duals(dual, _rollout(qc_cost=5.0), cfg, episode=ep)
        self.assertEqual(dual.quality_critical, 0.0)  # still frozen despite huge cost

    # --- legacy regression: defaults reproduce pre-v8 behaviour bit-for-bit ---

    def test_legacy_defaults_bit_identical(self) -> None:
        """With lambda_max_quality=20 (default) and dual_warmup_episodes=0
        (default), _update_duals is bit-for-bit identical to the pre-v8 path:
        quality channels clamp to the GLOBAL lambda_max and never freeze."""
        cfg = _risk_aware_cfg()  # defaults: cap=20, warmup=0
        self.assertEqual(cfg.lambda_max_quality, 20.0)
        self.assertEqual(cfg.dual_warmup_episodes, 0)

        # Reference: manual pre-v8 update (quality clamped to cfg.lambda_max,
        # no per-channel cap, no warm-up), replicated step by step.
        ref = DualState()
        got = DualState()
        for ep in range(300):
            r = _rollout(qc_cost=0.4, qn_cost=0.3)
            # pre-v8 quality update: _dual_update WITHOUT a lambda_max override
            # (falls back to cfg.lambda_max) and lambda_decay=0.
            ref.quality_normal = _dual_update(ref.quality_normal, 0.3, cfg.quality_cost_limit_normal, cfg, lambda_decay=0.0)
            ref.quality_critical = _dual_update(ref.quality_critical, 0.4, cfg.quality_cost_limit_critical, cfg, lambda_decay=0.0)
            _update_duals(got, r, cfg, episode=ep)
        self.assertAlmostEqual(got.quality_critical, ref.quality_critical, places=9)
        self.assertAlmostEqual(got.quality_normal, ref.quality_normal, places=9)

    def test_legacy_init_unchanged_when_cap_is_default(self) -> None:
        """With the default cap, a quality init below 20 is preserved exactly as
        the pre-v8 _init_dual_state did (clamp to lambda_max=20)."""
        cfg = PPOTrainConfig(lambda_init_quality_critical=9.74, lambda_init_quality_normal=5.0)
        dual = _init_dual_state(cfg)
        self.assertEqual(dual.quality_critical, 9.74)
        self.assertEqual(dual.quality_normal, 5.0)

    def test_warmup_zero_equals_no_gate(self) -> None:
        """dual_warmup_episodes=0 must not gate episode 0 (boundary: <0 is empty)."""
        cfg = _risk_aware_cfg(dual_warmup_episodes=0)
        dual = DualState()
        _update_duals(dual, _rollout(), cfg, episode=0)
        self.assertGreater(dual.quality_critical, 0.0)


if __name__ == "__main__":
    unittest.main()
