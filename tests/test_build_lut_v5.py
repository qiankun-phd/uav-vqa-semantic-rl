"""Task #28 v5 unit tests for the unified-LUT rebuild primitives.

Guards the change-1 (+-20% critical / +-10% normal counting re-judge) and
change-2 (count bucketing, Wilson interval, sparse-cell parent inheritance)
logic in scripts/build_lut_v5.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_lut_v5", ROOT / "scripts" / "build_lut_v5.py")
blv5 = importlib.util.module_from_spec(_spec)
sys.modules["build_lut_v5"] = blv5
assert _spec.loader is not None
_spec.loader.exec_module(blv5)


def test_count_bucket_boundaries():
    assert blv5.count_bucket(1) == "1-4"
    assert blv5.count_bucket(4) == "1-4"
    assert blv5.count_bucket(5) == "5-9"
    assert blv5.count_bucket(9) == "5-9"
    assert blv5.count_bucket(10) == "10-19"
    assert blv5.count_bucket(19) == "10-19"
    assert blv5.count_bucket(20) == "20-49"
    assert blv5.count_bucket(49) == "20-49"
    assert blv5.count_bucket(50) == "50+"
    assert blv5.count_bucket(500) == "50+"
    assert blv5.count_bucket(0) == "0"


def test_rejudge_tolerance_ge10_is_20pct():
    # GT=10 -> tol = max(1, round(0.20*10)) = 2 ; pred 12 correct, 13 wrong
    assert blv5.rejudge_counting("12", 10) is True
    assert blv5.rejudge_counting("13", 10) is False
    # GT=50 -> tol = round(0.20*50)=10 ; pred 60 correct, 61 wrong
    assert blv5.rejudge_counting("60", 50) is True
    assert blv5.rejudge_counting("61", 50) is False


def test_rejudge_tolerance_lt10_stays_10pct():
    # GT=8 -> tol = max(1, round(0.10*8)) = max(1,1)=1 ; pred 9 correct, 10 wrong
    assert blv5.rejudge_counting("9", 8) is True
    assert blv5.rejudge_counting("10", 8) is False
    # GT=4 -> tol = max(1, round(0.4)) = 1
    assert blv5.rejudge_counting("5", 4) is True
    assert blv5.rejudge_counting("6", 4) is False


def test_rejudge_ge10_widening_flips_boundary_case():
    # GT=10, pred=12: legacy +-10% (tol=1) -> WRONG ; v5 +-20% (tol=2) -> CORRECT
    assert blv5.rejudge_counting("12", 10) is True  # v5
    legacy_tol = max(1, round(0.10 * 10))
    assert abs(12 - 10) > legacy_tol  # would have been wrong under legacy


def test_rejudge_unparseable_returns_none():
    assert blv5.rejudge_counting("unknown", 10) is None
    assert blv5.rejudge_counting("", 10) is None


def test_wilson_interval_bounds_and_order():
    p, lo, hi = blv5.wilson(50, 100)
    assert 0.0 <= lo <= p <= hi <= 1.0
    # degenerate
    assert blv5.wilson(0, 0) == (0.0, 0.0, 0.0)
    # tighter interval with more samples
    _, lo1, hi1 = blv5.wilson(5, 10)
    _, lo2, hi2 = blv5.wilson(500, 1000)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_int_or_none():
    assert blv5._int_or_none("12") == 12
    assert blv5._int_or_none("12.0") == 12
    assert blv5._int_or_none("unknown") is None
    assert blv5._int_or_none("") is None
