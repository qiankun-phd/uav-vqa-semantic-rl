"""BUBBLES D2.1-conformant separation-minima chain and CPA tactical-conflict test.

This module is a self-contained (numpy/torch-free) implementation of the
separation-management geometry defined by the SESAR JU BUBBLES project,
*D2.1 Concept Formulation*, Ed. 04.00.00 (2022-09-27, Grant 893206). It is
consumed by :mod:`vqa_semcom.sim.multi_uav_env` only when the resource-allocation
environment is run with ``multi_uav_env.scenario_profile == "bubbles"``; with the
profile disabled the module is never touched, so default env behaviour is
unchanged.

Primary references (page numbers refer to the D2.1 PDF):

* **Appendix B, Table B-2 (p.99)** -- per-class performance envelope
  (cruise speed, rate of climb/descent, horizontal/vertical size).
* **Sec. 3.3.4 "Block 4: Separation minima" (p.60-61)** -- the chained
  separation-minima equations ``d_NMAC -> d_IC -> d_SL -> d_TC`` (and the
  vertical counterparts), with ``T4`` modelled as the sum of three normal
  delay components (tracking + separator + pilot).
* **Sec. 3.3 / p.38 & p.61-62** -- the *two-condition* tactical-conflict (TC)
  declaration: a conflict exists only when the predicted CPA separation is
  below the separation minimum **and** the time remaining to the CPA is below a
  pre-established threshold ``TC_th``.
* **Appendix G, Tables G-1/G-2/G-3/G-4 (p.123-125)** -- worked example. The
  default parameters below reproduce the Table G-4 horizontal minima; e.g.
  SAIL I-II horizontal ``d_TC`` ~= 370.53 m (see :mod:`tests.test_bubbles_separation`).
* **Appendix D (p.112)** -- Target Level of Safety ``TLS_MAC = 2.5e-7`` fatal
  mid-air collisions per flight-hour (25% of the overall 1e-6 FAT/FH TLS).

Units: metres, seconds, m/s throughout. Imperial NMAC distances are converted
with the exact factor ``ft -> m = 0.3048``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Target Level of Safety anchor (D2.1 Appendix D, p.110-112).                  #
# --------------------------------------------------------------------------- #
# Overall U-space TLS (p.110): 1e-6 fatalities / flight-hour.
TLS_OVERALL_FAT_PER_FH: float = 1.0e-6
# Mid-air-collision-specific TLS (p.112): 25 % of the overall risk is
# apportioned to MAC due to human/operational issues -> 2.5e-7 FAT/FH.
TLS_MAC_FAT_PER_FH: float = 2.5e-7

FT_TO_M: float = 0.3048

# NMAC half-distances for small UAS (D2.1 p.60): each aircraft contributes half
# of the pairwise separation-minima sum, hence 25 ft / 7.5 ft rather than the
# 250 ft / 50 ft used for larger/manned aircraft.
D_NMAC_FT: float = 25.0
H_NMAC_FT: float = 7.5


@dataclass(frozen=True)
class TrafficPerformance:
    """Per-class kinematic envelope from D2.1 Appendix B, Table B-2 (p.99)."""

    cruise_mps: float
    roc_mps: float      # rate of climb
    rod_mps: float      # rate of descent
    size_h_m: float     # horizontal characteristic size (e.g. wingspan)
    size_v_m: float     # vertical characteristic size (e.g. height)


# D2.1 Table B-2 (p.99). Keys use the traffic-class labels from Table B-1.
TABLE_B2: dict[str, TrafficPerformance] = {
    "A1": TrafficPerformance(5.0, 4.0, 3.0, 0.5, 0.25),
    "A2": TrafficPerformance(5.0, 4.0, 3.0, 1.0, 0.5),
    "A3": TrafficPerformance(10.0, 4.0, 3.0, 2.0, 1.0),
    "SAIL_I_II": TrafficPerformance(12.0, 4.0, 3.0, 1.0, 0.5),
    "SAIL_III_IV": TrafficPerformance(14.0, 5.0, 4.0, 2.0, 1.0),
    "SAIL_V_VI": TrafficPerformance(15.0, 5.0, 4.0, 2.0, 1.0),
    "NO_PASSENGER": TrafficPerformance(25.0, 5.0, 4.0, 4.0, 2.0),
    "PASSENGER": TrafficPerformance(25.0, 3.0, 2.0, 5.0, 2.5),
}


@dataclass(frozen=True)
class SeparationParams:
    """Error/delay parameters for the separation-minima chain.

    Defaults are the D2.1 Appendix G worked-example values:
      * Table G-2 (p.124): the three T4 delay components (mean/std, seconds).
      * Table G-3 (p.124): navigation/technical/surveillance error budgets.
      * T2 / T_res / vertical reference speed are the constant time windows that
        reproduce the Table G-4 vertical column (h_IC/h_SL/h_TC) exactly.

    The ``(95%)`` total-system-error used by the chain is treated as:
      * horizontal: the 95th-percentile radius of a 2-D (Rayleigh) error, i.e.
        ``sigma * sqrt(-2 ln 0.05) ~= 2.4477 * sigma``;
      * vertical: the ``2-sigma`` bound (D2.1 convention "95% ~= 2 sigma").
    """

    # Table G-3 (p.124) -- 1-sigma total-system-error (TSE) in metres.
    tse_h_sigma_m: float = 2.12
    tse_v_sigma_m: float = 1.82
    # Table G-3 -- surveillance system error (SSE) at 2-sigma, in metres.
    sse_h_2sigma_m: float = 4.0
    sse_v_2sigma_m: float = 2.0
    # Constant time windows (seconds) calibrated to the Table G-4 example.
    t2_s: float = 3.0        # DAA-CA detect/communicate/execute window (IC step)
    t_res_s: float = 4.236   # residual-risk window (SL step)
    vertical_ref_speed_mps: float = 1.0  # V_v, leveled-cruise vertical speed
    # Table G-2 (p.124) -- T4 = sum of three independent normals (seconds).
    t4_delay_means_s: tuple[float, ...] = (2.4, 1.8, 3.62)   # tracking, separator, pilot
    t4_delay_stds_s: tuple[float, ...] = (2.2, 1.0, 3.48)
    # Confidence expressed in standard deviations for the T4 window.
    # D2.1 p.62: 2 sigma -> 0.977, 1 sigma -> 0.841, 0 sigma -> 0.5 confidence.
    t4_confidence_sigma: float = 2.0

    def tse_h_95_m(self) -> float:
        """95th-percentile horizontal (2-D / Rayleigh) total system error."""
        return self.tse_h_sigma_m * math.sqrt(-2.0 * math.log(0.05))

    def tse_v_95_m(self) -> float:
        """95% (~2-sigma) vertical total system error."""
        return 2.0 * self.tse_v_sigma_m


# Confidence -> number-of-sigmas lookup for the T4 window (D2.1 p.62).
T4_CONFIDENCE_SIGMA: dict[str, float] = {
    "0sigma": 0.0,   # 0.500 confidence
    "1sigma": 1.0,   # 0.841 confidence
    "2sigma": 2.0,   # 0.977 confidence
}


def t4_distribution(params: SeparationParams | None = None) -> tuple[float, float]:
    """Return (mean, std) of T4 as the sum of three independent normals.

    Reproduces D2.1 Table G-2 (p.124): total mean 7.82 s, total std 4.24 s.
    """
    params = params or SeparationParams()
    mean = sum(params.t4_delay_means_s)
    std = math.sqrt(sum(s * s for s in params.t4_delay_stds_s))
    return mean, std


def t4_value(params: SeparationParams | None = None, confidence_sigma: float | None = None) -> float:
    """T4 time window at the requested confidence (mean + k*std)."""
    params = params or SeparationParams()
    mean, std = t4_distribution(params)
    k = params.t4_confidence_sigma if confidence_sigma is None else float(confidence_sigma)
    return mean + k * std


# --------------------------------------------------------------------------- #
# E7v2 (task #23): shared-link non-preemptive priority M/G/1 waiting time.     #
# --------------------------------------------------------------------------- #
# The tactical loop delivers evidence over a SHARED radio link; before v6 the
# three E7 call sites (separation-capacity script, spec-attainability certificate
# queue term, env queue_delay) each modelled queueing differently.  This is the
# ONE shared estimator: the Pollaczek-Khinchine mean wait for a service-time
# distribution S with utilisation rho, W = rho * E[S^2] / (2 * E[S]), extended to
# non-preemptive priority classes (C2 / critical evidence is high-priority).
#
# For a single aggregate class this is exactly the prompt's
#     W = rho * E[S^2] / (2 * E[S]).
# For K non-preemptive priority classes (index 0 = highest priority), the mean
# wait of class k is the standard Cobham (1954) result
#     W_k = R / ((1 - sigma_{k-1}) * (1 - sigma_k)),
#     R = sum_i lambda_i * E[S_i^2] / 2   (aggregate mean residual service),
#     sigma_k = sum_{i<=k} rho_i,  rho_i = lambda_i * E[S_i].


def mg1_pk_wait(rho: float, mean_service_s: float, second_moment_s2: float) -> float:
    """Pollaczek-Khinchine mean waiting time W = rho*E[S^2]/(2*E[S]).

    rho              : link utilisation of this traffic (lambda * E[S]), in [0,1).
    mean_service_s   : E[S], mean airtime per evidence delivery (s).
    second_moment_s2 : E[S^2] (s^2); for a deterministic S it equals E[S]^2, for
                       exponential 2*E[S]^2, generally = E[S]^2 + Var[S].
    Returns 0 for a saturated/degenerate queue guard (rho>=1 or E[S]<=0).
    """
    rho = float(rho)
    mean_service_s = float(mean_service_s)
    if mean_service_s <= 0.0 or rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return float("inf")
    return rho * float(second_moment_s2) / (2.0 * mean_service_s)


def mg1_priority_wait(classes: list[dict], target_index: int) -> float:
    """Non-preemptive priority M/G/1 mean wait for class `target_index`.

    classes: ordered HIGH->LOW priority; each a dict with keys
        lam   -- arrival rate lambda_i (1/s),
        es    -- mean service E[S_i] (s),
        es2   -- second moment E[S_i^2] (s^2)  [defaults to es^2 = deterministic].
    C2 / critical evidence is placed at index 0 (highest priority) by the caller.
    Returns the Cobham waiting time; inf if the cumulative load through the class
    saturates the link.
    """
    if not classes or not (0 <= target_index < len(classes)):
        return 0.0
    # aggregate mean residual R = sum_i lambda_i E[S_i^2] / 2
    residual = 0.0
    rhos: list[float] = []
    for c in classes:
        lam = float(c.get("lam", 0.0))
        es = float(c.get("es", 0.0))
        es2 = float(c.get("es2", es * es))
        residual += lam * es2 / 2.0
        rhos.append(lam * es)
    sigma_prev = sum(rhos[:target_index])            # sigma_{k-1}
    sigma_k = sigma_prev + rhos[target_index]         # sigma_k
    denom = (1.0 - sigma_prev) * (1.0 - sigma_k)
    if denom <= 0.0:
        return float("inf")
    return residual / denom


@dataclass(frozen=True)
class SeparationMinima:
    """Result of the D2.1 Block-4 separation-minima chain (metres)."""

    d_nmac_m: float
    d_ic_m: float
    d_sl_m: float
    d_tc_m: float
    h_nmac_m: float
    h_ic_m: float
    h_sl_m: float
    h_tc_m: float
    t1_s: float
    t2_s: float
    t3_s: float
    t4_s: float


def separation_minima(
    performance: TrafficPerformance,
    params: SeparationParams | None = None,
    confidence_sigma: float | None = None,
) -> SeparationMinima:
    """Compute the full horizontal and vertical separation-minima chain.

    Implements D2.1 Sec. 3.3.4 (p.60-61):

        d_IC = d_NMAC + TSE_h(95%) + V_c*(T1 + T2)
        h_IC = h_NMAC + TSE_v(95%) + V_v*T2,      T1 = (h_NMAC + TSE_h(95%))/ROC
        d_SL = d_IC + SSE_h + V_c*T_res
        h_SL = h_IC + SSE_v + V_v*T_res
        d_TC = d_SL + V_c*(T3 + T4),               T3 = h_SL/ROC
        h_TC = h_SL + V_v*T4
    """
    params = params or SeparationParams()
    vc = performance.cruise_mps
    roc = performance.roc_mps
    vv = params.vertical_ref_speed_mps

    d_nmac = D_NMAC_FT * FT_TO_M
    h_nmac = H_NMAC_FT * FT_TO_M
    tse_h95 = params.tse_h_95_m()
    tse_v95 = params.tse_v_95_m()

    # Imminent-collision (IC) distance.
    h_avoid = h_nmac + tse_h95
    t1 = h_avoid / roc if roc > 0 else 0.0
    t2 = params.t2_s
    d_ic = d_nmac + tse_h95 + vc * (t1 + t2)
    h_ic = h_nmac + tse_v95 + vv * t2

    # Separation-loss (SL) distance.
    d_sl = d_ic + params.sse_h_2sigma_m + vc * params.t_res_s
    h_sl = h_ic + params.sse_v_2sigma_m + vv * params.t_res_s

    # Tactical-conflict (TC) distance.
    t3 = h_sl / roc if roc > 0 else 0.0
    t4 = t4_value(params, confidence_sigma)
    d_tc = d_sl + vc * (t3 + t4)
    h_tc = h_sl + vv * t4

    return SeparationMinima(
        d_nmac_m=d_nmac,
        d_ic_m=d_ic,
        d_sl_m=d_sl,
        d_tc_m=d_tc,
        h_nmac_m=h_nmac,
        h_ic_m=h_ic,
        h_sl_m=h_sl,
        h_tc_m=h_tc,
        t1_s=t1,
        t2_s=t2,
        t3_s=t3,
        t4_s=t4,
    )


def tactical_conflict_distance(
    performance: TrafficPerformance,
    params: SeparationParams | None = None,
    confidence_sigma: float | None = None,
) -> tuple[float, float]:
    """Convenience wrapper returning ``(d_TC, h_TC)`` for a traffic class."""
    minima = separation_minima(performance, params, confidence_sigma)
    return minima.d_tc_m, minima.h_tc_m


# --------------------------------------------------------------------------- #
# Closest-point-of-approach (CPA) geometry and the two-condition TC criterion. #
# --------------------------------------------------------------------------- #

Vec = tuple[float, float, float]


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot2(a: Vec, b: Vec) -> float:
    """Horizontal (x, y) dot product."""
    return a[0] * b[0] + a[1] * b[1]


@dataclass(frozen=True)
class CPAResult:
    time_to_cpa_s: float          # >= 0; inf when the aircraft never converge
    horizontal_sep_m: float       # horizontal distance at the predicted CPA
    vertical_sep_m: float         # |dz| at the predicted CPA
    slant_sep_m: float            # 3-D distance at the predicted CPA


def closest_point_of_approach(p1: Vec, v1: Vec, p2: Vec, v2: Vec) -> CPAResult:
    """Predicted CPA of two aircraft assuming constant velocity.

    The time-to-CPA is derived from the horizontal relative motion (the plane in
    which the BUBBLES tactical-conflict horizon is defined). Aircraft that are
    not horizontally converging (near-zero relative horizontal speed, e.g. exact
    parallel same-speed flight) yield ``time_to_cpa_s = +inf`` so that they are
    never declared a tactical conflict.
    """
    rel_p = _sub(p1, p2)
    rel_v = _sub(v1, v2)
    rel_v_sq = _dot2(rel_v, rel_v)
    if rel_v_sq <= 1e-12:
        t_cpa = math.inf
        hp = rel_p
    else:
        t = -_dot2(rel_p, rel_v) / rel_v_sq
        t_cpa = t if t > 0.0 else 0.0
        hp = (rel_p[0] + rel_v[0] * t_cpa, rel_p[1] + rel_v[1] * t_cpa, rel_p[2] + rel_v[2] * t_cpa)
    if math.isinf(t_cpa):
        horizontal = math.hypot(rel_p[0], rel_p[1])
        vertical = abs(rel_p[2])
    else:
        horizontal = math.hypot(hp[0], hp[1])
        vertical = abs(hp[2])
    slant = math.hypot(horizontal, vertical)
    return CPAResult(
        time_to_cpa_s=t_cpa,
        horizontal_sep_m=horizontal,
        vertical_sep_m=vertical,
        slant_sep_m=slant,
    )


def is_tactical_conflict(
    p1: Vec,
    v1: Vec,
    p2: Vec,
    v2: Vec,
    d_tc_m: float,
    tc_threshold_s: float,
    h_tc_m: float | None = None,
) -> bool:
    """Two-condition BUBBLES tactical-conflict declaration (D2.1 p.38 / p.61).

    A conflict is declared iff **both** hold:
      1. horizontal separation at the CPA < ``d_tc_m`` (and, if ``h_tc_m`` is
         supplied, vertical separation at the CPA < ``h_tc_m``);
      2. time remaining to the CPA < ``tc_threshold_s`` (and the CPA is in the
         future, ``t_cpa >= 0``).
    """
    cpa = closest_point_of_approach(p1, v1, p2, v2)
    if not (0.0 <= cpa.time_to_cpa_s < tc_threshold_s):
        return False
    horizontal_breach = cpa.horizontal_sep_m < d_tc_m
    if h_tc_m is None:
        return horizontal_breach
    return horizontal_breach and cpa.vertical_sep_m < h_tc_m


# 24-hour normalised demand curve, D2.1 Table G-13 (p.128). Values are the
# simultaneous-UAS counts per hour; peak concurrency 20, daily mean 8.17,
# 784 operations/day at 15 min mean duration.
BUBBLES_DAILY_DEMAND: tuple[int, ...] = (
    1, 1, 0, 0, 0, 1, 5, 10, 15, 20, 20, 20,
    20, 20, 10, 15, 15, 10, 5, 3, 2, 1, 1, 1,
)
BUBBLES_DAILY_PEAK_CONCURRENCY: int = 20
BUBBLES_DAILY_MEAN_CONCURRENCY: float = sum(BUBBLES_DAILY_DEMAND) / 24.0  # ~= 8.17


def bubbles_daily_generation_step(idx: int, count: int, episode_steps: int) -> int:
    """Map task index ``idx`` (of ``count``) to a generation step following the
    Table G-13 daily demand shape, scaled to ``episode_steps``.

    Tasks are placed by inverse-CDF sampling of the hourly demand so that busy
    hours (peak concurrency 20) receive proportionally more arrivals.
    """
    count = max(1, int(count))
    steps = max(1, int(episode_steps))
    total = float(sum(BUBBLES_DAILY_DEMAND))
    target = (idx + 0.5) / count * total
    cumulative = 0.0
    hour = 0
    for hour, demand in enumerate(BUBBLES_DAILY_DEMAND):
        cumulative += demand
        if cumulative >= target:
            break
    step = int(hour / 24.0 * steps)
    return max(0, min(steps - 1, step))
