"""Mapping a TSFM's predictive quantiles to VaR and ES.

Two declared paths (``docs/preregistration.md`` §3.1, revised by amendment 1):

**Vol-path (primary).** Read a scale from the central quantiles, aggregate it
over the horizon, and apply a location-scale model with Student-t innovations —
the same recipe the GARCH baselines use. Primary because most models in the
roster emit quantiles only on a 0.1-0.9 grid, so VaR at 1-5% is not directly
available from them; extrapolating past the lowest predicted quantile would
silently reimpose a distributional assumption.

**Direct-quantile path (secondary).** Read VaR straight off the predictive
quantile function and integrate its tail for ES. Reported only where the model's
native grid actually reaches the requested alpha. Available at h=1 for any
model; for h>1 it needs the *joint* distribution of the h-day sum, which
marginal quantiles do not pin down, so it is restricted to models that emit
sample paths.

The difference between the two paths, where both exist, measures the model's own
tail shape as distinct from its volatility level — the decomposition RQ4 needs.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

from tsfm_risk.models.base import RiskForecast
from tsfm_risk.risk.measures import (
    var_es_from_quantiles,
    var_es_normal,
    var_es_student_t,
)

CENTRAL_LO, CENTRAL_HI = 0.1, 0.9


def scale_from_quantiles(
    levels: np.ndarray,
    values: np.ndarray,
    lo: float = CENTRAL_LO,
    hi: float = CENTRAL_HI,
) -> float:
    """Gaussian-equivalent scale implied by the *central* predictive quantiles.

    Least squares of ``values`` on ``Phi^{-1}(levels)`` over levels in
    ``[lo, hi]``; the slope is the scale. Restricting to the central region is
    deliberate: it uses only the part of the grid every model reports, and it
    keeps the scale estimate independent of the tail shape, which is exactly the
    quantity the other path is meant to measure. Using the full grid instead
    would let a fat-tailed predictive distribution inflate "volatility" and
    confound the two effects.

    With >= 3 central levels this is more efficient than the IQR rule and
    reduces to it in the Gaussian case.
    """
    lv = np.asarray(levels, dtype=float)
    q = np.asarray(values, dtype=float)
    if lv.shape != q.shape or lv.ndim != 1:
        raise ValueError("levels and values must be 1-D and equal length")
    mask = (lv >= lo) & (lv <= hi)
    if mask.sum() < 2:
        raise ValueError(f"need >= 2 quantile levels within [{lo}, {hi}]")
    z = stats.norm.ppf(lv[mask])
    y = q[mask]
    # slope of y on z with intercept (the intercept absorbs any location shift)
    z_c = z - z.mean()
    denom = float(z_c @ z_c)
    if denom <= 0:
        raise ValueError("degenerate quantile levels")
    scale = float((z_c @ (y - y.mean())) / denom)
    if scale <= 0:
        raise ValueError("implied scale is non-positive; quantiles are not increasing")
    return scale


def fit_student_t_nu(z: np.ndarray, nu_bounds: tuple[float, float] = (2.1, 100.0)) -> float:
    """MLE degrees of freedom of a standardized Student-t fitted to residuals.

    ``z`` are standardized residuals ``r_t / sigma_t`` built from the model's own
    past forecasts, so the innovation distribution is estimated the same way for
    every model in the study and uses only information available at the origin.
    """
    x = np.asarray(z, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        raise ValueError(f"need >= 100 residuals to fit nu, got {x.size}")

    def nll(nu: float) -> float:
        c = (
            special.gammaln((nu + 1) / 2)
            - special.gammaln(nu / 2)
            - 0.5 * np.log(np.pi * (nu - 2))
        )
        return -float(np.sum(c - (nu + 1) / 2 * np.log1p(x**2 / (nu - 2))))

    res = optimize.minimize_scalar(nll, bounds=nu_bounds, method="bounded")
    return float(res.x)


def vol_path_forecast(
    levels: np.ndarray,
    quantiles: np.ndarray,
    horizons: tuple[int, ...],
    alphas: tuple[float, ...],
    nu: float | None = None,
) -> list[RiskForecast]:
    """Vol-path: per-day scales -> aggregate variance -> location-scale VaR/ES.

    ``quantiles`` has shape ``(max(horizons), len(levels))``. The h-day variance
    is the sum of per-day variances, which assumes serially uncorrelated returns
    — the same assumption every baseline makes when aggregating.

    ``nu`` selects the innovation law: Student-t when given, normal otherwise.
    """
    lv = np.asarray(levels, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 2 or q.shape[1] != lv.size:
        raise ValueError("quantiles must have shape (horizon, len(levels))")
    if max(horizons) > q.shape[0]:
        raise ValueError(f"need {max(horizons)} forecast days, got {q.shape[0]}")

    daily_var = np.array([scale_from_quantiles(lv, q[i]) ** 2 for i in range(q.shape[0])])

    out: list[RiskForecast] = []
    for h in horizons:
        agg = float(daily_var[:h].sum())
        sigma = float(np.sqrt(agg))
        var_d, es_d = {}, {}
        for a in alphas:
            if nu is None:
                var_d[a], es_d[a] = var_es_normal(sigma, a)
            else:
                var_d[a], es_d[a] = var_es_student_t(sigma, nu, a)
        out.append(RiskForecast(horizon=h, variance=agg, var=var_d, es=es_d))
    return out


def direct_quantile_forecast(
    levels: np.ndarray,
    quantiles: np.ndarray,
    alphas: tuple[float, ...],
) -> RiskForecast:
    """Direct path at h=1: VaR/ES read off the predictive quantile function.

    Raises if ``alphas`` reach outside the supplied grid — extrapolating below
    the lowest predicted quantile is exactly what this path exists to avoid.
    """
    lv = np.asarray(levels, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if q.ndim == 2:
        q = q[0]
    var_d, es_d = {}, {}
    for a in alphas:
        var_d[a], es_d[a] = var_es_from_quantiles(lv, q, a)
    variance = scale_from_quantiles(lv, q) ** 2
    return RiskForecast(horizon=1, variance=variance, var=var_d, es=es_d)
