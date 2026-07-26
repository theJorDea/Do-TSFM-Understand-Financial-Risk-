"""Value-at-Risk and Expected Shortfall.

Conventions (fixed in docs/preregistration.md):

- ``alpha`` is the tail probability (e.g. 0.01 for 99% VaR);
- VaR and ES are reported as **positive loss magnitudes**:
  ``VaR_a = -q_a(r)`` where ``q_a`` is the a-quantile of the return
  distribution, and ``ES_a = -E[r | r <= q_a(r)]``.

Closed forms
------------
Normal, r ~ N(mu, sigma^2):
    VaR_a = -(mu + sigma * z_a),                z_a = Phi^{-1}(a)
    ES_a  = -mu + sigma * phi(z_a) / a          (McNeil, Frey & Embrechts 2015, ex. 2.14)

Standardized Student-t (unit variance), r = mu + sigma * X,
X = sqrt((nu-2)/nu) * T_nu:
    VaR_a = -(mu + sigma * sqrt((nu-2)/nu) * t_a),   t_a = F_nu^{-1}(a)
    ES_a  = -mu + sigma * sqrt((nu-2)/nu)
                  * f_nu(t_a)/a * (nu + t_a^2)/(nu - 1)
(tail expectation of the Student-t; McNeil, Frey & Embrechts 2015, ex. 2.15).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")


def var_es_normal(sigma: float, alpha: float, mu: float = 0.0) -> tuple[float, float]:
    """VaR and ES (positive losses) for r ~ N(mu, sigma^2)."""
    _check_alpha(alpha)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    z = stats.norm.ppf(alpha)
    var = -(mu + sigma * z)
    es = -mu + sigma * stats.norm.pdf(z) / alpha
    return var, es


def var_es_student_t(
    sigma: float, nu: float, alpha: float, mu: float = 0.0
) -> tuple[float, float]:
    """VaR and ES for r = mu + sigma * X with X standardized Student-t(nu).

    ``sigma`` is the standard deviation of r (the t is scaled to unit
    variance), so GARCH-t plugs its conditional sigma in directly.
    """
    _check_alpha(alpha)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if nu <= 2:
        raise ValueError("nu must exceed 2 for finite variance")
    scale = np.sqrt((nu - 2.0) / nu)
    t_a = stats.t.ppf(alpha, df=nu)
    var = -(mu + sigma * scale * t_a)
    es_std = stats.t.pdf(t_a, df=nu) / alpha * (nu + t_a**2) / (nu - 1.0)
    es = -mu + sigma * scale * es_std
    return var, es


def var_es_empirical(sample: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR/ES from a sample of returns (historical simulation).

    VaR is the empirical a-quantile (linear interpolation); ES averages all
    observations at or below that quantile.
    """
    _check_alpha(alpha)
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        raise ValueError(f"need >= 100 observations for empirical tail, got {x.size}")
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return -q, -float(tail.mean())


def var_es_from_quantiles(
    q_levels: np.ndarray, q_values: np.ndarray, alpha: float
) -> tuple[float, float]:
    """VaR/ES from a discrete predictive quantile function (the TSFM path).

    ``q_levels`` are probabilities (ascending, must bracket ``alpha``),
    ``q_values`` the corresponding return quantiles. VaR interpolates the
    quantile function at ``alpha``; ES integrates it over (0, alpha] via the
    trapezoidal rule on the available grid — the standard discrete
    approximation of ES_a = -(1/a) * int_0^a q_u du.
    """
    _check_alpha(alpha)
    lv = np.asarray(q_levels, dtype=float)
    qv = np.asarray(q_values, dtype=float)
    if lv.ndim != 1 or lv.shape != qv.shape:
        raise ValueError("q_levels and q_values must be 1-D and equal length")
    if np.any(np.diff(lv) <= 0):
        raise ValueError("q_levels must be strictly increasing")
    if np.any(np.diff(qv) < 0):
        raise ValueError("q_values must be non-decreasing (a valid quantile function)")
    if alpha < lv[0] or alpha > lv[-1]:
        raise ValueError(f"alpha={alpha} outside quantile grid [{lv[0]}, {lv[-1]}]")

    var = -float(np.interp(alpha, lv, qv))

    # integrate q(u) over [lv[0], alpha], then extend to (0, lv[0]) by the
    # lowest available quantile (conservative flat-tail assumption, reported
    # as a limitation when the grid is coarse).
    mask = lv <= alpha
    grid_l = np.append(lv[mask], alpha)
    grid_q = np.append(qv[mask], -var)
    integral = np.trapezoid(grid_q, grid_l) + lv[0] * qv[0]
    es = -integral / alpha
    return var, float(max(es, var))
