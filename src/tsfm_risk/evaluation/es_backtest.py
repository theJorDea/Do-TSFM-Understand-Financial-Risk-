"""Acerbi-Szekely (2014) direct ES backtest, unconditional Z2 statistic.

    Z2 = sum_t( X_t * I_t / (T * alpha * ES_t) ) + 1,
    I_t = 1{X_t < -VaR_t},  ES_t > 0 (loss magnitude), X_t realized return.

Under correct (VaR, ES) forecasts E[Z2] = 0; ES *under*-estimation drives
Z2 negative. The p-value is one-sided by Monte Carlo under H0.

H0 simulation needs a predictive distribution per day, which the stored
(VaR_t, ES_t) pair does not pin down. We use the *implied-nu* location-scale
Student-t: for each day, choose the t whose alpha-tail reproduces both the
VaR and the ES of the forecast (the ES/VaR ratio identifies nu; the VaR
level identifies the scale). Pairs with ratios at or below the Gaussian
limit fall back to a VaR-matched normal. This keeps the null anchored to
the model's own stated tail rather than to an arbitrary reference family;
the choice is documented in the paper and stress-checked in tests.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from tsfm_risk.evaluation.backtests import BacktestResult
from tsfm_risk.risk.measures import var_es_normal, var_es_student_t

_NU_LO, _NU_HI = 2.05, 200.0


def _es_var_ratio_t(nu: float, alpha: float) -> float:
    var, es = var_es_student_t(1.0, nu, alpha)
    return es / var


def implied_nu(var: float, es: float, alpha: float) -> float | None:
    """Degrees of freedom of the standardized t matching the ES/VaR ratio.

    Returns None when the ratio is at or below the Gaussian limit
    (fall back to normal).
    """
    if es <= var:
        raise ValueError("ES must exceed VaR")
    ratio = es / var
    _, es_n = var_es_normal(1.0, alpha)
    var_n, _ = var_es_normal(1.0, alpha)
    gauss_ratio = es_n / var_n
    if ratio <= gauss_ratio * (1.0 + 1e-9):
        return None
    lo_ratio = _es_var_ratio_t(_NU_LO, alpha)
    if ratio >= lo_ratio:
        return _NU_LO
    return float(
        optimize.brentq(lambda nu: _es_var_ratio_t(nu, alpha) - ratio, _NU_LO, _NU_HI)
    )


def acerbi_szekely_z2(
    returns: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    alpha: float,
    n_sim: int = 2000,
    seed: int = 2026,
) -> BacktestResult:
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    if not (r.shape == v.shape == e.shape) or r.ndim != 1:
        raise ValueError("inputs must be aligned 1-D arrays")
    if np.any(e < v) or np.any(v <= 0):
        raise ValueError("need ES >= VaR > 0 (positive loss magnitudes)")
    t_len = r.size
    if t_len < 100:
        raise ValueError("as-z2: need >= 100 observations")

    hits = r < -v
    z2_obs = float(np.sum(r * hits / (t_len * alpha * e)) + 1.0)

    # per-day implied predictive distribution (scale in return space);
    # root-find once per unique (VaR, ES) pair
    pairs = np.column_stack([v, e])
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    u_nus = np.empty(uniq.shape[0])
    u_scales = np.empty(uniq.shape[0])
    for i, (vi, ei) in enumerate(uniq):
        nu = implied_nu(float(vi), float(ei), alpha)
        if nu is None:
            u_nus[i] = np.inf
            u_scales[i] = vi / -stats.norm.ppf(alpha)
        else:
            u_nus[i] = nu
            var_std, _ = var_es_student_t(1.0, nu, alpha)
            u_scales[i] = vi / var_std
    nus = u_nus[inverse]
    scales = u_scales[inverse]

    rng = np.random.default_rng(seed)
    normal_mask = np.isinf(nus)
    x = np.empty((n_sim, t_len))
    if normal_mask.any():
        x[:, normal_mask] = rng.standard_normal((n_sim, int(normal_mask.sum())))
    if (~normal_mask).any():
        nu_arr = nus[~normal_mask]
        draws = rng.standard_t(np.broadcast_to(nu_arr, (n_sim, nu_arr.size)))
        x[:, ~normal_mask] = draws * np.sqrt((nu_arr - 2.0) / nu_arr)
    x *= scales

    sim_hits = x < -v
    z2_sim = np.sum(x * sim_hits / (t_len * alpha * e), axis=1) + 1.0
    p_value = float(np.mean(z2_sim <= z2_obs))  # one-sided: ES understated

    return BacktestResult(
        "acerbi_szekely_z2", z2_obs, p_value, t_len, int(hits.sum())
    )
