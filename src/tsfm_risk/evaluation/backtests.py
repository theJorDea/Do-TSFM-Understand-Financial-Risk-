"""Regulatory-grade VaR backtests.

All tests take the realized returns and the VaR forecasts (positive loss
magnitudes, project convention) for a single alpha and return a
``BacktestResult`` with the statistic, p-value and breach counts.

- Kupiec (1995) proportion-of-failures: unconditional coverage, LR ~ chi2(1).
- Christoffersen (1998): independence of breaches (first-order Markov),
  LR ~ chi2(1); conditional coverage combines both, LR ~ chi2(2).
- Engle & Manganelli (2004) dynamic quantile: regresses centered hits on a
  constant, lagged hits and the current VaR level; under correct
  specification the DQ statistic is chi2(k). Catches dependence on the
  VaR level itself that Markov tests miss.

Overlapping multi-day observations violate the independence assumptions of
these tests; for h > 1 the paper reports them descriptively and relies on
DM/MCS with HAC errors for ranking (see docs/preregistration.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BacktestResult:
    name: str
    statistic: float
    p_value: float
    n_obs: int
    n_breaches: int

    @property
    def breach_rate(self) -> float:
        return self.n_breaches / self.n_obs if self.n_obs else np.nan


def _hits(returns: np.ndarray, var: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    if r.shape != v.shape or r.ndim != 1:
        raise ValueError("returns and var must be 1-D arrays of equal length")
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(v)):
        raise ValueError("inputs contain non-finite values")
    return (r < -v).astype(int)


def kupiec_pof(returns: np.ndarray, var: np.ndarray, alpha: float) -> BacktestResult:
    h = _hits(returns, var)
    n, x = h.size, int(h.sum())
    if n < 50:
        raise ValueError("kupiec: need >= 50 observations")
    p_hat = x / n
    # log-likelihood under H0 (p=alpha) vs MLE (p=p_hat); guard x in {0, n}
    def ll(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            p = min(max(p, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1.0 - p) + x * np.log(p)

    lr = -2.0 * (ll(alpha) - ll(p_hat)) if x not in (0, n) else -2.0 * ll(alpha)
    p_value = float(stats.chi2.sf(lr, df=1))
    return BacktestResult("kupiec_pof", float(lr), p_value, n, x)


def christoffersen_independence(returns: np.ndarray, var: np.ndarray) -> BacktestResult:
    h = _hits(returns, var)
    n = h.size
    if n < 50:
        raise ValueError("christoffersen: need >= 50 observations")
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    def safe_log(p: float) -> float:
        return np.log(min(max(p, 1e-12), 1.0))

    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    ll0 = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    ll1 = (
        n00 * safe_log(1 - pi0)
        + n01 * safe_log(pi0)
        + n10 * safe_log(1 - pi1)
        + n11 * safe_log(pi1)
    )
    lr = -2.0 * (ll0 - ll1)
    p_value = float(stats.chi2.sf(lr, df=1))
    return BacktestResult("christoffersen_ind", float(lr), p_value, n, int(h.sum()))


def christoffersen_conditional_coverage(
    returns: np.ndarray, var: np.ndarray, alpha: float
) -> BacktestResult:
    pof = kupiec_pof(returns, var, alpha)
    ind = christoffersen_independence(returns, var)
    lr = pof.statistic + ind.statistic
    p_value = float(stats.chi2.sf(lr, df=2))
    return BacktestResult("christoffersen_cc", float(lr), p_value, pof.n_obs, pof.n_breaches)


def engle_manganelli_dq(
    returns: np.ndarray, var: np.ndarray, alpha: float, n_lags: int = 4
) -> BacktestResult:
    h = _hits(returns, var)
    n = h.size
    if n < 100:
        raise ValueError("dq: need >= 100 observations")
    d = h.astype(float) - alpha  # centered hit sequence

    t0 = n_lags
    y = d[t0:]
    cols = [np.ones(n - t0)]
    for lag in range(1, n_lags + 1):
        cols.append(d[t0 - lag : n - lag])
    cols.append(np.asarray(var, dtype=float)[t0:])  # VaR level regressor
    x = np.column_stack(cols)

    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    xty = x.T @ y
    dq = float(xty @ xtx_inv @ xty) / (alpha * (1.0 - alpha))
    k = x.shape[1]
    p_value = float(stats.chi2.sf(dq, df=k))
    return BacktestResult("engle_manganelli_dq", dq, p_value, n, int(h.sum()))
