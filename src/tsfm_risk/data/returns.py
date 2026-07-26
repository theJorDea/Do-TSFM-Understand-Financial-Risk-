"""Log-returns and volatility proxies.

Conventions (shared with Riskforge):
- log-returns in decimal units, ``r_t = ln(P_t / P_{t-1})``;
- volatility proxy for day t is the squared return ``r_t**2`` — a noisy but
  conditionally unbiased proxy; losses must be proxy-robust (QLIKE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log-returns; drops the first observation and non-positive prices."""
    p = prices[prices > 0].astype(float)
    r = np.log(p).diff().dropna()
    r.name = prices.name
    return r


def squared_return_proxy(returns: pd.Series) -> pd.Series:
    """Conditionally unbiased (noisy) proxy for daily variance."""
    return returns.pow(2).rename(returns.name)


def aggregate_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """h-day forward cumulative log-return aligned at the forecast origin.

    The value at date t is the return over (t, t+h], i.e. what a forecast
    issued at t must predict. Trailing dates without a full window are dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    fwd = returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return fwd.dropna()


def clean_returns(returns: pd.Series, max_abs: float = 1.0) -> pd.Series:
    """Drop obviously corrupt observations (|log-return| > max_abs, i.e. >170% move).

    Removals must be rare and are reported in the data card; the threshold is
    deliberately loose so genuine crashes (-20%, -30%) always survive.
    """
    mask = returns.abs() <= max_abs
    return returns[mask]
