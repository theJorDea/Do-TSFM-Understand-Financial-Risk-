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


def find_spike_reversals(
    returns: pd.Series,
    n_scales: float = 12.0,
    retracement: float = 0.8,
    window: int = 121,
) -> pd.DatetimeIndex:
    """Dates that look like a single bad price print rather than a market move.

    A vendor typo makes the series jump and snap straight back, so both
    conditions below must hold for day t:

    1. **Implausible size**: ``|r_t|`` exceeds ``n_scales`` times the *local*
       typical move (median ``|r|`` over a centred ``window``, which is robust
       to the spike itself). Scaling locally is what separates the two cases —
       a 16% EUR/USD print is ~40 normal days, whereas the -10% S&P 500 move of
       2020-03-12 is barely 2 days of the volatility then prevailing.
    2. **Near-exact reversal**: the next day retraces at least ``retracement``
       of it, i.e. ``|r_t + r_{t+1}| <= (1 - retracement) * |r_t|``.

    Retracement alone is not enough: real crashes whipsaw too (2020-03-12/13
    retraced ~89%, the same as the EUR/USD artifact). Size alone is not enough
    either: Apple's genuine -52% on 2000-09-29 is huge but never comes back.

    Returns the dates only — the caller decides whether to drop them, and every
    removal belongs in the data card.
    """
    r = returns.astype(float)
    if len(r) < 10:
        return pd.DatetimeIndex([])
    scale = r.abs().rolling(window, center=True, min_periods=20).median()
    nxt = r.shift(-1)
    big = r.abs() > n_scales * scale
    reverts = (r + nxt).abs() <= (1.0 - retracement) * r.abs()
    flagged = (big & reverts).fillna(False)
    return pd.DatetimeIndex(r.index[flagged])


def drop_spike_reversals(
    returns: pd.Series,
    n_scales: float = 12.0,
    retracement: float = 0.8,
    window: int = 121,
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Remove bad-tick pairs (the spike and its reversal). Returns (clean, dropped)."""
    spikes = find_spike_reversals(returns, n_scales, retracement, window)
    if len(spikes) == 0:
        return returns, spikes
    pos = returns.index.get_indexer(spikes)
    drop_pos = np.unique(np.concatenate([pos, pos + 1]))
    drop_pos = drop_pos[drop_pos < len(returns)]
    keep = np.ones(len(returns), dtype=bool)
    keep[drop_pos] = False
    return returns[keep], spikes
