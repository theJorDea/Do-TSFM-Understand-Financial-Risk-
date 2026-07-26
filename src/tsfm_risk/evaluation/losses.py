"""Scoring rules (losses). Lower is better throughout.

QLIKE (volatility, Patton 2011)
    L(s2_hat; p) = ln s2_hat + p / s2_hat
    with p a conditionally unbiased variance proxy (squared return, RV).
    Robust to proxy noise: the expected-loss ranking under a noisy proxy
    matches the ranking under true variance. This form is well defined for
    p = 0 (unlike the p/s2 - ln(p/s2) - 1 variant) and shares its argmin.

FZ0 (joint VaR-ES, Fissler & Ziegel 2016; 0-homogeneous parameterization
of Patton, Ziegel & Chen 2019, eq. 4)
    With v = return-space VaR quantile (negative in the tail),
    e = return-space ES (e <= v < 0), hit = 1{r <= v}:

    L = -hit * (v - r) / (alpha * e) + v / e + ln(-e) - 1

    Strictly consistent for the (VaR, ES) pair: the true conditional
    (VaR, ES) uniquely minimizes expected loss. Our public API takes VaR/ES
    as positive loss magnitudes (project convention) and converts.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(*xs: np.ndarray) -> tuple[np.ndarray, ...]:
    out = tuple(np.asarray(x, dtype=float) for x in xs)
    n = out[0].shape
    if any(o.shape != n for o in out):
        raise ValueError("all inputs must share the same shape")
    return out


def qlike(variance_forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE; ``proxy`` >= 0, ``variance_forecast`` > 0."""
    s2, p = _as_arrays(variance_forecast, proxy)
    if np.any(s2 <= 0):
        raise ValueError("variance forecasts must be strictly positive")
    if np.any(p < 0):
        raise ValueError("variance proxy must be non-negative")
    return np.log(s2) + p / s2


def mse_variance(variance_forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    s2, p = _as_arrays(variance_forecast, proxy)
    return (s2 - p) ** 2


def fz0(
    var: np.ndarray,
    es: np.ndarray,
    returns: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Per-observation FZ0 loss.

    ``var``/``es`` are positive loss magnitudes (project convention),
    ``returns`` realized returns over the matching horizon.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    v_pos, e_pos, r = _as_arrays(var, es, returns)
    if np.any(e_pos <= 0):
        raise ValueError("ES must be strictly positive (loss magnitude)")
    if np.any(e_pos < v_pos):
        raise ValueError("ES must be >= VaR")
    v = -v_pos  # to return space
    e = -e_pos
    hit = (r <= v).astype(float)
    return -hit * (v - r) / (alpha * e) + v / e + np.log(-e) - 1.0
