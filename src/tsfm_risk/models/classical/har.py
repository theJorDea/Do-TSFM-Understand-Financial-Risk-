"""HAR model on a daily variance proxy (Corsi 2009).

The heterogeneous autoregression regresses future variance on daily, weekly
(5-day) and monthly (22-day) averages of past variance:

    y_t(h) = b0 + b_d * v_t + b_w * mean(v_{t-4..t}) + b_m * mean(v_{t-21..t}) + e

where v_t is the variance proxy for day t and the target y_t(h) is the
*forward h-day aggregate* variance sum_{k=1..h} v_{t+k}. One OLS per horizon
(direct projection) — the standard multi-step HAR variant, avoiding
iterated-forecast bias.

With daily data only, v_t = r_t^2 (noisy but conditionally unbiased —
"HAR-proxy"); when 5-minute realized variance is available the same class
runs on RV (true HAR-RV). The paper reports HAR primarily as the volatility
benchmark under QLIKE; its tail path uses normal innovations and is
secondary.
"""

from __future__ import annotations

import numpy as np

from tsfm_risk.models.base import RiskForecast, RiskModel
from tsfm_risk.risk.measures import var_es_normal

_LAGS = (1, 5, 22)


def _har_regressors(v: np.ndarray) -> np.ndarray:
    """Design matrix aligned at t (usable from index max(_LAGS)-1 onward)."""
    n = v.size
    m = max(_LAGS)
    c = np.concatenate([[0.0], np.cumsum(v)])
    rows = n - m + 1
    x = np.empty((rows, 1 + len(_LAGS)))
    x[:, 0] = 1.0
    t = np.arange(m - 1, n)  # regressor date index
    for j, lag in enumerate(_LAGS):
        x[:, 1 + j] = (c[t + 1] - c[t + 1 - lag]) / lag
    return x


class Har(RiskModel):
    def __init__(self, horizons: tuple[int, ...] = (1, 5, 20)):
        self.name = "har-proxy"
        self._coefs: dict[int, np.ndarray] = {}
        self._horizons = horizons

    def fit(self, returns: np.ndarray) -> None:
        r = self._validate_window(returns)
        v = r * r
        x_all = _har_regressors(v)
        m = max(_LAGS)
        c = np.concatenate([[0.0], np.cumsum(v)])
        self._coefs = {}
        for h in self._horizons:
            # target: forward h-day variance sum for origins m-1 .. n-h-1
            t = np.arange(m - 1, v.size - h)
            y = c[t + 1 + h] - c[t + 1]
            x = x_all[: t.size]
            if y.size < 100:
                raise ValueError(f"har: too few observations for horizon {h}")
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            self._coefs[h] = coef

    def forecast(
        self,
        returns: np.ndarray,
        horizons: tuple[int, ...],
        alphas: tuple[float, ...],
    ) -> list[RiskForecast]:
        if not self._coefs:
            raise RuntimeError("har: call fit() before forecast()")
        r = self._validate_window(returns)
        v = r * r
        x_last = _har_regressors(v)[-1]
        floor = 0.05 * float(np.mean(v))  # guard against negative OLS forecasts
        out: list[RiskForecast] = []
        for h in horizons:
            if h not in self._coefs:
                raise KeyError(f"har: horizon {h} was not fitted")
            agg = float(max(x_last @ self._coefs[h], floor * h))
            sigma = float(np.sqrt(agg))
            var_d, es_d = {}, {}
            for a in alphas:
                var_d[a], es_d[a] = var_es_normal(sigma, a)
            out.append(RiskForecast(horizon=h, variance=agg, var=var_d, es=es_d))
        return out
