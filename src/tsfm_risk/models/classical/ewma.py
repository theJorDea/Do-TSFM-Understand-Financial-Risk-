"""RiskMetrics EWMA volatility (J.P. Morgan, 1996).

    s2_{t+1} = lambda * s2_t + (1 - lambda) * r_t^2,   lambda = 0.94 (daily)

EWMA is an IGARCH(1,1) special case, so the variance forecast is flat in the
horizon: the h-day aggregate variance is h * s2_{t+1}. Tail measures use
normal innovations — the canonical RiskMetrics recipe, kept deliberately
untouched as the industry-baseline reference point.
"""

from __future__ import annotations

import numpy as np

from tsfm_risk.models.base import RiskForecast, RiskModel
from tsfm_risk.risk.measures import var_es_normal


def ewma_variance_path(r: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """``s2[t]`` conditions on info through t-1; also returns s2_{T+1}."""
    n = r.size
    s2 = np.empty(n)
    s2_prev = float(np.var(r))
    for t in range(n):
        s2[t] = s2_prev
        s2_prev = lam * s2_prev + (1.0 - lam) * r[t] * r[t]
    return s2, s2_prev


class Ewma(RiskModel):
    name = "ewma-rm"

    def __init__(self, lam: float = 0.94):
        if not 0.0 < lam < 1.0:
            raise ValueError("lambda must be in (0, 1)")
        self.lam = lam

    def fit(self, returns: np.ndarray) -> None:
        """No free parameters (lambda is fixed by convention)."""

    def forecast(
        self,
        returns: np.ndarray,
        horizons: tuple[int, ...],
        alphas: tuple[float, ...],
    ) -> list[RiskForecast]:
        r = self._validate_window(returns)
        _, s2_next = ewma_variance_path(r, self.lam)
        out: list[RiskForecast] = []
        for h in horizons:
            agg = h * s2_next
            sigma = float(np.sqrt(agg))
            var_d, es_d = {}, {}
            for a in alphas:
                var_d[a], es_d[a] = var_es_normal(sigma, a)
            out.append(RiskForecast(horizon=h, variance=agg, var=var_d, es=es_d))
        return out
