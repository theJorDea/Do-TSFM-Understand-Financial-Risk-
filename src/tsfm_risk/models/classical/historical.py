"""Historical and filtered historical simulation.

HistoricalSimulation
    Empirical VaR/ES from the raw return window. For horizon h > 1 the
    empirical distribution is built from *overlapping* h-day return sums
    within the window (standard practice; the induced serial dependence
    affects backtest power, not validity of the point estimate, and is
    handled by HAC/DQ tests downstream).

FilteredHistoricalSimulation (Barone-Adesi, Giannopoulos & Vosper 1999;
Hull & White 1998)
    Devolatilize the window with an EWMA filter, z_t = r_t / s_t, then
    re-inflate by the current volatility forecast. For h = 1 this is the
    empirical quantile of z scaled by s_{t+1}. For h > 1 we bootstrap
    standardized residuals and propagate the EWMA recursion *inside each
    simulated path*, so volatility reacts to the simulated shocks (proper
    multi-day FHS, not sqrt-h scaling). Bootstrap is seeded — forecasts are
    deterministic given the window.
"""

from __future__ import annotations

import numpy as np

from tsfm_risk.models.base import RiskForecast, RiskModel
from tsfm_risk.models.classical.ewma import ewma_variance_path
from tsfm_risk.risk.measures import var_es_empirical


def overlapping_sums(r: np.ndarray, h: int) -> np.ndarray:
    if h == 1:
        return r
    c = np.concatenate([[0.0], np.cumsum(r)])
    return c[h:] - c[:-h]


class HistoricalSimulation(RiskModel):
    name = "historical"

    def fit(self, returns: np.ndarray) -> None:
        """Non-parametric; nothing to estimate."""

    def forecast(
        self,
        returns: np.ndarray,
        horizons: tuple[int, ...],
        alphas: tuple[float, ...],
    ) -> list[RiskForecast]:
        r = self._validate_window(returns)
        out: list[RiskForecast] = []
        for h in horizons:
            sample = overlapping_sums(r, h)
            var_d, es_d = {}, {}
            for a in alphas:
                var_d[a], es_d[a] = var_es_empirical(sample, a)
            out.append(
                RiskForecast(horizon=h, variance=float(np.var(sample)), var=var_d, es=es_d)
            )
        return out


class FilteredHistoricalSimulation(RiskModel):
    name = "fhs-ewma"

    def __init__(self, lam: float = 0.94, n_paths: int = 20_000, seed: int = 1234):
        self.lam = lam
        self.n_paths = n_paths
        self.seed = seed

    def fit(self, returns: np.ndarray) -> None:
        """Non-parametric given lambda; nothing to estimate."""

    def forecast(
        self,
        returns: np.ndarray,
        horizons: tuple[int, ...],
        alphas: tuple[float, ...],
    ) -> list[RiskForecast]:
        r = self._validate_window(returns)
        s2_path, s2_next = ewma_variance_path(r, self.lam)
        z = r / np.sqrt(s2_path)

        out: list[RiskForecast] = []
        for h in horizons:
            if h == 1:
                sample = float(np.sqrt(s2_next)) * z
            else:
                sample = self._simulate_paths(z, s2_next, h)
            var_d, es_d = {}, {}
            for a in alphas:
                var_d[a], es_d[a] = var_es_empirical(sample, a)
            out.append(
                RiskForecast(horizon=h, variance=float(np.var(sample)), var=var_d, es=es_d)
            )
        return out

    def _simulate_paths(self, z: np.ndarray, s2_start: float, h: int) -> np.ndarray:
        """Bootstrap h-day returns with EWMA volatility updated along each path."""
        rng = np.random.default_rng(self.seed)  # fixed seed: deterministic forecasts
        draws = rng.choice(z, size=(self.n_paths, h), replace=True)
        s2 = np.full(self.n_paths, s2_start)
        total = np.zeros(self.n_paths)
        for k in range(h):
            step = np.sqrt(s2) * draws[:, k]
            total += step
            s2 = self.lam * s2 + (1.0 - self.lam) * step * step
        return total
