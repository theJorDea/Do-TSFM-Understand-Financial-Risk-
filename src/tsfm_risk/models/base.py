"""Common interface every risk model implements.

The walk-forward engine drives models through exactly two calls:

- ``fit(returns)`` — (re-)estimate parameters on the estimation window.
  Called every ``refit_every`` days. May be a no-op (historical simulation,
  zero-shot TSFMs).
- ``forecast(returns, horizons, alphas)`` — produce h-day-ahead risk
  forecasts conditional on the window ending at the forecast origin.
  Called daily; must be cheap. Parametric models re-filter their variance
  recursion over ``returns`` with the parameters frozen at the last ``fit``.

``returns`` are daily log-returns in decimal units, oldest first, with the
last element being the return of the origin day itself. Nothing dated after
the origin may ever be passed in — the engine enforces this and a unit test
asserts forecasts are invariant to future data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RiskForecast:
    """h-day-ahead forecast issued at one origin.

    ``variance`` is the forecast of the h-day *aggregate* return variance;
    ``var``/``es`` map tail level alpha -> positive loss magnitude.
    """

    horizon: int
    variance: float
    var: dict[float, float] = field(default_factory=dict)
    es: dict[float, float] = field(default_factory=dict)


class RiskModel(ABC):
    """Base class; subclasses set ``name`` used in results tables."""

    name: str = "unnamed"

    @abstractmethod
    def fit(self, returns: np.ndarray) -> None:
        """Estimate parameters on the estimation window."""

    @abstractmethod
    def forecast(
        self,
        returns: np.ndarray,
        horizons: tuple[int, ...],
        alphas: tuple[float, ...],
    ) -> list[RiskForecast]:
        """Risk forecasts for each horizon, conditional on window end."""

    @staticmethod
    def _validate_window(returns: np.ndarray, min_obs: int = 250) -> np.ndarray:
        x = np.asarray(returns, dtype=float)
        if x.ndim != 1:
            raise ValueError("returns must be 1-D")
        if x.size < min_obs:
            raise ValueError(f"need >= {min_obs} observations, got {x.size}")
        if not np.all(np.isfinite(x)):
            raise ValueError("returns contain non-finite values")
        return x
