"""Common interface for time-series foundation models.

Every TSFM in the study is reduced to one operation:

    predict_quantiles(context, horizon, levels) -> array of shape (horizon, n_levels)

``context`` is the trailing window of daily log-returns ending at the forecast
origin (nothing after it), ``levels`` are probabilities in (0, 1) ascending, and
the returned quantiles are of the **return distribution at each future day**,
in the same units as the context.

Design notes
------------
* Zero-shot: there is no ``fit``. The model sees the series only through the
  context window, exactly as the walk-forward engine supplies it.
* Inference is the expensive part, so it is separated from evaluation: a
  runner writes quantile forecasts once into a parquet cache
  (:mod:`tsfm_risk.models.tsfm.cache`), and the risk adapters read from it.
  Re-running the analysis never re-runs a model.
* Models that emit a parametric predictive distribution (Moirai) or samples
  (Lag-Llama) implement this interface by evaluating/empirically estimating
  their own quantiles, so all models are compared through one channel.

Contamination bookkeeping: every wrapper declares ``data_cutoff``, the date
after which the model provably could not have seen data (in practice the
checkpoint release date, a conservative upper bound). The cutoff-stratified
analysis required by ``docs/amendments.md`` reads it from here.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod

import numpy as np

# quantile grid requested from every model; the subset a model actually
# supports natively is declared per wrapper in ``native_levels``
DEFAULT_LEVELS: tuple[float, ...] = (
    0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,
)


class TSFMForecaster(ABC):
    """Zero-shot probabilistic forecaster over a context window."""

    name: str = "unnamed-tsfm"
    #: conservative upper bound on what the model could have seen in pretraining
    data_cutoff: dt.date | None = None
    #: quantile levels the model produces natively; others must be interpolated
    native_levels: tuple[float, ...] | None = None
    #: longest context the model accepts (None = unbounded)
    max_context: int | None = None

    @abstractmethod
    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        levels: tuple[float, ...] = DEFAULT_LEVELS,
    ) -> np.ndarray:
        """Predictive quantiles for each of the next ``horizon`` days.

        Returns an array of shape ``(horizon, len(levels))``, non-decreasing
        along the level axis.
        """

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _validate(context: np.ndarray, horizon: int, levels: tuple[float, ...]) -> np.ndarray:
        x = np.asarray(context, dtype=float)
        if x.ndim != 1:
            raise ValueError("context must be 1-D")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        lv = np.asarray(levels, dtype=float)
        if np.any(np.diff(lv) <= 0):
            raise ValueError("levels must be strictly increasing")
        if lv[0] <= 0 or lv[-1] >= 1:
            raise ValueError("levels must lie strictly inside (0, 1)")
        return x

    def check_native(self, levels: tuple[float, ...]) -> None:
        """Refuse levels the model was not trained to produce.

        This is not pedantry. Chronos-Bolt, asked for the 1% quantile, returns
        its 10% quantile *clamped* — a number that looks like a forecast, is not
        one, and would silently become a VaR estimate. Requests outside
        ``native_levels`` must fail loudly; the vol-path is how such models
        reach small alphas.
        """
        if self.native_levels is None:
            return
        native = np.asarray(self.native_levels)
        outside = [lv for lv in levels if not np.any(np.isclose(lv, native))]
        if outside:
            raise ValueError(
                f"{self.name} was trained on quantile levels {self.native_levels}; "
                f"refusing to fabricate {outside}. Use the vol-path for these alphas."
            )

    def levels_for_request(self, levels: tuple[float, ...]) -> tuple[float, ...]:
        """The subset of ``levels`` this model can actually produce."""
        if self.native_levels is None:
            return levels
        native = np.asarray(self.native_levels)
        return tuple(lv for lv in levels if np.any(np.isclose(lv, native)))

    @staticmethod
    def _enforce_monotone(q: np.ndarray) -> np.ndarray:
        """Repair quantile crossing (models can emit non-monotone grids).

        Crossing is a known artefact of independently-parameterised quantile
        heads; the standard fix is a cumulative maximum along the level axis.
        Any repair is counted so the paper can report how often it happened.
        """
        return np.maximum.accumulate(np.asarray(q, dtype=float), axis=-1)

    def truncate_context(self, context: np.ndarray) -> np.ndarray:
        """Trim to the model's maximum supported context (keeping the tail)."""
        if self.max_context is None or context.size <= self.max_context:
            return context
        return context[-self.max_context :]
