"""Walk-forward evaluation engine.

Protocol (frozen in docs/preregistration.md):

- rolling estimation window of ``window`` trading days;
- parameters re-estimated every ``refit_every`` days; between refits the
  model's ``forecast`` re-filters its state over the current window with
  frozen parameters (so conditional variance still updates daily);
- at each origin t the model sees returns up to and including t, and
  forecasts the h-day aggregate return over (t, t+h].

Leakage guard: the engine slices ``returns.iloc[i - window + 1 : i + 1]``
and nothing else — a unit test asserts that corrupting all data after an
origin leaves that origin's forecasts bit-identical.

Output is a tidy DataFrame, one row per
(origin, model, horizon, alpha): columns ``sigma2`` (h-day aggregate
variance forecast), ``var``, ``es`` (positive loss magnitudes), plus the
realized h-day return ``realized`` aligned by the same convention as
``aggregate_returns`` (NaN when the future window is incomplete). Realized
values are attached *after* forecasting, purely for evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tsfm_risk.data.returns import aggregate_returns
from tsfm_risk.models.base import RiskModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardConfig:
    window: int = 1000
    refit_every: int = 21
    horizons: tuple[int, ...] = (1, 5, 20)
    alphas: tuple[float, ...] = (0.01, 0.025, 0.05)
    start: pd.Timestamp | None = None  # first allowed origin (e.g. OOS start)

    def __post_init__(self) -> None:
        if self.window < 250:
            raise ValueError("window must be >= 250 trading days")
        if self.refit_every < 1:
            raise ValueError("refit_every must be >= 1")
        if any(h < 1 for h in self.horizons):
            raise ValueError("horizons must be >= 1")
        if any(not 0 < a < 0.5 for a in self.alphas):
            raise ValueError("alphas must be in (0, 0.5)")


@dataclass
class WalkForwardResult:
    forecasts: pd.DataFrame
    n_refits: int = 0
    failures: list[tuple[pd.Timestamp, str]] = field(default_factory=list)


def run_walkforward(
    model: RiskModel,
    returns: pd.Series,
    config: WalkForwardConfig,
) -> WalkForwardResult:
    """Run one model over one return series."""
    r = returns.dropna().astype(float)
    if r.index.has_duplicates or not r.index.is_monotonic_increasing:
        raise ValueError("returns index must be unique and sorted")
    values = r.to_numpy()
    dates = r.index

    realized = {h: aggregate_returns(r, h) for h in config.horizons}

    rows: list[dict] = []
    failures: list[tuple[pd.Timestamp, str]] = []
    n_refits = 0
    fitted = False

    first_i = config.window - 1
    if config.start is not None:
        first_i = max(first_i, int(dates.searchsorted(config.start)))

    for k, i in enumerate(range(first_i, len(values))):
        origin = dates[i]
        window = values[i - config.window + 1 : i + 1]
        try:
            if k % config.refit_every == 0:
                model.fit(window)
                n_refits += 1
                fitted = True
            if not fitted:  # first fit failed earlier; retry now
                model.fit(window)
                n_refits += 1
                fitted = True
            forecasts = model.forecast(window, config.horizons, config.alphas)
        except Exception as exc:  # noqa: BLE001 - one bad origin must not sink the run
            failures.append((origin, repr(exc)))
            logger.warning("%s failed at %s: %r", model.name, origin, exc)
            continue

        for f in forecasts:
            realized_h = realized[f.horizon].get(origin, np.nan)
            for a in config.alphas:
                rows.append(
                    {
                        "origin": origin,
                        "model": model.name,
                        "horizon": f.horizon,
                        "alpha": a,
                        "sigma2": f.variance,
                        "var": f.var[a],
                        "es": f.es[a],
                        "realized": realized_h,
                    }
                )

    df = pd.DataFrame(rows)
    if failures:
        logger.warning(
            "%s: %d/%d origins failed", model.name, len(failures), len(values) - first_i
        )
    return WalkForwardResult(forecasts=df, n_refits=n_refits, failures=failures)


def run_many(
    models: list[RiskModel],
    series: dict[str, pd.Series],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """All models over all series; adds a ``series`` column."""
    parts: list[pd.DataFrame] = []
    for ticker, r in series.items():
        for model in models:
            res = run_walkforward(model, r, config)
            if res.forecasts.empty:
                logger.warning("no forecasts for %s on %s", model.name, ticker)
                continue
            df = res.forecasts
            df.insert(0, "series", ticker)
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
