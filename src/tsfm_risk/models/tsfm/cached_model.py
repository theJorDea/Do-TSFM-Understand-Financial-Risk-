"""Turn cached TSFM quantile forecasts into the project's forecast table.

The classical baselines produce a tidy frame with one row per
(series, origin, model, horizon, alpha). This module produces exactly the same
frame from a TSFM cache, so `scripts/make_tables.py` treats both families
identically and no comparison code needs to know which is which.

Innovation law for the vol-path
-------------------------------
The vol-path needs a distribution to attach to the forecast scale. It is
estimated the same way for every model, from that model's own history and using
only information available at the origin: standardized residuals
``z_t = r_t / sigma_t`` are accumulated as the walk-forward proceeds, and the
Student-t degrees of freedom are re-fitted by MLE every ``refit_every`` origins
on a trailing window. Before ``min_residuals`` residuals have accumulated the
path falls back to normal innovations — those early origins are the warm-up and
are reported as such rather than silently mixed in.

This mirrors what GARCH-t does (it estimates nu jointly with the variance
recursion) while keeping the estimate strictly out-of-sample.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_risk.data.returns import aggregate_returns
from tsfm_risk.models.tsfm.adapters import (
    direct_quantile_forecast,
    fit_student_t_nu,
    scale_from_quantiles,
    vol_path_forecast,
)
from tsfm_risk.models.tsfm.cache import CACHE_ROOT, cache_path, load_cached, to_cube

logger = logging.getLogger(__name__)


def forecasts_from_cache(
    model_name: str,
    series: str,
    returns: pd.Series,
    *,
    horizons: tuple[int, ...] = (1, 5, 20),
    alphas: tuple[float, ...] = (0.01, 0.025, 0.05),
    cached_horizon: int = 20,
    context: int = 1024,
    root: Path = CACHE_ROOT,
    refit_every: int = 21,
    nu_window: int = 1000,
    min_residuals: int = 250,
    path_name: str = "vol",
) -> pd.DataFrame:
    """Build the standard forecast frame for one model on one series.

    ``path_name`` selects the mapping: ``"vol"`` for the primary vol-path,
    ``"direct"`` for the direct-quantile path (h=1 only, and only when the
    model's grid reaches the requested alphas).
    """
    path = cache_path(model_name, series, cached_horizon, context, root=root)
    df = load_cached(path)
    if df is None:
        raise FileNotFoundError(f"no cached forecasts at {path}")

    origins, levels, cube = to_cube(df)
    if max(horizons) > cube.shape[1]:
        raise ValueError(
            f"cache holds {cube.shape[1]} forecast days, need {max(horizons)}"
        )

    realized = {h: aggregate_returns(returns, h) for h in horizons}
    r_lookup = returns.reindex(origins)

    # per-origin one-day scale, used both for VaR and for the residual series
    sigma1 = np.array([scale_from_quantiles(levels, cube[i, 0, :]) for i in range(len(origins))])

    rows: list[dict] = []
    residuals: list[float] = []
    nu: float | None = None
    n_warmup = 0

    for i, origin in enumerate(origins):
        # A residual becomes observable only when the forecast it tests has been
        # realized. sigma1[i-1] is the scale forecast issued at the previous
        # origin for the following day, and that day is the current origin, so
        # its return is exactly the realization. Pairing r_t with sigma1[t]
        # instead would divide today's return by a forecast of tomorrow — a
        # one-day look-ahead, and the reason this is done here and not later.
        if i > 0:
            r_now = r_lookup.iloc[i]
            if np.isfinite(r_now) and sigma1[i - 1] > 0:
                residuals.append(float(r_now / sigma1[i - 1]))

        if path_name == "vol":
            if i % refit_every == 0 and len(residuals) >= min_residuals:
                try:
                    nu = fit_student_t_nu(np.asarray(residuals[-nu_window:]))
                except ValueError:
                    nu = None
            if nu is None:
                n_warmup += 1
            fcs = vol_path_forecast(levels, cube[i], horizons, alphas, nu=nu)
        else:
            fcs = [direct_quantile_forecast(levels, cube[i, 0, :], alphas)]

        for f in fcs:
            realized_h = realized[f.horizon].get(origin, np.nan)
            for a in alphas:
                rows.append(
                    {
                        "series": series,
                        "origin": origin,
                        "model": f"{model_name}-{path_name}",
                        "horizon": f.horizon,
                        "alpha": a,
                        "sigma2": f.variance,
                        "var": f.var[a],
                        "es": f.es[a],
                        "realized": realized_h,
                    }
                )

    if n_warmup:
        logger.info(
            "%s/%s: %d of %d origins used normal innovations (nu warm-up)",
            model_name, series, n_warmup, len(origins),
        )
    return pd.DataFrame(rows)


def build_all(
    model_name: str,
    series_returns: dict[str, pd.Series],
    **kwargs,
) -> pd.DataFrame:
    """Concatenate the forecast frames of every series that has a cache."""
    parts = []
    for ticker, r in series_returns.items():
        try:
            parts.append(forecasts_from_cache(model_name, ticker, r, **kwargs))
        except FileNotFoundError:
            logger.warning("no cache for %s/%s, skipping", model_name, ticker)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
