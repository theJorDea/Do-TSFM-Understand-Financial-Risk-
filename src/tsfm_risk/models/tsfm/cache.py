"""Parquet cache of TSFM quantile forecasts.

Inference is the expensive half of the study and the only part that needs a
model checkpoint; evaluation is cheap and gets re-run constantly as tables,
regimes and ablations change. Separating them means a model is executed exactly
once per (config, series, origin) and every later analysis reads numbers.

Layout — one file per (model, series, horizon, context length)::

    forecasts/{model}/{series}__h{H}__c{C}.parquet

with columns ``origin``, ``day``, ``level``, ``quantile``. ``day`` runs from 1 to
H: a single run at H=20 therefore serves the 1-, 5- and 20-day horizons at once,
because the vol-path aggregates per-day variances. Re-running inference per
horizon would trebled the cost for no new information — Chronos-Bolt emits all
days in one forward pass anyway.

Long format rather than one column per level because the quantile grid differs
between models (Chronos-Bolt stops at 0.1/0.9, Chronos-2 reaches 0.01) and a
shared wide schema would be mostly nulls.

The cache is append-only and resumable: a run that dies halfway leaves valid
partial output, and re-running skips origins already present. This matters
because a full sweep is ~100k origins per model.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("forecasts")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def cache_path(
    model: str,
    series: str,
    horizon: int,
    context: int,
    root: Path = CACHE_ROOT,
) -> Path:
    return Path(root) / _safe(model) / f"{_safe(series)}__h{horizon}__c{context}.parquet"


def load_cached(path: Path) -> pd.DataFrame | None:
    """Existing forecasts, or None if nothing cached yet."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["origin"] = pd.to_datetime(df["origin"])
    return df


def cached_origins(path: Path) -> set[pd.Timestamp]:
    df = load_cached(path)
    return set() if df is None else set(df["origin"].unique())


def append_forecasts(
    path: Path,
    origins: list[pd.Timestamp],
    levels: tuple[float, ...],
    quantiles: np.ndarray,
) -> None:
    """Append a batch of forecasts, de-duplicating against what is stored.

    ``quantiles`` has shape ``(n_origins, horizon, n_levels)``; every forecast
    day is written, tagged by ``day`` = 1..horizon.
    """
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 3:
        raise ValueError("quantiles must have shape (n_origins, horizon, n_levels)")
    if q.shape[0] != len(origins):
        raise ValueError("origins and quantiles disagree on batch size")
    if q.shape[2] != len(levels):
        raise ValueError("quantiles and levels disagree on grid size")

    n_o, n_d, n_l = q.shape
    new = pd.DataFrame(
        {
            "origin": np.repeat(pd.DatetimeIndex(origins).values, n_d * n_l),
            "day": np.tile(np.repeat(np.arange(1, n_d + 1), n_l), n_o),
            "level": np.tile(np.asarray(levels, dtype=float), n_o * n_d),
            "quantile": q.reshape(-1),
        }
    )

    existing = load_cached(path)
    if existing is not None:
        new = new[~new["origin"].isin(set(existing["origin"].unique()))]
        if new.empty:
            return
        new = pd.concat([existing, new], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["origin", "day", "level"]).to_parquet(path, index=False)


def to_matrix(
    df: pd.DataFrame, day: int = 1
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Long cache table -> (origins, levels, quantiles[n_origins, n_levels])."""
    sub = df[df["day"] == day]
    if sub.empty:
        raise ValueError(f"no forecasts cached for day {day}")
    wide = sub.pivot_table(index="origin", columns="level", values="quantile").sort_index()
    return pd.DatetimeIndex(wide.index), wide.columns.to_numpy(), wide.to_numpy()


def to_cube(df: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Long cache table -> (origins, levels, quantiles[n_origins, n_days, n_levels]).

    This is what the vol-path consumes: it needs every forecast day to sum
    per-day variances over the horizon.
    """
    days = np.sort(df["day"].unique())
    origins, levels, first = to_matrix(df, day=int(days[0]))
    cube = np.empty((len(origins), len(days), len(levels)))
    cube[:, 0, :] = first
    for i, d in enumerate(days[1:], start=1):
        o, lv, mat = to_matrix(df, day=int(d))
        if len(o) != len(origins) or not np.array_equal(lv, levels):
            raise ValueError(f"day {d} has a different origin/level grid than day {days[0]}")
        cube[:, i, :] = mat
    return origins, levels, cube
