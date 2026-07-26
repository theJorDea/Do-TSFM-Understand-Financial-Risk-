"""Price loading with a local parquet cache.

Network access happens only on cache miss, so experiments are reproducible
offline once ``data/cache/`` is populated. The cache is content-addressed by
ticker and date range; delete a file to force a re-download.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import pandas as pd

from tsfm_risk.config import UniverseConfig

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


def _cache_path(ticker: str, start: dt.date, end: dt.date, cache_dir: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    return cache_dir / f"{safe}_{start}_{end}.parquet"


def load_prices(
    ticker: str,
    start: dt.date,
    end: dt.date,
    cache_dir: Path = CACHE_DIR,
) -> pd.Series:
    """Daily adjusted close for `ticker`, as a date-indexed float series."""
    path = _cache_path(ticker, start, end, cache_dir)
    if path.exists():
        df = pd.read_parquet(path)
    else:
        import yfinance as yf

        logger.info("cache miss, downloading %s", ticker)
        df = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False
        )
        if df is None or df.empty:
            raise RuntimeError(f"no data returned for {ticker}")
        df = df[["Close"]].rename(columns={"Close": "adj_close"})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    s = df["adj_close"].dropna().astype(float)
    s.name = ticker
    return s


def load_universe_prices(
    universe: UniverseConfig, cache_dir: Path = CACHE_DIR
) -> dict[str, pd.Series]:
    """Load every series in the universe; returns {ticker: prices}.

    Failures are logged and skipped (recorded downstream in the data card),
    so one delisted ticker does not sink a full run.
    """
    out: dict[str, pd.Series] = {}
    for asset_class, spec in universe.all_series():
        try:
            out[spec.ticker] = load_prices(spec.ticker, universe.start, universe.end, cache_dir)
        except Exception:
            logger.exception("failed to load %s (%s)", spec.ticker, asset_class)
    return out
