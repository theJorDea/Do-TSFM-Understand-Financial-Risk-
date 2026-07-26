"""Run classical baselines over the asset universe (or a ticker subset).

Usage:
    uv run python scripts/run_baselines.py                    # full universe
    uv run python scripts/run_baselines.py --tickers SPY BTC-USD
    uv run python scripts/run_baselines.py --out results/baselines.parquet

Produces a tidy parquet of walk-forward forecasts (see
pipelines/walkforward.py for the schema) and prints a per-model summary of
1-day 5% VaR breach rates as a first-glance sanity check.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tsfm_risk.config import load_universe
from tsfm_risk.data.loaders import load_universe_prices
from tsfm_risk.data.returns import clean_returns, log_returns
from tsfm_risk.models.classical import (
    Ewma,
    FilteredHistoricalSimulation,
    Garch,
    Har,
    HistoricalSimulation,
)
from tsfm_risk.pipelines.walkforward import WalkForwardConfig, run_many

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_models(horizons: tuple[int, ...]):
    return [
        HistoricalSimulation(),
        FilteredHistoricalSimulation(),
        Ewma(),
        Garch(dist="normal"),
        Garch(dist="t"),
        Garch(dist="t", leverage=True),
        Har(horizons=horizons),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--out", default="results/baselines.parquet")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--refit-every", type=int, default=21)
    args = ap.parse_args()

    universe = load_universe()
    prices = load_universe_prices(universe)
    if args.tickers:
        missing = set(args.tickers) - set(prices)
        if missing:
            raise SystemExit(f"tickers not in universe/loadable: {sorted(missing)}")
        prices = {t: prices[t] for t in args.tickers}

    series = {t: clean_returns(log_returns(p)) for t, p in prices.items()}
    for t, r in series.items():
        print(f"{t}: {len(r)} returns, {r.index[0].date()} .. {r.index[-1].date()}")

    cfg = WalkForwardConfig(window=args.window, refit_every=args.refit_every)
    df = run_many(build_models(cfg.horizons), series, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nwrote {len(df):,} forecast rows -> {out}")

    sel = df[(df.horizon == 1) & (df.alpha == 0.05)].dropna(subset=["realized"])
    breach = (
        sel.assign(breach=sel.realized < -sel["var"])
        .groupby("model")["breach"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "breach_rate_5pct", "count": "n_days"})
    )
    print("\n1-day 5% VaR breach rates (nominal = 0.050):")
    print(breach.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
