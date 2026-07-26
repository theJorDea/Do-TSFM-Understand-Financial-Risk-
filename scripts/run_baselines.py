"""Run classical baselines over the asset universe (or a ticker subset).

Usage:
    uv run python scripts/run_baselines.py                     # full universe
    uv run python scripts/run_baselines.py --tickers SPY BTC-USD
    uv run python scripts/run_baselines.py --jobs 10           # parallel

Each (series, model) pair is an independent walk-forward run, so the grid is
embarrassingly parallel across processes. Workers pin BLAS to one thread to
avoid oversubscription. Produces a tidy parquet of forecasts (schema in
pipelines/walkforward.py) and prints a per-model summary of 1-day 5% VaR
breach rates as a first-glance sanity check.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_baselines")


def build_models(horizons: tuple[int, ...]):
    from tsfm_risk.models.classical import (
        Ewma,
        FilteredHistoricalSimulation,
        Garch,
        Har,
        HistoricalSimulation,
    )

    return [
        HistoricalSimulation(),
        FilteredHistoricalSimulation(),
        Ewma(),
        Garch(dist="normal"),
        Garch(dist="t"),
        Garch(dist="t", leverage=True),
        Har(horizons=horizons),
    ]


def _worker_init() -> None:
    # one BLAS thread per worker: the grid parallelism is at process level
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _run_one(ticker: str, model_index: int, returns: pd.Series, window: int,
             refit_every: int) -> tuple[str, str, pd.DataFrame, int]:
    from tsfm_risk.pipelines.walkforward import WalkForwardConfig, run_walkforward

    cfg = WalkForwardConfig(window=window, refit_every=refit_every)
    model = build_models(cfg.horizons)[model_index]
    res = run_walkforward(model, returns, cfg)
    df = res.forecasts
    if not df.empty:
        df.insert(0, "series", ticker)
    return ticker, model.name, df, len(res.failures)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--out", default="results/baselines.parquet")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--refit-every", type=int, default=21)
    ap.add_argument("--jobs", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    args = ap.parse_args()

    from tsfm_risk.config import load_universe
    from tsfm_risk.data.loaders import load_universe_prices
    from tsfm_risk.data.returns import clean_returns, log_returns

    universe = load_universe()
    prices = load_universe_prices(universe)
    if args.tickers:
        missing = set(args.tickers) - set(prices)
        if missing:
            raise SystemExit(f"tickers not in universe/loadable: {sorted(missing)}")
        prices = {t: prices[t] for t in args.tickers}

    series = {}
    for t, p in prices.items():
        r = clean_returns(log_returns(p))
        if len(r) < args.window + 250:
            logger.warning("skipping %s: only %d returns", t, len(r))
            continue
        series[t] = r
        print(f"{t}: {len(r)} returns, {r.index[0].date()} .. {r.index[-1].date()}",
              flush=True)

    n_models = len(build_models((1, 5, 20)))
    tasks = [(t, m) for t in series for m in range(n_models)]
    print(f"\n{len(tasks)} (series, model) runs on {args.jobs} workers", flush=True)

    parts: list[pd.DataFrame] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_worker_init) as pool:
        futures = {
            pool.submit(_run_one, t, m, series[t], args.window, args.refit_every): (t, m)
            for t, m in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, model_name, df, n_fail = fut.result()
            parts.append(df)
            fail_note = f" ({n_fail} failed origins)" if n_fail else ""
            print(f"[{i}/{len(tasks)}] {ticker} x {model_name}: "
                  f"{len(df):,} rows{fail_note}  [{time.time() - t0:,.0f}s]", flush=True)

    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nwrote {len(df):,} forecast rows -> {out}", flush=True)

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
