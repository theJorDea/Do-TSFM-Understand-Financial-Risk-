"""Run a TSFM over the walk-forward origins and cache its quantile forecasts.

    uv run python scripts/run_tsfm_inference.py --model amazon/chronos-bolt-base
    uv run python scripts/run_tsfm_inference.py --tickers "^GSPC" BTC-USD --context 512

Inference is separated from evaluation on purpose: this script is the only part
that needs a model checkpoint, and it writes numbers that every later analysis
reads. It is resumable — origins already in the cache are skipped, so a run that
is interrupted can simply be started again.

The forecast origins are exactly those the classical baselines used
(``WalkForwardConfig``), so the two families are compared on the same dates with
the same information set. Context is the trailing window ending at the origin;
nothing after it is ever passed to the model.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from tsfm_risk.config import load_universe
from tsfm_risk.data.loaders import load_universe_prices
from tsfm_risk.data.returns import clean_returns, drop_spike_reversals, log_returns
from tsfm_risk.models.tsfm.cache import append_forecasts, cache_path, cached_origins
from tsfm_risk.models.tsfm.chronos import ChronosForecaster
from tsfm_risk.pipelines.walkforward import WalkForwardConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tsfm-inference")


def build_forecaster(model_id: str, device: str, context: int, batch: int):
    if "chronos" in model_id:
        return ChronosForecaster(
            model_id=model_id, device=device, max_context=context, batch_size=batch
        )
    raise SystemExit(f"no wrapper registered for '{model_id}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-bolt-base")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window", type=int, default=1000,
                    help="estimation window of the baselines; sets the first origin")
    ap.add_argument("--out", default="forecasts")
    args = ap.parse_args()

    model = build_forecaster(args.model, args.device, args.context, args.batch)
    levels = model.levels_for_request(
        tuple(sorted({0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                      0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99}))
    )
    if not levels:
        raise SystemExit(f"{model.name}: no usable quantile levels")
    logger.info("model=%s levels=%s cutoff=%s", model.name, levels, model.data_cutoff)

    universe = load_universe()
    prices = load_universe_prices(universe)
    if args.tickers:
        missing = set(args.tickers) - set(prices)
        if missing:
            raise SystemExit(f"tickers not loadable: {sorted(missing)}")
        prices = {t: prices[t] for t in args.tickers}

    cfg = WalkForwardConfig(window=args.window)
    t_start = time.time()
    total_new = 0

    for ticker, px in prices.items():
        r = clean_returns(log_returns(px))
        r, spikes = drop_spike_reversals(r)
        if len(spikes):
            logger.info("%s: dropped %d bad-tick pairs", ticker, len(spikes))
        if len(r) < cfg.window + 250:
            logger.warning("skipping %s: only %d returns", ticker, len(r))
            continue

        path = cache_path(model.name, ticker, args.horizon, args.context, root=Path(args.out))
        done = cached_origins(path)

        values, dates = r.to_numpy(), r.index
        todo = [i for i in range(cfg.window - 1, len(values)) if dates[i] not in done]
        if not todo:
            print(f"{ticker}: уже посчитано ({len(done):,} origins)", flush=True)
            continue

        print(f"{ticker}: {len(todo):,} новых origins (в кэше {len(done):,})", flush=True)
        t0 = time.time()
        for start in range(0, len(todo), args.batch):
            idx = todo[start : start + args.batch]
            contexts = [values[max(0, i - args.context + 1) : i + 1] for i in idx]
            q = model.predict_batch(contexts, args.horizon, levels)
            append_forecasts(path, [dates[i] for i in idx], levels, q)
            total_new += len(idx)
            if start % (args.batch * 20) == 0 and start:
                rate = (start + len(idx)) / (time.time() - t0)
                left = (len(todo) - start) / max(rate, 1e-9)
                print(f"   {start + len(idx):,}/{len(todo):,}  "
                      f"{rate:.0f} прогнозов/с  осталось ~{left / 60:.1f} мин", flush=True)
        print(f"   готово за {(time.time() - t0) / 60:.1f} мин -> {path}", flush=True)

    elapsed = (time.time() - t_start) / 60
    print(f"\nвсего новых прогнозов: {total_new:,} за {elapsed:.1f} мин")
    if total_new:
        print(f"скорость: {total_new / (elapsed * 60):.0f} прогнозов/с")


if __name__ == "__main__":
    main()
