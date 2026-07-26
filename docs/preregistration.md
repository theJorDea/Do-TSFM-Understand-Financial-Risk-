# Pre-registered evaluation protocol

This document freezes the evaluation protocol **before any TSFM inference is run**.
The commit introducing this file (and `configs/regimes.yaml`, `configs/universe.yaml`)
is the pre-registration timestamp. Any later deviation must be documented in the paper.

## Primary endpoints

1. **Volatility forecasting:** QLIKE loss against the squared-return proxy
   (proxy-robust; Patton 2011). MSE is secondary/descriptive.
2. **Tail risk:** FZ0 joint scoring loss for the (VaR, ES) pair
   (Fissler–Ziegel strictly consistent scoring), at α ∈ {1%, 2.5%, 5%}.

Backtests (Kupiec POF, Christoffersen independence & conditional coverage,
Engle–Manganelli DQ with 4 lags, Acerbi–Székely Z1/Z2) are reported as
adequacy diagnostics, not ranking criteria.

## Forecast targets and horizons

- Daily log-returns of the series in `configs/universe.yaml`.
- Horizons: 1, 5, and 20 trading days (multi-day = aggregated return / scaled vol).
- VaR and ES are stated as positive loss magnitudes (Riskforge convention).

## Out-of-sample protocol

- Walk-forward evaluation. Estimation window: 1000 trading days (rolling).
- Refit frequency: every 21 trading days for classical/neural models.
  TSFMs are zero-shot: context window supplied at each forecast date, no refit.
- Out-of-sample period: 2010-01-01 through 2026-06-30 (series permitting);
  series with shorter history start when 1000 in-sample days are available.
- No information dated after the forecast origin may enter any input
  (prices, fitted parameters, scalers — everything).

## Model comparison rules

- Pairwise: Diebold–Mariano tests on loss differentials with HAC (Newey–West)
  standard errors, horizon-adjusted lag truncation.
- Universe-wide: Model Confidence Set (Hansen–Lunde–Nason) at confidence 90%,
  run per {asset class × regime × horizon × α} cell and pooled.
- Significance threshold 5% for DM; no superiority claim without it.

## Regimes

- Crisis dates are the fixed windows in `configs/regimes.yaml`; all other
  out-of-sample dates are calm. Windows chosen by documented market events,
  before running any experiment.

## Contamination policy

For each TSFM we record its declared pretraining data cutoff and corpus notes.
Analysis on the subset of out-of-sample dates strictly after the latest
pretraining cutoff (and on crypto series where excluded from pretraining) is
reported alongside full-sample results.

## Amendments

Recorded in `docs/amendments.md`. Amendment 1 (2026-07-26) revises the
contamination design, makes the vol-path the primary output mapping, and extends
the model roster — all before any inference was run.
