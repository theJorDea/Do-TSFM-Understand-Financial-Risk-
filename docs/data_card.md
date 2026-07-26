# Data card

Source: Yahoo Finance via `yfinance`, daily adjusted closes (`auto_adjust=True`),
cached to parquet on first download so every experiment reruns offline.
Universe defined in `configs/universe.yaml`; returns are daily log-returns.

## Cleaning rules

| Rule | Implementation | Effect on the universe |
|---|---|---|
| Non-positive prices dropped | `log_returns` | none observed |
| \|log-return\| > 1.0 (>170% move) dropped | `clean_returns` | none observed |
| Bad-tick spike/reversal pairs flagged | `find_spike_reversals` | **5 dates, all FX** (see below) |

## Bad-tick audit

A single mistyped vendor close makes the series jump and snap back the next day.
Detection requires **both** an implausible size relative to *local* volatility
(>12× the median absolute return in a centred 121-day window) **and** a
near-exact reversal (≥80% retraced the following day).

Both conditions are necessary. Retracement alone flags genuine crash whipsaws —
the S&P 500 fell 10.0% on 2020-03-12 and rebounded 8.9% on 2020-03-13, an 89%
retracement identical in shape to the artifacts below. Size alone flags real
one-day collapses — Apple's −52% on 2000-09-29 (a genuine profit warning) never
reverts. Only the conjunction isolates vendor errors: an earlier retracement-only
rule flagged 304 days across the universe, nearly all of them real market moves.

**Flagged dates (all in 2008, all FX, all on the 8th of a month):**

| Series | Date | Printed log-return | Note |
|---|---|---|---|
| EURUSD=X | 2008-01-08 | +5.9% | |
| EURUSD=X | 2008-02-08 | +7.3% | |
| EURUSD=X | 2008-10-08 | +9.6% | |
| EURUSD=X | 2008-12-08 | +16.0% | close printed 1.4918 between 1.2717 and 1.2926 |
| USDJPY=X | 2008-12-08 | +16.3% | |

The clustering on the 8th of the month across two independent currency pairs is
itself evidence of a systematic vendor glitch rather than market events; major
FX pairs do not move 16% in a day and fully reverse the next.

**Impact if left uncorrected.** These five days inflate the sample excess
kurtosis of EUR/USD to ≈100 — implausible for a major currency pair and enough
to distort tail-risk estimates, GARCH parameters, and every VaR backtest on the
FX block.

**Handling.** Flagged pairs (spike day and its reversal) are removed before
estimation; removals are enumerated above rather than applied silently. Equity,
index, commodity, crypto and bond-ETF series are unaffected.

## Series inventory

Twenty-one series across six asset classes; per-series spans are printed by
`scripts/run_baselines.py` at the start of each run and recorded in
`results/full_run.log`. Series with fewer than `window + 250` observations are
skipped, and the skip is logged.

## Known limitations

- Yahoo FX quotes are indicative, not traded prices; the FX block is
  correspondingly less reliable than the equity block. Retained because
  cross-asset breadth matters for the study's claims, but conclusions specific
  to FX are reported with this caveat.
- Crypto trades 7 days a week while every other series does not; calendars are
  aligned per series, not pooled.
- Survivorship: single stocks are current constituents, so the equity block is
  mildly survivorship-biased. It is used for volatility/tail dynamics, not for
  return-level claims, which limits the damage.
