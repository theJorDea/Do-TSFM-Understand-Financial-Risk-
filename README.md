# Do Time-Series Foundation Models Understand Financial Risk?

**A rigorous, regulation-grade evaluation of zero-shot time-series foundation models for volatility forecasting and tail-risk estimation.**

> 📄 Paper: *work in progress* (target: arXiv preprint → NeurIPS workshop → ACM ICAIF)
> 🔬 Status: **Phase 1 — data & baselines** (see [PLAN.md](PLAN.md) for the full roadmap)

---

## TL;DR

Time-series foundation models (TSFMs) — Chronos, TimesFM, Moirai, Lag-Llama — report strong zero-shot results on standard forecasting benchmarks. But financial risk management does not care about point forecasts: it cares about the **tails of the return distribution**. Whether TSFMs can price tail risk at all is essentially untested.

We evaluate TSFMs head-to-head against 40 years of econometric wisdom (GARCH-family models, HAR-RV, filtered historical simulation) on the two tasks that actually matter to a risk desk:

1. **Volatility forecasting** — evaluated with robust proxies and QLIKE loss;
2. **Value-at-Risk / Expected Shortfall estimation** — evaluated with the full regulatory backtesting toolkit *and* modern joint elicitable scoring rules.

Every comparison is backed by formal statistical tests (Diebold–Mariano, Model Confidence Set), split by market regime (calm vs. crisis), and fully reproducible from config files.

**We commit to publishing the outcome whatever it is.** "Foundation models fail at tail risk" and "foundation models quietly obsolete GARCH" are both publishable findings — what matters is that the evaluation is airtight.

---

## Research questions

| # | Question |
|---|----------|
| **RQ1** | Do zero-shot TSFMs forecast conditional volatility better than econometric baselines (GARCH family, HAR-RV) at horizons of 1–20 trading days? |
| **RQ2** | Do VaR/ES estimates derived from TSFM predictive distributions pass regulatory-grade backtests (Kupiec, Christoffersen, dynamic quantile, Acerbi–Székely), and how do they score under the joint FZ0 loss? |
| **RQ3** | How does the ranking change across **market regimes** (calm vs. crisis windows), **asset classes** (equities, FX, crypto, commodities), and **forecast horizons**? |
| **RQ4** | *What* do TSFMs actually learn about return distributions? Do their predictive quantiles exhibit volatility clustering, leverage effects, and fat tails — or do they regress to unconditional shapes? |

RQ4 is the differentiator: beyond a leaderboard, we run **distributional diagnostics** on TSFM outputs (implied kurtosis, quantile reactivity after shocks, PIT calibration) to explain *why* the models win or lose.

---

## Model universe

| Family | Model | Type | Role |
|---|---|---|---|
| Non-parametric | Historical simulation (rolling window) | quantile | baseline |
| Non-parametric | Filtered historical simulation (EWMA-scaled) | quantile | strong classical baseline |
| Econometric | EWMA / RiskMetrics (λ = 0.94) | vol | baseline |
| Econometric | GARCH(1,1) — Normal & Student-t | vol + parametric quantile | canonical baseline |
| Econometric | GJR-GARCH(1,1)-t | vol + parametric quantile | leverage effect |
| Econometric | HAR-RV (where realized vol available) | vol | high bar for vol forecasting |
| Neural (supervised) | LSTM trained with QLIKE | vol | trained-from-scratch DL reference |
| **TSFM** | Chronos-Bolt (Amazon) | quantile forecasts | zero-shot & fine-tuned |
| **TSFM** | TimesFM (Google) | point + quantile | zero-shot |
| **TSFM** | Moirai (Salesforce) | full predictive distribution | zero-shot |
| **TSFM** | Lag-Llama | probabilistic | zero-shot & fine-tuned |
| Optional | TimeGPT (Nixtla), Toto (Datadog) | API / open | robustness of conclusions |

Two pipelines map any model to risk estimates, so the comparison is apples-to-apples:

- **Direct-quantile path**: predictive quantiles → VaR; tail-averaged quantiles → ES.
- **Vol-path**: volatility forecast → VaR/ES via a location-scale model with Student-t innovations (same recipe for GARCH and TSFM vol forecasts).

Ablations: raw returns vs. scaled inputs, context length (256 / 512 / 1024 / max), zero-shot vs. light fine-tuning.

---

## Data

- **Assets:** equity indices (S&P 500, EURO STOXX 50, Nikkei 225), single stocks, FX majors, BTC/ETH, gold & oil — ~25–30 daily series.
- **Span:** 2000–2026 where available; strict walk-forward out-of-sample evaluation over ≥ 15 years, including the 2008 GFC, COVID-2020, the 2022 rate shock, and 2024–2026 volatility events.
- **Volatility proxies:** squared daily returns (with proxy-robust QLIKE, Patton 2011) everywhere; 5-min realized variance (Oxford-Man Realized Library; exchange data for crypto) on the subset where it exists.
- **Leakage control:** TSFMs are pretrained on public corpora that may include financial series. We treat this explicitly: contamination analysis is part of the protocol, and the crypto/recent-period subsets (post-cutoff for each model's training data) serve as a clean out-of-pretraining test.

---

## Evaluation protocol

**Volatility:** QLIKE (primary, robust to noisy proxies), MSE (secondary), per-series and pooled.

**Tail risk** at α ∈ {1%, 2.5%, 5%}:

- Unconditional coverage — Kupiec POF;
- Independence & conditional coverage — Christoffersen, Engle–Manganelli dynamic quantile (DQ) test;
- ES adequacy — Acerbi–Székely Z-statistics;
- **Joint VaR+ES scoring — Fissler–Ziegel (FZ0) loss**, the strictly consistent scoring rule for the (VaR, ES) pair. Rarely used in ML papers; standard in the econometrics literature. This is our headline comparison metric.

**Statistical rigor:** pairwise Diebold–Mariano tests with HAC standard errors; **Model Confidence Set** (Hansen–Lunde–Nason) over the full model universe per asset class × regime × horizon. No claim of superiority without a significant test behind it.

**Regime analysis:** all metrics reported separately for pre-registered calm/crisis windows, defined by dates *before* running any experiment (see `configs/regimes.yaml`).

---

## Repository structure

```
├── configs/            # experiment configs — every result maps to one config
├── src/tsfm_risk/
│   ├── data/           # loaders, returns, vol proxies, regime windows
│   ├── models/
│   │   ├── classical/  # historical, FHS, EWMA, GARCH family, HAR-RV
│   │   ├── neural/     # LSTM (QLIKE)
│   │   └── tsfm/       # unified wrappers: Chronos, TimesFM, Moirai, Lag-Llama
│   ├── risk/           # quantile→VaR/ES mapping, location-scale path
│   ├── evaluation/     # QLIKE, FZ0, backtests, DM, MCS
│   └── pipelines/      # walk-forward engine
├── scripts/            # run_experiment.py, make_tables.py, make_figures.py
├── notebooks/          # exploratory analysis & diagnostics (RQ4)
├── tests/              # unit tests incl. validation against arch/statsmodels
├── results/            # generated tables/figures (never edited by hand)
└── paper/              # LaTeX source
```

## Reproducibility

- Deterministic seeds, pinned dependencies (`uv.lock`), config-driven runs.
- One command reproduces any table/figure in the paper: `python scripts/run_experiment.py --config configs/<name>.yaml`.
- CI runs unit tests + a smoke-scale end-to-end experiment on every push.
- Classical estimators are cross-validated against `arch` and `statsmodels` in tests.

## Relation to prior work

Built on top of the author's [Riskforge](https://github.com/theJorDea/Riskforge) toolkit (GARCH/GJR MLE, VaR/ES backtests validated on synthetic data). Closest published work benchmarks TSFMs on generic point-forecast metrics or single markets; to our knowledge no study evaluates TSFMs with joint elicitable VaR/ES scoring, regulatory backtests, and regime-conditional MCS analysis across asset classes. See `paper/related_work.md` for the annotated bibliography.

## Citation

```bibtex
@misc{levin2026tsfmrisk,
  title  = {Do Time-Series Foundation Models Understand Financial Risk?},
  author = {Levin, Kirill},
  year   = {2026},
  note   = {Work in progress}
}
```

## License

MIT
