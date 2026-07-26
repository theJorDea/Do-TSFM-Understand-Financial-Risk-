# Project Plan

Target outcome: a paper that survives hostile review — every claim backed by a statistical test, every design choice pre-registered or ablated, everything reproducible from one command.

**Publication path:** arXiv preprint (late Sept 2026) → NeurIPS 2026 workshop (time series / finance ML) → full version at ACM ICAIF 2027.

Assumed pace: ~10–15 h/week. Dates are targets, not promises; scope control notes at the bottom.

---

## Phase 0 — Infrastructure ✅ (completed 2026-07-26, ahead of schedule)

- [x] Python project scaffolding: `uv` + `pyproject.toml`, `ruff`, `pytest`, GitHub Actions CI.
- [x] Package skeleton `src/tsfm_risk/` per README structure.
- [x] Config system (YAML + pydantic validation): every experiment = one config file.
- [x] **Pre-registration commit**: `configs/regimes.yaml` with crisis/calm windows fixed by date, `docs/preregistration.md` with primary metrics (QLIKE, FZ0), α levels, and the decision rules — committed *before* any TSFM is run. This becomes a citable artifact ("we froze the protocol at commit X"). *Pre-registration commit: `01422e8`.*

**Done when:** CI is green on an empty-but-importable package; pre-registration committed. ✅

## Phase 1 — Data & classical baselines (weeks 1–3: Aug 3 – Aug 16) — nearly complete

- [x] Data loaders (yfinance + cached parquet), log-returns, 21-series universe across 6 asset classes.
- [ ] Data card documenting each series (source, span, missing-data policy, removed observations).
- [x] Volatility proxy: squared returns (proxy-robust QLIKE track).
- [ ] RV where available (Oxford-Man for pre-2022 indices, Binance 5-min for crypto) — supporting track.
- [x] GARCH(1,1)-N/-t, GJR-GARCH-t via custom MLE, validated against `arch` + synthetic recovery; Nelder-Mead fallback for flat likelihoods.
- [x] Historical VaR/ES, EWMA, filtered historical simulation (path-wise vol updating for h>1), HAR (direct projection per horizon).
- [x] Walk-forward engine: 1000-day window, refit every 21 days, daily variance filtering; leakage unit test (corrupted future ⇒ bit-identical forecasts).
- [x] Parallel grid runner (ProcessPoolExecutor over (series, model) pairs).
- [ ] First end-to-end run: classical models on full universe → baseline results table. *(running: `results/baselines_full.parquet`)*

**Done when:** baseline QLIKE/coverage tables regenerate from one command and GARCH results match `arch` within tolerance.

## Phase 2 — TSFM integration (weeks 3–6: Aug 17 – Sep 6)

- [ ] Unified `TSFMForecaster` interface: `predict_quantiles(context, horizon, q_levels)` — one wrapper per model (Chronos-Bolt, TimesFM, Moirai, Lag-Llama).
- [ ] Both risk pipelines: direct-quantile path and vol-path (location-scale, Student-t).
- [ ] Input-scaling & context-length ablation harness.
- [ ] Contamination note per model: pretraining corpus, data cutoff, which of our series/periods are plausibly in-pretraining vs. clean.
- [ ] GPU budget check: batch inference on Colab/Kaggle; cache all raw quantile forecasts to parquet so evaluation never re-runs inference.
- [ ] LSTM-QLIKE baseline (port from Riskforge, retrain per walk-forward split).
- [ ] Optional stretch: light fine-tuning for Chronos & Lag-Llama; TimeGPT via API.

**Done when:** every TSFM produces cached quantile forecasts for the full universe; smoke-test config runs end-to-end in CI (tiny series, CPU).

## Phase 3 — Evaluation & statistics (weeks 6–8: Sep 7 – Sep 20) — statistics built early

- [x] Losses: proxy-robust QLIKE, MSE; **FZ0 joint VaR-ES loss** (Fissler–Ziegel).
- [x] Backtests: Kupiec, Christoffersen (independence + conditional coverage), Engle–Manganelli DQ, Acerbi–Székely Z2 with implied-ν Student-t null and Monte-Carlo p-values.
- [x] Diebold–Mariano with HAC + Harvey–Leybourne–Newbold correction; **Model Confidence Set** (T_max, moving-block bootstrap), validated on synthetic data with a known best model.
- [x] Size/power of every test confirmed by Monte Carlo in the test suite.
- [ ] Results database (one tidy parquet: model × series × horizon × α × regime × metric) + `make_tables.py`, `make_figures.py`.

**Done when:** synthetic-data tests confirm each test's size/power behaves as published ✅; full results table generates automatically.

## Phase 4 — Main experiments & ablations (weeks 8–10: Sep 21 – Oct 4)

- [ ] Main grid: all models × all series × horizons {1, 5, 20} × α {1%, 2.5%, 5%}.
- [ ] Regime-split analysis using pre-registered windows.
- [ ] Ablations: context length, input scaling, zero-shot vs. fine-tuned, direct-quantile vs. vol-path.
- [ ] Contamination check: compare in-pretraining vs. clean subsets (post-cutoff periods, crypto).
- [ ] RQ4 diagnostics: PIT calibration histograms, implied kurtosis of predictive distributions, quantile reactivity after shock days (event study around top-decile |return| days), volatility-clustering signature in predicted quantiles.

**Done when:** every number destined for the paper sits in the results database with a config hash.

## Phase 5 — Paper (weeks 10–13: Oct 5 – Nov 1)

- [ ] `paper/related_work.md` → full related-work section (keep annotated bibliography from week 2 onward, ~1 paper read/week during Phases 1–4).
- [ ] Figures: the 3–4 that tell the story (regime-split FZ0 ranking; MCS membership heatmap; RQ4 reactivity plot; calibration).
- [ ] Draft (NeurIPS workshop format, then ICAIF format): intro framed around RQ1–RQ4, honest limitations section.
- [ ] Internal red-team pass: attack own paper as Reviewer 2 (leakage? multiple testing? proxy noise? cherry-picked regimes?) — fix or acknowledge each point.
- [ ] Feedback loop: ITMO advisors / quant community readers.
- [ ] arXiv preprint + code release tag `v1.0-paper`.

---

## What makes this "a notch stronger" — non-negotiables

1. **Pre-registered protocol** (regimes, metrics, α levels frozen before TSFM runs) — kills the cherry-picking critique.
2. **FZ0 joint scoring + MCS** — econometrics-grade statistics that ML-venue papers almost never bring.
3. **Contamination analysis** — reviewers *will* ask whether TSFMs saw the test data; we answer before they ask.
4. **RQ4 diagnostics** — an explanation, not just a leaderboard.
5. **Full reproducibility** — one command per table/figure; cached forecasts published with the code.

## Risks & scope control

| Risk | Mitigation |
|---|---|
| GPU/compute shortage | TSFMs are inference-only; cache forecasts once; Chronos-Bolt & TimesFM run on CPU acceptably for daily data |
| Model list balloons | Core 4 TSFMs are enough for the claim; TimeGPT/Toto/fine-tuning are stretch goals |
| RV data availability | Squared-return proxy + robust QLIKE is the primary track; RV is a supporting subset |
| Timeline slips | Cut order: fine-tuning → extra assets → extra TSFMs. Never cut: statistics, pre-registration, RQ4 |
| A TSFM wrapper breaks | Cached-forecast design isolates inference from evaluation; drop the model, note it in the paper |
