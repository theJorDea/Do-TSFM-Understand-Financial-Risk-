# Do Time-Series Foundation Models Understand Financial Risk?

**A rigorous, regulation-grade evaluation of zero-shot time-series foundation models for volatility forecasting and tail-risk estimation.**

> 📄 Paper: *work in progress* (target: arXiv preprint → NeurIPS workshop → ACM ICAIF)
> 🔬 Status: **Phase 1 — baselines running on the full universe; evaluation statistics complete** (details in [Roadmap](#roadmap--status))

---

## TL;DR

Time-series foundation models (TSFMs) — Chronos, TimesFM, Moirai, Lag-Llama — report strong zero-shot results on standard forecasting benchmarks. But financial risk management does not care about point forecasts: it cares about the **tails of the return distribution**. Whether TSFMs can price tail risk at all is essentially untested.

We evaluate TSFMs head-to-head against 40 years of econometric wisdom (GARCH family, HAR, filtered historical simulation) on the two tasks a risk desk actually runs: **volatility forecasting** and **Value-at-Risk / Expected Shortfall estimation** — with the full regulatory backtesting toolkit, modern joint elicitable scoring rules, formal statistical comparison, and a pre-registered protocol.

**We commit to publishing the outcome whatever it is.** "Foundation models fail at tail risk" and "foundation models quietly obsolete GARCH" are both publishable findings — what matters is that the evaluation is airtight.

---

## 1. Why this study

**The gap.** TSFM papers evaluate on point/probabilistic accuracy over broad benchmark suites (M-competitions, Monash, GIFT-Eval) dominated by demand, traffic, and energy series. Financial returns are a pathological case for exactly the properties those benchmarks underweight: near-zero predictable mean, heavy tails, volatility clustering, leverage asymmetry, and regime shifts. Meanwhile, the econometrics literature has spent four decades building models for precisely these properties, together with a mature theory of how to *score* tail forecasts (elicitability) and how to *test* forecast superiority. The two literatures barely talk to each other. No existing study, to our knowledge, evaluates TSFMs with joint elicitable VaR/ES scoring, regulatory backtests, and regime-conditional model confidence sets across asset classes.

**Why it matters.** If TSFMs can match GARCH-class models zero-shot, risk systems gain a model that needs no per-asset re-estimation and transfers across markets. If they cannot — especially in crises, which is when risk models earn their keep — that is a concrete, quantified warning against a tempting deployment, and the diagnostics tell vendors *what* is missing from pretraining.

**Why us / why now.** The evaluation machinery (GARCH MLE, VaR/ES backtests) builds on the author's prior [Riskforge](https://github.com/theJorDea/Riskforge) toolkit; TSFM inference is zero-shot, so the study is feasible on modest compute. The model class stabilized in 2024–2025 (Chronos-Bolt, TimesFM 2.x, Moirai, Lag-Llama), making a systematic audit timely rather than premature.

---

## 2. Research questions, hypotheses, and what each outcome would mean

| # | Question | Pre-stated expectation (H) | Interpretation of outcomes |
|---|----------|---------------------------|---------------------------|
| **RQ1** | Do zero-shot TSFMs forecast conditional volatility better than econometric baselines (GARCH family, HAR) at 1–20 day horizons? | H1: TSFMs are *competitive but not superior* on QLIKE at h=1; the gap narrows or reverses at h=20 where mean reversion dominates. | TSFM win ⇒ pretraining transfers vol dynamics across domains. TSFM loss ⇒ generic pretraining does not substitute for a 3-parameter recursion tuned to the series — a striking efficiency statement either way. |
| **RQ2** | Do VaR/ES estimates from TSFM predictive distributions pass regulatory backtests, and how do they score on the joint FZ0 loss? | H2: TSFM quantiles are *under-dispersed in the tails* (predictive distributions too Gaussian), producing VaR breach rates above nominal at α=1% and FZ0 losses worse than GARCH-t. | Systematic under-coverage would be direct evidence that TSFM predictive distributions regress toward unconditional/thin-tailed shapes — invisible in MAE/CRPS-style benchmark metrics, decisive for risk use. |
| **RQ3** | How does the ranking change across market regimes (calm vs. crisis), asset classes, and horizons? | H3: TSFMs degrade *more than* econometric models in pre-registered crisis windows; crypto (least represented in pretraining, most non-Gaussian) is their weakest class. | Regime-conditional MCS membership is the deployment-relevant answer: a model that wins on average but exits the confidence set in crises is not a risk model. |
| **RQ4** | *What* do TSFM predictive distributions actually encode — volatility clustering, leverage effects, fat tails — or none of these? | H4: TSFM quantile bands widen after shocks (some clustering is learned) but implied kurtosis is far below Student-t(ν≈3–6) fits; little or no leverage asymmetry. | This is the explanatory layer: it converts a leaderboard into findings about *what pretraining teaches a forecaster about financial data*, and tells model builders what to fix. |

Hypotheses are stated before any TSFM is run (see the pre-registration commit); confirming *or* refuting them is reported symmetrically.

---

## 3. Study design and the reason for every choice

### 3.1 Two tasks, two pipelines

**Tasks.** (a) Conditional volatility forecasting; (b) VaR and ES at α ∈ {1%, 2.5%, 5%}, horizons h ∈ {1, 5, 20} trading days. These are the quantities regulation (Basel FRTB) and desks actually consume; point return forecasts are deliberately out of scope (daily returns are ≈ unpredictable in mean, so point-forecast metrics mostly measure noise).

**Apples-to-apples mapping.** Every model reaches VaR/ES through one of two declared pipelines:
- **Direct-quantile path** — predictive quantiles → VaR; tail integration of the quantile function → ES. Natural for TSFMs that emit quantiles (Chronos, Moirai).
- **Vol-path** — variance forecast → location-scale distribution (Student-t innovations) → VaR/ES. Natural for GARCH/EWMA/HAR; also applied to TSFM *variance* forecasts so the comparison isolates *where* TSFMs fail, if they do (bad vol level vs. bad tail shape).

Running TSFMs through both pipelines is itself an ablation for RQ4.

### 3.2 Model universe

| Family | Model | Why it is in the study |
|---|---|---|
| Non-parametric | Historical simulation (rolling 1000d) | The industry floor; any serious model must beat it |
| Non-parametric | Filtered historical simulation (EWMA-devolatilized, path-wise vol updating for h>1) | The strongest simple classical tail model; the honest bar for TSFMs |
| Econometric | EWMA / RiskMetrics (λ=0.94, normal) | The canonical industry baseline, kept unmodified by convention |
| Econometric | GARCH(1,1)-Normal / GARCH(1,1)-t | The default academic benchmark; the -t variant separates "vol dynamics" from "tail shape" |
| Econometric | GJR-GARCH(1,1)-t | Adds the leverage effect — tests whether asymmetry matters for tails |
| Econometric | HAR (direct projection per horizon) | The high bar for pure volatility forecasting; runs on squared-return proxy now, realized variance where available |
| Neural supervised | LSTM trained with QLIKE | Controls for "any neural net" vs. "pretrained foundation model" |
| **TSFM** | Chronos-Bolt (Amazon) | Quantile-native, strongest published zero-shot record |
| **TSFM** | TimesFM (Google) | Largest-scale pretraining corpus |
| **TSFM** | Moirai (Salesforce) | Emits a full parametric predictive distribution — ideal for RQ4 |
| **TSFM** | Lag-Llama | Probabilistic, small, fine-tunable — the fine-tuning ablation vehicle |

Stretch: TimeGPT (API), Toto; light fine-tuning of Chronos/Lag-Llama. These test robustness of conclusions, not the headline claim.

### 3.3 Data

~21 daily series, 2000–2026, across six classes (equity indices, single stocks, FX, crypto, commodities, bond ETFs) — defined in `configs/universe.yaml`. Reasons: multiple classes prevent "works on the S&P" overfitting of conclusions; crypto doubles as a *pretraining-contamination control* (post-cutoff and low-representation data); the 26-year span covers four major crisis episodes in the out-of-sample period (2010–2026).

**Volatility proxy.** Daily squared returns everywhere — noisy but conditionally unbiased, which is exactly the condition under which QLIKE ranking is robust (Patton 2011). Five-minute realized variance on the subset where it exists (Oxford-Man for indices to 2022, exchange data for crypto) as a supporting track. Reason: proxy-robustness makes the primary track immune to the "your proxy is noise" objection; RV subset shows conclusions survive a sharper proxy.

### 3.4 Out-of-sample protocol (leakage discipline)

Strict walk-forward: 1000-day rolling estimation window, classical models re-fit every 21 days with daily variance filtering between refits, TSFMs zero-shot with the same context window. No information after the forecast origin enters any input — enforced by the engine and **proven by a unit test that corrupts all post-origin data and asserts bit-identical forecasts**. Multi-day targets are aggregated (t, t+h] returns, never overlapping into the estimation window.

**Pretraining contamination** is the leakage the engine cannot control: TSFMs may have seen our test years in pretraining. We (a) record each model's declared data cutoff and corpus, (b) report results separately on post-cutoff periods and on crypto, (c) treat agreement between contaminated and clean subsets as evidence the conclusions are not memorization artifacts. Reviewers will ask; the protocol answers first.

### 3.5 Scoring: why QLIKE and FZ0 are the primary endpoints

- **QLIKE** for variance: robust to noisy proxies (MSE is reported but known to be outlier-dominated). Primary volatility metric.
- **FZ0** (Fissler–Ziegel) for tails: (VaR, ES) is *jointly elicitable* — there exists a strictly consistent scoring rule for the pair, and FZ0 is its standard 0-homogeneous form. This is the econometrics-grade way to *rank* tail forecasters. ML papers almost never use it; regulatory backtests alone cannot rank (a model can pass Kupiec while being far from optimal). Primary tail metric.
- **Backtests as adequacy diagnostics, not rankings:** Kupiec POF (unconditional coverage), Christoffersen (breach independence + conditional coverage), Engle–Manganelli DQ (dependence on lagged hits and the VaR level itself), Acerbi–Székely Z2 for ES (with the null simulated from the *implied-ν* Student-t matched to each day's forecast pair — the null is anchored to the model's own stated tail, not an arbitrary reference family).

### 3.6 Statistical comparison: no claim without a test

- **Diebold–Mariano** with HAC (Newey–West) errors and the Harvey–Leybourne–Newbold small-sample correction; lag truncation h−1 to absorb the MA(h−1) dependence of overlapping multi-day losses.
- **Model Confidence Set** (Hansen–Lunde–Nason, T_max, moving-block bootstrap) over the full model universe, per {asset class × regime × horizon × α} cell and pooled. The MCS answers the question a single pairwise test cannot: *which set of models is statistically indistinguishable from the best?*
- **Every test's size and power is itself verified** by Monte Carlo on synthetic data in the test suite (e.g., Kupiec rejects ~5% under a true model; DQ rejects >60% under GARCH misspecification). Reason: a comparison is only as credible as the tests it uses.

### 3.7 Pre-registration

`configs/regimes.yaml` (eight crisis windows from the 2008 GFC to the April-2025 tariff shock, fixed by date) and `docs/preregistration.md` (metrics, α levels, horizons, walk-forward spec, comparison rules) were committed **before any TSFM inference**. Deviations require a dated amendment. Reason: regime-conditional analysis is where cherry-picking accusations live; freezing the windows first kills that critique.

### 3.8 RQ4 diagnostics (the explanatory layer)

- **PIT calibration** histograms of realized returns under each model's predictive distribution;
- **implied kurtosis / tail-index** of TSFM predictive distributions vs. GARCH-t fitted ν;
- **shock reactivity**: event-study of predicted quantile bands around top-decile |return| days — do TSFM tails widen, how fast, and do they discriminate sign (leverage)?
- **volatility-clustering signature**: autocorrelation of predicted vs. realized variance.

Reason: a leaderboard says *whether*; these diagnostics say *why*, which is what makes the paper more than a benchmark.

---

## 4. Results so far

**Infrastructure validation (complete).** 82 unit tests, CI green:
- GARCH MLE matches the `arch` package within 0.01–0.015 on parameters and 3% on forecasts; recovers true parameters (including Student-t ν and GJR γ) on 20k-observation synthetic series.
- All closed-form VaR/ES expressions verified by Monte Carlo (2–4M draws).
- Backtest size ≈ nominal and power >60–90% against planted misspecifications, on synthetic data with known truth.
- Leakage test passes: corrupting all future data leaves forecasts bit-identical.

**First real-data pass (S&P 500 2000–2026 + BTC 2014–2026, 563k forecast rows).** One-day 5% VaR breach rates across ~8,950 out-of-sample days:

| model | breach rate (nominal 0.050) |
|---|---|
| historical | 0.0496 |
| garch11-n | 0.0490 |
| har-proxy | 0.0461 |
| fhs-ewma | 0.0521 |
| gjr-garch11-t | 0.0562 |
| garch11-t | 0.0566 |
| ewma-rm | 0.0557 |

Sanity confirmed: all baselines within [0.046, 0.057] of nominal at the 5% level. (Formal Kupiec/DQ verdicts, 1% level, and FZ0 rankings come with the full-universe run — in progress.)

**Full-universe baseline grid (in progress).** 21 series × 7 models × 3 horizons × 3 α levels on all 12 cores; output lands in `results/baselines_full.parquet`.

---

## Roadmap & status

- [x] **Phase 0 — Infrastructure**: uv/ruff/pytest/CI scaffolding; pydantic-validated configs; **pre-registered protocol committed before any TSFM run**
- [x] **Phase 1 — Data & classical baselines**: cached loaders, leakage-safe return aggregation; historical, FHS, EWMA, GARCH-N/-t, GJR-t, HAR; walk-forward engine with leakage proof; parallel grid runner *(full-universe run executing; data card and RV subset remain)*
- [ ] **Phase 2 — TSFM integration**: unified `predict_quantiles` wrapper (Chronos-Bolt, TimesFM, Moirai, Lag-Llama); both risk pipelines; forecast caching to parquet; contamination notes per model; LSTM-QLIKE baseline; *(stretch: fine-tuning, TimeGPT)*
- [x] **Phase 3 — Evaluation & statistics** *(built early — dependencies for Phase 2 analysis)*: QLIKE, FZ0, Kupiec, Christoffersen, DQ, Acerbi–Székely Z2, Diebold–Mariano + HLN, Model Confidence Set; all size/power-validated on synthetic data *(remaining: results database + table/figure generators)*
- [ ] **Phase 4 — Main experiments & ablations**: full grid; regime splits; context-length / scaling / pipeline ablations; contamination analysis; RQ4 diagnostics
- [ ] **Phase 5 — Paper**: figures, related work, red-team pass, arXiv preprint + `v1.0-paper` tag

Detailed week-by-week plan with acceptance criteria and risk controls: [PLAN.md](PLAN.md).

---

## Repository structure

```
├── configs/            # experiment configs — every result maps to one config
├── docs/               # preregistration.md and protocol amendments
├── src/tsfm_risk/
│   ├── config.py       # pydantic-validated experiment configs
│   ├── simulate.py     # synthetic generators for estimator/test validation
│   ├── data/           # cached loaders, returns, vol proxies
│   ├── models/
│   │   ├── base.py     # RiskModel interface (fit / forecast contract)
│   │   ├── classical/  # historical, FHS, EWMA, GARCH family, HAR
│   │   ├── neural/     # LSTM (QLIKE)                     [Phase 2]
│   │   └── tsfm/       # Chronos, TimesFM, Moirai, Lag-Llama wrappers [Phase 2]
│   ├── risk/           # VaR/ES: closed forms, empirical, quantile-grid
│   ├── evaluation/     # losses, backtests, ES test, DM, MCS
│   └── pipelines/      # walk-forward engine
├── scripts/            # run_baselines.py (parallel grid runner)
├── notebooks/          # exploratory analysis & diagnostics
├── tests/              # 82 tests: unit, vs-arch validation, MC size/power
├── results/            # generated outputs (never edited by hand)
└── paper/              # LaTeX source                      [Phase 5]
```

## Reproducibility

- Deterministic seeds everywhere (including bootstrap and FHS simulation), pinned dependencies (`uv.lock`), config-driven runs.
- Data cached to parquet on first download; experiments re-run offline.
- CI runs lint + full test suite on every push.
- Classical estimators cross-validated against `arch`; statistical tests validated by Monte Carlo in the same suite.

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
