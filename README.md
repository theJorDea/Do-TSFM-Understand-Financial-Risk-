# Do Time-Series Foundation Models Understand Financial Risk?

**English** | [Русский](#понимают-ли-foundation-модели-временных-рядов-финансовый-риск)

**A rigorous, regulation-grade evaluation of zero-shot time-series foundation models for volatility forecasting and tail-risk estimation.**

> 📄 Paper: *work in progress* (target: arXiv preprint → NeurIPS workshop → ACM ICAIF)
> 🔬 Status: **Phase 1 complete — full baseline grid evaluated; next up: TSFM integration** (details in [Roadmap](#roadmap--status))

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

**Full-universe baselines (complete).** Walk-forward over 21 series × 7 models × 3 horizons × 3 α levels: **6.6M evaluated forecasts, no failed origins.** Full tables in [`results/tables/baselines.md`](results/tables/baselines.md), regenerated by `scripts/make_tables.py`.

Tail risk at h=1, α=1% (FZ0 lower is better; `pass_*` = share of the 21 series passing that backtest at 5%):

| model | FZ0 | breach rate | pass Kupiec | pass Christoffersen | pass DQ |
|---|---|---|---|---|---|
| **gjr-garch11-t** | **−3.2181** | 0.0118 | 0.67 | **1.00** | 0.57 |
| garch11-t | −3.2105 | 0.0118 | 0.67 | 0.90 | 0.48 |
| fhs-ewma | −3.1903 | 0.0113 | 0.90 | 0.71 | 0.29 |
| garch11-n | −3.1366 | 0.0166 | 0.05 | 0.81 | 0.14 |
| ewma-rm | −3.0549 | 0.0200 | 0.00 | 0.81 | 0.00 |
| har-proxy | −3.0433 | 0.0158 | 0.10 | 0.62 | 0.00 |
| historical | −2.8983 | 0.0125 | 0.52 | 0.14 | 0.10 |

Four findings that set up the TSFM comparison:

1. **GJR-GARCH-t is the sole occupant of the 90% Model Confidence Set** — on QLIKE *and* on FZ0. Leverage plus Student-t tails is the bar the foundation models must clear, not a generic GARCH(1,1).
2. **Normal innovations fail unconditional coverage almost everywhere** at the 1% level: breach rates of 1.66% (garch11-n) and 2.00% (ewma-rm) against nominal 1%, passing Kupiec on 5% and 0% of series. This is the exact failure mode we hypothesise for TSFMs (H2) — so the study can detect it.
3. **Historical simulation passes Christoffersen independence on only 14% of series** (breaches cluster in crises) while GJR-GARCH-t passes on 100%.
4. **Even the winner passes the DQ test on just 57% of series.** The baselines are strong but not unbeatable — there is real headroom for a model that reads the tail better.

---

## Roadmap & status

- [x] **Phase 0 — Infrastructure**: uv/ruff/pytest/CI scaffolding; pydantic-validated configs; **pre-registered protocol committed before any TSFM run**
- [x] **Phase 1 — Data & classical baselines**: cached loaders, leakage-safe return aggregation, bad-tick audit and [data card](docs/data_card.md); historical, FHS, EWMA, GARCH-N/-t, GJR-t, HAR; walk-forward engine with leakage proof; parallel grid runner; **full-universe run complete** *(RV subset remains)*
- [ ] **Phase 2 — TSFM integration**: unified `predict_quantiles` wrapper (Chronos-Bolt, TimesFM, Moirai, Lag-Llama); both risk pipelines; forecast caching to parquet; contamination notes per model; LSTM-QLIKE baseline; *(stretch: fine-tuning, TimeGPT)*
- [x] **Phase 3 — Evaluation & statistics** *(built early — dependencies for Phase 2 analysis)*: QLIKE, FZ0, Kupiec, Christoffersen, DQ, Acerbi–Székely Z2, Diebold–Mariano + HLN, Model Confidence Set; all size/power-validated on synthetic data *(table generator done: `scripts/make_tables.py`; figures remain)*
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

---
---

# Понимают ли foundation-модели временных рядов финансовый риск?

[English](#do-time-series-foundation-models-understand-financial-risk) | **Русский**

**Строгая оценка регуляторного уровня zero-shot foundation-моделей временных рядов в задачах прогноза волатильности и хвостового риска.**

> 📄 Статья: *в работе* (цель: препринт arXiv → воркшоп NeurIPS → ACM ICAIF)
> 🔬 Статус: **Phase 1 завершена — полный грид бейзлайнов оценён; далее — интеграция TSFM**

---

## Кратко

Foundation-модели для временных рядов (TSFM) — Chronos, TimesFM, Moirai, Lag-Llama — показывают сильные zero-shot результаты на стандартных бенчмарках прогнозирования. Но риск-менеджменту не нужны точечные прогнозы: ему нужны **хвосты распределения доходностей**. Умеют ли TSFM оценивать хвостовой риск — по сути, никем не проверено.

Мы сталкиваем TSFM лицом к лицу с 40 годами эконометрики (семейство GARCH, HAR, фильтрованная историческая симуляция) на двух задачах, которые реально решает риск-деск: **прогноз волатильности** и **оценка Value-at-Risk / Expected Shortfall** — с полным набором регуляторных бэктестов, современными совместными скоринговыми правилами, формальными статистическими тестами и пре-регистрированным протоколом.

**Мы обязуемся опубликовать любой исход.** И «foundation-модели не справляются с хвостовым риском», и «foundation-модели незаметно устаревают GARCH» — публикуемые результаты; важно лишь, чтобы оценка была безупречной.

---

## 1. Зачем это исследование

**Пробел.** Статьи о TSFM оценивают модели по точечной/вероятностной точности на широких наборах (M-competitions, Monash, GIFT-Eval), где доминируют ряды спроса, трафика и энергетики. Финансовые доходности — патологический случай именно по тем свойствам, которые эти бенчмарки недооценивают: почти нулевое предсказуемое среднее, тяжёлые хвосты, кластеризация волатильности, leverage-асимметрия, смены режимов. Эконометрика же четыре десятилетия строила модели ровно под эти свойства — вместе со зрелой теорией того, как *оценивать* хвостовые прогнозы (элиситабельность) и как *тестировать* превосходство прогнозов. Эти две литературы почти не пересекаются. Насколько нам известно, ни одна работа не оценивала TSFM с совместным элиситабельным скорингом VaR/ES, регуляторными бэктестами и режимно-условными множествами доверия моделей по классам активов.

**Почему это важно.** Если TSFM zero-shot догоняют GARCH-класс — риск-системы получают модель без пересчёта под каждый актив, переносимую между рынками. Если нет — особенно в кризисы, когда риск-модели и зарабатывают свой хлеб, — это конкретное, количественное предостережение против соблазнительного деплоя, а диагностика подскажет разработчикам, *чего именно* не хватает претрейну.

**Почему мы и почему сейчас.** Оценочная машинерия (MLE для GARCH, бэктесты VaR/ES) опирается на предыдущий инструментарий автора — [Riskforge](https://github.com/theJorDea/Riskforge); инференс TSFM zero-shot, поэтому исследование выполнимо на скромном железе. Класс моделей стабилизировался в 2024–2025 (Chronos-Bolt, TimesFM 2.x, Moirai, Lag-Llama) — систематический аудит своевременен, а не преждевременен.

---

## 2. Вопросы исследования, гипотезы и интерпретация исходов

| # | Вопрос | Заранее заявленное ожидание (H) | Интерпретация исходов |
|---|--------|--------------------------------|----------------------|
| **RQ1** | Прогнозируют ли zero-shot TSFM условную волатильность лучше эконометрических бейзлайнов (GARCH, HAR) на горизонтах 1–20 дней? | H1: TSFM *конкурентны, но не превосходят* по QLIKE при h=1; разрыв сужается или разворачивается при h=20, где доминирует возврат к среднему. | Победа TSFM ⇒ претрейн переносит динамику волатильности между доменами. Поражение ⇒ универсальный претрейн не заменяет рекурсию из трёх параметров, настроенную под ряд, — яркое утверждение об эффективности в любую сторону. |
| **RQ2** | Проходят ли VaR/ES из предиктивных распределений TSFM регуляторные бэктесты и как они ранжируются по совместному лоссу FZ0? | H2: квантили TSFM *недорассеяны в хвостах* (распределения слишком гауссовские): частота пробоев VaR выше номинала при α=1%, FZ0 хуже, чем у GARCH-t. | Систематическое недопокрытие — прямое свидетельство того, что предиктивные распределения TSFM стягиваются к безусловным/тонкохвостым формам: невидимо в метриках типа MAE/CRPS, решающе для риска. |
| **RQ3** | Как меняется ранжирование между режимами рынка (спокойный/кризис), классами активов и горизонтами? | H3: в пре-регистрированных кризисных окнах TSFM деградируют *сильнее* эконометрики; крипта (наименее представленная в претрейне, наиболее негауссовская) — их слабейший класс. | Режимно-условное членство в MCS — и есть ответ уровня деплоя: модель, выигрывающая «в среднем», но выпадающая из множества доверия в кризис, — не риск-модель. |
| **RQ4** | *Что* на самом деле кодируют предиктивные распределения TSFM — кластеризацию волатильности, leverage-эффект, тяжёлые хвосты — или ничего из этого? | H4: квантильные полосы TSFM расширяются после шоков (какая-то кластеризация выучена), но implied-эксцесс существенно ниже Student-t(ν≈3–6); leverage-асимметрии мало или нет. | Объяснительный слой: превращает лидерборд в выводы о том, *чему претрейн учит прогнозиста про финансовые данные*, и говорит разработчикам, что чинить. |

Гипотезы зафиксированы до запуска какой-либо TSFM (см. пре-регистрационный коммит); подтверждение *и* опровержение публикуются симметрично.

---

## 3. Дизайн исследования и причина каждого выбора

### 3.1 Две задачи, два пайплайна

**Задачи.** (a) Прогноз условной волатильности; (b) VaR и ES при α ∈ {1%, 2.5%, 5%}, горизонты h ∈ {1, 5, 20} торговых дней. Именно эти величины потребляют регуляция (Basel FRTB) и дески; точечные прогнозы доходности сознательно вне рамок (дневные доходности почти непредсказуемы в среднем, и такие метрики меряют в основном шум).

**Сравнение «яблок с яблоками».** Каждая модель приходит к VaR/ES одним из двух объявленных пайплайнов:
- **Прямой квантильный путь** — предиктивные квантили → VaR; интегрирование хвоста квантильной функции → ES. Естественно для TSFM, выдающих квантили (Chronos, Moirai).
- **Vol-путь** — прогноз дисперсии → location-scale распределение (инновации Student-t) → VaR/ES. Естественно для GARCH/EWMA/HAR; применяется и к *дисперсионным* прогнозам TSFM, чтобы изолировать, *где* TSFM ломаются (уровень волатильности vs форма хвоста).

Прогон TSFM через оба пайплайна — сам по себе абляция для RQ4.

### 3.2 Вселенная моделей

| Семейство | Модель | Зачем она в исследовании |
|---|---|---|
| Непараметрич. | Историческая симуляция (окно 1000 дн.) | Индустриальный пол; любую серьёзную модель обязан бить |
| Непараметрич. | Фильтрованная историческая симуляция (EWMA-девольтализация, обновление волатильности внутри путей при h>1) | Сильнейшая простая классическая хвостовая модель; честная планка для TSFM |
| Эконометрика | EWMA / RiskMetrics (λ=0.94, нормальные инновации) | Канонический индустриальный бейзлайн, по конвенции не трогаем |
| Эконометрика | GARCH(1,1)-Normal / GARCH(1,1)-t | Академический бенчмарк по умолчанию; вариант -t отделяет «динамику волатильности» от «формы хвоста» |
| Эконометрика | GJR-GARCH(1,1)-t | Добавляет leverage-эффект — проверка, важна ли асимметрия для хвостов |
| Эконометрика | HAR (прямая проекция на каждый горизонт) | Высокая планка чистого прогноза волатильности; сейчас на прокси r², на RV — где доступно |
| Нейросети | LSTM, обученная на QLIKE | Контроль «любая нейросеть» vs «претрейненная foundation-модель» |
| **TSFM** | Chronos-Bolt (Amazon) | Квантильно-нативная, сильнейший опубликованный zero-shot результат |
| **TSFM** | TimesFM (Google) | Крупнейший корпус претрейна |
| **TSFM** | Moirai (Salesforce) | Выдаёт полное параметрическое предиктивное распределение — идеальна для RQ4 |
| **TSFM** | Lag-Llama | Вероятностная, компактная, дообучаемая — носитель абляции с файнтюнингом |

Опционально: TimeGPT (API), Toto; лёгкий файнтюнинг Chronos/Lag-Llama. Это проверка робастности выводов, не главное утверждение.

### 3.3 Данные

~21 дневная серия, 2000–2026, шесть классов (индексы акций, отдельные акции, валюты, крипта, сырьё, облигационные ETF) — заданы в `configs/universe.yaml`. Причины: несколько классов исключают перенос выводов «работает на S&P» на всё подряд; крипта заодно служит *контролем контаминации претрейна* (пост-cutoff данные с низкой представленностью); 26-летний диапазон покрывает четыре крупных кризиса в out-of-sample периоде (2010–2026).

**Прокси волатильности.** Квадраты дневных доходностей везде — шумный, но условно несмещённый прокси, и это ровно то условие, при котором ранжирование по QLIKE робастно (Patton 2011). Пятиминутная реализованная дисперсия на подмножестве, где она есть (Oxford-Man для индексов до 2022, биржевые данные для крипты), — поддерживающий трек. Причина: робастность к прокси делает основной трек неуязвимым к возражению «ваш прокси — шум»; RV-подмножество показывает, что выводы выживают и при более точном прокси.

### 3.4 Out-of-sample протокол (дисциплина утечек)

Строгий walk-forward: скользящее окно оценивания 1000 дней, классические модели переоцениваются каждый 21 день с ежедневной фильтрацией дисперсии между переоценками, TSFM zero-shot с тем же контекстным окном. Никакая информация после даты прогноза не попадает во входы — это обеспечивает движок и **доказывает юнит-тест: при подмене всех данных после даты прогноза мусором прогнозы совпадают бит-в-бит**. Многодневные цели — агрегированные доходности (t, t+h], никогда не перекрывающиеся с окном оценивания.

**Контаминация претрейна** — та утечка, которую движок контролировать не может: TSFM могли видеть наши тестовые годы при обучении. Мы (a) фиксируем заявленный cutoff и корпус каждой модели, (b) отдельно репортим результаты на пост-cutoff периодах и крипте, (c) согласие «контаминированных» и «чистых» подмножеств трактуем как свидетельство того, что выводы — не артефакт запоминания. Рецензенты спросят — протокол отвечает заранее.

### 3.5 Скоринг: почему первичные метрики — QLIKE и FZ0

- **QLIKE** для дисперсии: робастен к шумным прокси (MSE репортим, но он, как известно, задавлен выбросами). Первичная метрика волатильности.
- **FZ0** (Фисслер–Цигель) для хвостов: пара (VaR, ES) *совместно элиситабельна* — для неё существует строго согласованное скоринговое правило, и FZ0 — его стандартная 0-однородная форма. Это эконометрический способ *ранжировать* хвостовые прогнозы. В ML-статьях он почти не встречается; одни регуляторные бэктесты ранжировать не умеют (модель может пройти Купика, будучи далеко не оптимальной). Первичная хвостовая метрика.
- **Бэктесты как диагностика адекватности, а не рейтинг:** Kupiec POF (безусловное покрытие), Christoffersen (независимость пробоев + условное покрытие), DQ Энгла–Манганелли (зависимость от лагов пробоев и самого уровня VaR), Acerbi–Székely Z2 для ES (нуль симулируется из *implied-ν* Student-t, подогнанного под собственную (VaR, ES)-пару прогноза на каждый день — нуль привязан к заявленному моделью хвосту, а не к произвольному эталону).

### 3.6 Статистическое сравнение: ни одного утверждения без теста

- **Diebold–Mariano** с HAC-ошибками (Newey–West) и поправкой малой выборки Harvey–Leybourne–Newbold; усечение лагов h−1 поглощает MA(h−1)-зависимость перекрывающихся многодневных лоссов.
- **Model Confidence Set** (Hansen–Lunde–Nason, статистика T_max, блочный бутстрап) по всей вселенной моделей, в каждой ячейке {класс активов × режим × горизонт × α} и суммарно. MCS отвечает на вопрос, недоступный парным тестам: *какое множество моделей статистически неотличимо от лучшей?*
- **Размер и мощность каждого теста проверены** Монте-Карло на синтетике прямо в тестовом наборе (например, Купик отвергает ~5% при верной модели; DQ отвергает >60% при GARCH-мисспецификации). Причина: сравнение стоит ровно столько, сколько стоят используемые тесты.

### 3.7 Пре-регистрация

`configs/regimes.yaml` (восемь кризисных окон от GFC-2008 до тарифного шока апреля 2025, зафиксированы датами) и `docs/preregistration.md` (метрики, уровни α, горизонты, спецификация walk-forward, правила сравнения) закоммичены **до какого-либо инференса TSFM**. Отклонения требуют датированной поправки. Причина: обвинения в cherry-picking живут именно в режимном анализе; заморозка окон заранее убивает эту критику.

### 3.8 Диагностика RQ4 (объяснительный слой)

- **PIT-калибровка** — гистограммы реализованных доходностей под предиктивным распределением каждой модели;
- **implied-эксцесс / хвостовой индекс** предиктивных распределений TSFM против подогнанного ν у GARCH-t;
- **реактивность на шоки** — event-study квантильных полос вокруг дней с |доходностью| из верхнего дециля: расширяются ли хвосты TSFM, как быстро, различают ли знак (leverage);
- **сигнатура кластеризации волатильности** — автокорреляция предсказанной и реализованной дисперсии.

Причина: лидерборд отвечает *«лучше ли»*; диагностика отвечает *«почему»* — именно это делает работу больше, чем бенчмарком.

---

## 4. Что уже получено

**Валидация инфраструктуры (завершена).** 82 юнит-теста, CI зелёный:
- MLE GARCH совпадает с пакетом `arch` в пределах 0.01–0.015 по параметрам и 3% по прогнозам; восстанавливает истинные параметры (включая ν Стьюдента и γ GJR) на синтетике в 20 тыс. наблюдений.
- Все закрытые формулы VaR/ES проверены Монте-Карло (2–4 млн испытаний).
- Размер бэктестов ≈ номинальному, мощность >60–90% против подложенных мисспецификаций — на синтетике с известной истиной.
- Тест утечек проходит: порча всех будущих данных оставляет прогнозы бит-в-бит идентичными.

**Первый прогон на реальных данных (S&P 500 2000–2026 + BTC 2014–2026, 563 тыс. прогнозных строк).** Частота пробоев однодневного 5% VaR на ~8 950 out-of-sample днях:

| модель | частота пробоев (номинал 0.050) |
|---|---|
| historical | 0.0496 |
| garch11-n | 0.0490 |
| har-proxy | 0.0461 |
| fhs-ewma | 0.0521 |
| gjr-garch11-t | 0.0562 |
| garch11-t | 0.0566 |
| ewma-rm | 0.0557 |

Здравость подтверждена: все бейзлайны в [0.046, 0.057] при номинале 5%. (Формальные вердикты Купика/DQ, уровень 1% и ранжирование по FZ0 — после полного прогона, он выполняется.)

**Полный грид бейзлайнов (завершён).** 21 серия × 7 моделей × 3 горизонта × 3 уровня α: **6.6 млн оценённых прогнозов, ни одного сбоя**. Таблицы — в [`results/tables/baselines.md`](results/tables/baselines.md).

Хвостовой риск при h=1, α=1% (FZ0 меньше — лучше; `pass_*` — доля из 21 серии, прошедшая бэктест на уровне 5%):

| модель | FZ0 | частота пробоев | Kupiec | Christoffersen | DQ |
|---|---|---|---|---|---|
| **gjr-garch11-t** | **−3.2181** | 0.0118 | 0.67 | **1.00** | 0.57 |
| garch11-t | −3.2105 | 0.0118 | 0.67 | 0.90 | 0.48 |
| fhs-ewma | −3.1903 | 0.0113 | 0.90 | 0.71 | 0.29 |
| garch11-n | −3.1366 | 0.0166 | 0.05 | 0.81 | 0.14 |
| ewma-rm | −3.0549 | 0.0200 | 0.00 | 0.81 | 0.00 |
| har-proxy | −3.0433 | 0.0158 | 0.10 | 0.62 | 0.00 |
| historical | −2.8983 | 0.0125 | 0.52 | 0.14 | 0.10 |

Четыре вывода, задающие рамку для сравнения с TSFM:

1. **GJR-GARCH-t — единственный обитатель 90% Model Confidence Set**, и по QLIKE, и по FZ0. Планка для foundation-моделей — именно он, а не обычный GARCH(1,1).
2. **Нормальные инновации почти повсеместно проваливают безусловное покрытие** на уровне 1%: частоты пробоев 1.66% и 2.00% при номинале 1%, тест Купика проходят на 5% и 0% серий. Это ровно тот режим отказа, который мы предполагаем у TSFM (H2), — значит, исследование способно его обнаружить.
3. **Историческая симуляция проходит тест независимости лишь на 14% серий** (пробои кучкуются в кризисы), тогда как GJR-GARCH-t — на 100%.
4. **Даже победитель проходит DQ-тест лишь на 57% серий.** Бейзлайны сильны, но небезупречны — запас для модели, лучше читающей хвост, реально существует.

---

## Дорожная карта и статус

- [x] **Phase 0 — Инфраструктура**: каркас uv/ruff/pytest/CI; pydantic-валидация конфигов; **пре-регистрированный протокол закоммичен до запуска TSFM**
- [x] **Phase 1 — Данные и классические бейзлайны**: кэширующие загрузчики, безопасная к утечкам агрегация, аудит битых тиков и [data card](docs/data_card.md); historical, FHS, EWMA, GARCH-N/-t, GJR-t, HAR; walk-forward движок с доказательством отсутствия утечек; параллельный грид-раннер; **полный прогон завершён** *(остаётся RV-подмножество)*
- [ ] **Phase 2 — Интеграция TSFM**: единый интерфейс `predict_quantiles` (Chronos-Bolt, TimesFM, Moirai, Lag-Llama); оба риск-пайплайна; кэш прогнозов в parquet; заметки о контаминации; LSTM-QLIKE бейзлайн; *(опционально: файнтюнинг, TimeGPT)*
- [x] **Phase 3 — Оценка и статистика** *(построена досрочно)*: QLIKE, FZ0, Kupiec, Christoffersen, DQ, Acerbi–Székely Z2, Diebold–Mariano + HLN, Model Confidence Set; всё валидировано по размеру/мощности на синтетике *(генератор таблиц готов: `scripts/make_tables.py`; остались графики)*
- [ ] **Phase 4 — Основные эксперименты и абляции**: полный грид; режимные срезы; абляции контекста/масштабирования/пайплайна; анализ контаминации; диагностика RQ4
- [ ] **Phase 5 — Статья**: графики, related work, red-team проход, препринт arXiv + тег `v1.0-paper`

Детальный понедельный план с критериями приёмки и контролем рисков: [PLAN.md](PLAN.md).

## Воспроизводимость

- Детерминированные сиды везде (включая бутстрап и симуляцию FHS), зафиксированные зависимости (`uv.lock`), запуски из конфигов.
- Данные кэшируются в parquet при первой загрузке; эксперименты воспроизводятся офлайн.
- CI гоняет линтер и полный тестовый набор на каждый push.
- Классические оценщики кросс-валидированы против `arch`; статистические тесты валидированы Монте-Карло в том же наборе.

## Лицензия

MIT
