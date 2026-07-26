# Protocol amendments

The protocol in `docs/preregistration.md` was frozen at commit `3f2766d`, before
any TSFM inference. Every deviation is recorded here with its date and reason,
so the frozen protocol and the executed protocol can be compared line by line.

---

## Amendment 1 — 2026-07-26: model roster, contamination design, mandatory vol-path

**Trigger.** A pre-implementation survey of the TSFM landscape (model cards,
pretraining corpora, published quantile grids) surfaced three facts that the
original protocol did not anticipate. No experiment had been run at this point;
no result influenced these changes.

### 1.1 Crypto is no longer treated as an uncontaminated holdout

*Original protocol* (§ "Contamination policy") assumed crypto series are largely
absent from pretraining corpora and could serve as a clean out-of-pretraining test.

*Finding.* Salesforce's LOTSA corpus — the pretraining set for Moirai — contains a
`bitcoin_with_missing` subset (plus `fred_md` macro data). The assumption is
therefore false for at least one model in the roster.

*Change.* Crypto is demoted from "clean control" to "one series class among
several". Contamination control moves to a **cutoff-stratified design**: each
model is additionally evaluated on the out-of-sample window that begins after
*its own* published release/data cutoff, and all models are jointly evaluated on
the window after the latest cutoff in the roster. Divergence between a model's
own-window and common-window performance is reported as a contamination
diagnostic.

*Cost.* The common post-cutoff window is short (order of months), so 1% VaR
backtests there have low power. Mitigation: pool across the full asset cross-
section and report the common-window analysis as a robustness check, not as the
primary result. If intraday data can be sourced, a higher-frequency target is
the preferred way to recover power; this is recorded as an open item, not a
commitment.

### 1.2 The vol-path becomes mandatory, not optional

*Original protocol* offered two mappings from model output to VaR/ES
(direct-quantile and vol-path) as complementary.

*Finding.* Most models in the roster emit quantiles only on a 0.1–0.9 grid.
VaR at α ∈ {1%, 2.5%, 5%} is therefore **not directly available** from them;
obtaining it would require extrapolating beyond the lowest predicted quantile,
which silently reimposes a distributional assumption — precisely what the
direct-quantile path was meant to avoid.

*Change.* The vol-path is the primary mapping for all models. The
direct-quantile path is reported only where the model's native grid actually
reaches the required α (e.g. Chronos-2, whose grid includes 0.01 and 0.05), and
the A-vs-B difference is interpreted as that model's own tail-shape contribution.

### 1.3 Roster update

Added: **Chronos-2** (Oct 2025; 21-quantile output including 0.01/0.05) and
**Moirai 2.0** (Aug 2025). Keeping both Chronos-Bolt and Chronos-2 is deliberate:
their different release dates give two distinct post-cutoff boundaries, which the
stratified design in 1.1 exploits.

### 1.4 Related work positioning (no protocol change, recorded for transparency)

Two recent works overlap with this study and must be cited and positioned
against rather than discovered by a reviewer:

- *Forecasting Realized Volatility with Time Series Foundation Models* —
  TSFMs vs GARCH-family and HAR-Q with QLIKE and VaR backtests. Closest prior
  work. Our stated differences: joint (VaR, ES) elicitable scoring via FZ0,
  Model Confidence Set over the full roster, the cutoff-stratified contamination
  design above, and regime-conditional analysis on pre-registered windows.
- *Time-Series Foundation Model for Value-at-Risk Forecasting* — reports a
  **fine-tuned** TimesFM beating GARCH/GAS. The zero-shot question this study
  asks remains open, and the contrast between their fine-tuned result and our
  zero-shot result is itself a finding worth reporting.

**Unchanged by this amendment:** primary endpoints (QLIKE, FZ0), α levels,
horizons, walk-forward specification, crisis windows, and the comparison rules
(DM with HAC, MCS at 90%).
