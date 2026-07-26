"""Turn walk-forward forecasts into the paper's evaluation tables.

    uv run python scripts/make_tables.py --forecasts results/baselines_full.parquet

Reads the tidy forecast parquet (schema in pipelines/walkforward.py), computes
the pre-registered endpoints (QLIKE for variance, FZ0 for the VaR/ES pair),
runs the adequacy backtests, and ranks models with Diebold-Mariano and the
Model Confidence Set — overall, per asset class, and per regime.

Writes markdown tables to results/tables/ and one tidy parquet of every metric
so figures never recompute anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_risk.config import load_regimes, load_universe
from tsfm_risk.evaluation.backtests import (
    christoffersen_independence,
    engle_manganelli_dq,
    kupiec_pof,
)
from tsfm_risk.evaluation.comparison import diebold_mariano, model_confidence_set
from tsfm_risk.evaluation.losses import fz0, qlike

ALPHA_MAIN = 0.01  # headline tail level for FZ0 tables


def asset_class_map() -> dict[str, str]:
    universe = load_universe()
    return {spec.ticker: ac for ac, spec in universe.all_series()}


def load_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["realized"])
    df["origin"] = pd.to_datetime(df["origin"])
    regimes = load_regimes()
    dates = df["origin"].dt.date
    crisis = dates.map(regimes.is_crisis)
    df["regime"] = np.where(crisis, "crisis", "calm")
    df["asset_class"] = df["series"].map(asset_class_map())
    return df


# ------------------------------------------------------------------ metrics


def variance_table(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Mean QLIKE per model (variance forecasting endpoint)."""
    sub = df[(df.horizon == horizon) & (df.alpha == df.alpha.min())].copy()
    sub["proxy"] = sub["realized"] ** 2
    sub = sub[sub["sigma2"] > 0]
    sub["qlike"] = qlike(sub["sigma2"].to_numpy(), sub["proxy"].to_numpy())
    out = sub.groupby("model")["qlike"].agg(["mean", "count"])
    return out.rename(columns={"mean": "QLIKE", "count": "n"}).sort_values("QLIKE")


def tail_table(df: pd.DataFrame, horizon: int, alpha: float) -> pd.DataFrame:
    """FZ0 plus adequacy backtests per model."""
    sub = df[(df.horizon == horizon) & (np.isclose(df.alpha, alpha))].copy()
    sub = sub[(sub["var"] > 0) & (sub["es"] >= sub["var"])]
    sub["fz0"] = fz0(
        sub["var"].to_numpy(), sub["es"].to_numpy(), sub["realized"].to_numpy(), alpha
    )

    rows = []
    for model, g in sub.groupby("model"):
        r = g["realized"].to_numpy()
        v = g["var"].to_numpy()
        row = {"model": model, "FZ0": g["fz0"].mean(), "breach_rate": float((r < -v).mean())}
        # backtests are per-series; report the share of series that pass
        passes = {"kupiec": [], "christoffersen": [], "dq": []}
        for _, gs in g.groupby("series"):
            rs, vs = gs["realized"].to_numpy(), gs["var"].to_numpy()
            if rs.size < 200:
                continue
            try:
                passes["kupiec"].append(kupiec_pof(rs, vs, alpha).p_value > 0.05)
                passes["christoffersen"].append(
                    christoffersen_independence(rs, vs).p_value > 0.05
                )
                passes["dq"].append(engle_manganelli_dq(rs, vs, alpha).p_value > 0.05)
            except ValueError:
                continue
        for k, vals in passes.items():
            row[f"pass_{k}"] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values("FZ0")


def aligned_losses(
    df: pd.DataFrame, horizon: int, alpha: float, loss: str
) -> dict[str, np.ndarray]:
    """Per-model loss series aligned on the common (series, origin) index."""
    sub = df[(df.horizon == horizon) & (np.isclose(df.alpha, alpha))].copy()
    if loss == "fz0":
        sub = sub[(sub["var"] > 0) & (sub["es"] >= sub["var"])]
        sub["loss"] = fz0(
            sub["var"].to_numpy(), sub["es"].to_numpy(), sub["realized"].to_numpy(), alpha
        )
    else:
        sub = sub[sub["sigma2"] > 0]
        sub["loss"] = qlike(sub["sigma2"].to_numpy(), (sub["realized"] ** 2).to_numpy())

    wide = sub.pivot_table(index=["series", "origin"], columns="model", values="loss")
    wide = wide.dropna()
    return {m: wide[m].to_numpy() for m in wide.columns}


def rank_models(losses: dict[str, np.ndarray], horizon: int) -> tuple[pd.DataFrame, list[str]]:
    means = {m: float(v.mean()) for m, v in losses.items()}
    best = min(means, key=means.get)
    rows = []
    for m, v in losses.items():
        if m == best:
            rows.append({"model": m, "mean_loss": means[m], "vs_best_p": np.nan})
            continue
        dm = diebold_mariano(v, losses[best], horizon=horizon)
        rows.append({"model": m, "mean_loss": means[m], "vs_best_p": dm.p_value})
    mcs = model_confidence_set(losses, confidence=0.90, n_boot=1000, seed=2026)
    tab = pd.DataFrame(rows).set_index("model").sort_values("mean_loss")
    tab["in_MCS"] = [m in mcs.included for m in tab.index]
    return tab, mcs.included


def md(df: pd.DataFrame, float_fmt: str = "%.4f") -> str:
    return df.to_markdown(floatfmt=float_fmt.replace("%", "").replace("f", "f"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", default="results/baselines_full.parquet")
    ap.add_argument("--outdir", default="results/tables")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.forecasts} ...", flush=True)
    df = load_forecasts(Path(args.forecasts))
    print(f"{len(df):,} rows, {df.series.nunique()} series, {df.model.nunique()} models")

    parts: list[str] = ["# Baseline results\n"]
    parts.append(
        f"Generated from `{args.forecasts}` — {len(df):,} evaluated forecasts, "
        f"{df.series.nunique()} series, {df.origin.min().date()} to {df.origin.max().date()}.\n"
    )

    # ---- volatility endpoint
    vt = variance_table(df, horizon=1)
    parts.append("\n## Volatility forecasting (QLIKE, h=1, lower is better)\n")
    parts.append(vt.to_markdown(floatfmt=".4f"))

    ql = aligned_losses(df, 1, df.alpha.min(), "qlike")
    rank_q, mcs_q = rank_models(ql, horizon=1)
    parts.append("\n\n### Ranking with formal tests (QLIKE, h=1)\n")
    parts.append(rank_q.to_markdown(floatfmt=".4f"))
    parts.append(f"\n\n**Model Confidence Set (90%):** {', '.join(mcs_q)}\n")

    # ---- tail endpoint
    for h in (1, 5, 20):
        tt = tail_table(df, horizon=h, alpha=ALPHA_MAIN)
        parts.append(f"\n## Tail risk, h={h}, alpha={ALPHA_MAIN:.0%} "
                     f"(FZ0 lower is better; pass_* = share of series passing at 5%)\n")
        parts.append(tt.to_markdown(floatfmt=".4f"))
        parts.append("\n")

    fz = aligned_losses(df, 1, ALPHA_MAIN, "fz0")
    rank_f, mcs_f = rank_models(fz, horizon=1)
    parts.append(f"\n### Ranking with formal tests (FZ0, h=1, alpha={ALPHA_MAIN:.0%})\n")
    parts.append(rank_f.to_markdown(floatfmt=".4f"))
    parts.append(f"\n\n**Model Confidence Set (90%):** {', '.join(mcs_f)}\n")

    # ---- regime split
    parts.append("\n## Regime split (FZ0, h=1, alpha=1%)\n")
    reg_rows = []
    for regime, g in df.groupby("regime"):
        losses = aligned_losses(g, 1, ALPHA_MAIN, "fz0")
        if not losses or min(len(v) for v in losses.values()) < 200:
            continue
        rk, inc = rank_models(losses, horizon=1)
        for m in rk.index:
            reg_rows.append({"regime": regime, "model": m,
                             "FZ0": rk.loc[m, "mean_loss"], "in_MCS": rk.loc[m, "in_MCS"]})
    reg = pd.DataFrame(reg_rows).pivot_table(index="model", columns="regime",
                                             values=["FZ0", "in_MCS"], aggfunc="first")
    parts.append(reg.to_markdown(floatfmt=".4f"))

    # ---- asset class split
    parts.append("\n\n## Per asset class (FZ0, h=1, alpha=1%; * = in MCS)\n")
    ac_rows = []
    for ac, g in df.groupby("asset_class"):
        losses = aligned_losses(g, 1, ALPHA_MAIN, "fz0")
        if not losses or min(len(v) for v in losses.values()) < 200:
            continue
        rk, inc = rank_models(losses, horizon=1)
        for m in rk.index:
            ac_rows.append({"asset_class": ac, "model": m,
                            "value": f"{rk.loc[m, 'mean_loss']:.4f}"
                                     + ("*" if rk.loc[m, "in_MCS"] else "")})
    ac_tab = pd.DataFrame(ac_rows).pivot(index="model", columns="asset_class", values="value")
    parts.append(ac_tab.to_markdown())

    (out / "baselines.md").write_text("\n".join(parts))

    tidy = pd.concat(
        [
            vt.assign(metric="QLIKE", horizon=1, alpha=np.nan).reset_index(),
            rank_f.assign(metric="FZ0", horizon=1, alpha=ALPHA_MAIN).reset_index(),
        ],
        ignore_index=True,
    )
    tidy.to_parquet(out / "summary.parquet")
    print(f"\nwrote {out/'baselines.md'} and {out/'summary.parquet'}")
    print("\n" + vt.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nMCS (QLIKE): {mcs_q}")
    print(f"MCS (FZ0 1%): {mcs_f}")


if __name__ == "__main__":
    main()
