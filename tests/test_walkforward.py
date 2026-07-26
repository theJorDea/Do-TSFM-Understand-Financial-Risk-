import numpy as np
import pandas as pd
import pytest

from tsfm_risk.models.classical import Ewma, Garch, HistoricalSimulation
from tsfm_risk.pipelines.walkforward import (
    WalkForwardConfig,
    run_many,
    run_walkforward,
)
from tsfm_risk.simulate import simulate_garch

CFG = WalkForwardConfig(window=500, refit_every=63, horizons=(1, 5), alphas=(0.01, 0.05))


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    values = simulate_garch(900, seed=21)
    return pd.Series(values, index=pd.bdate_range("2015-01-01", periods=900))


class TestEngine:
    def test_shape_and_columns(self, returns):
        res = run_walkforward(Ewma(), returns, CFG)
        df = res.forecasts
        n_origins = 900 - 500 + 1
        assert len(df) == n_origins * len(CFG.horizons) * len(CFG.alphas)
        assert set(df.columns) == {
            "origin", "model", "horizon", "alpha", "sigma2", "var", "es", "realized",
        }
        assert not res.failures

    def test_realized_alignment(self, returns):
        res = run_walkforward(Ewma(), returns, CFG)
        df = res.forecasts
        row = df[(df.horizon == 5) & (df.alpha == 0.05)].iloc[0]
        t = returns.index.get_loc(row.origin)
        np.testing.assert_allclose(row.realized, returns.iloc[t + 1 : t + 6].sum())
        # last origins have no complete 5-day future window -> NaN realized
        last = df[(df.horizon == 5)].iloc[-1]
        assert np.isnan(last.realized)

    def test_refit_schedule(self, returns):
        res = run_walkforward(Garch(dist="normal"), returns, CFG)
        n_origins = 900 - 500 + 1
        assert res.n_refits == int(np.ceil(n_origins / CFG.refit_every))

    def test_no_leakage_from_future(self, returns):
        """Forecasts at an origin must be bit-identical when all data after
        that origin is replaced with garbage."""
        cutoff = 700  # an origin index inside the evaluation span
        clean = returns.iloc[: cutoff + 1]

        corrupted = returns.copy()
        corrupted.iloc[cutoff + 1 :] = 0.5  # absurd +65% daily log-returns

        for model in (Ewma(), HistoricalSimulation(), Garch(dist="normal")):
            res_clean = run_walkforward(model, clean, CFG)
            res_corrupt = run_walkforward(model, corrupted, CFG)
            a = res_clean.forecasts
            b = res_corrupt.forecasts
            b = b[b.origin <= clean.index[-1]]
            for col in ("sigma2", "var", "es"):
                np.testing.assert_array_equal(
                    a[col].to_numpy(), b[col].to_numpy(),
                    err_msg=f"{model.name}: future data leaked into {col}",
                )

    def test_oos_start_respected(self, returns):
        start = pd.Timestamp("2017-06-01")
        res = run_walkforward(Ewma(), returns, WalkForwardConfig(
            window=500, refit_every=63, horizons=(1,), alphas=(0.05,), start=start,
        ))
        assert res.forecasts.origin.min() >= start

    def test_duplicate_index_rejected(self):
        idx = pd.DatetimeIndex(["2020-01-01"] * 300 + ["2020-01-02"] * 300)
        r = pd.Series(0.001, index=idx)
        with pytest.raises(ValueError, match="unique and sorted"):
            run_walkforward(Ewma(), r, CFG)

    def test_run_many_tags_series_and_models(self, returns):
        df = run_many(
            [Ewma(), HistoricalSimulation()],
            {"A": returns, "B": returns * 1.5},
            CFG,
        )
        assert set(df.series.unique()) == {"A", "B"}
        assert set(df.model.unique()) == {"ewma-rm", "historical"}


class TestSanityOnGarchData:
    def test_var_coverage_near_nominal(self, returns):
        """On simulated GARCH data, 5% VaR breach frequency should be
        within a loose band of nominal for a correctly specified model."""
        res = run_walkforward(Garch(dist="normal"), returns, CFG)
        df = res.forecasts
        sel = df[(df.horizon == 1) & (df.alpha == 0.05)].dropna(subset=["realized"])
        breach_rate = (sel.realized < -sel["var"]).mean()
        assert 0.02 <= breach_rate <= 0.09
