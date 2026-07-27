"""The cache-to-forecast-table bridge, with the leakage guard the project relies on."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tsfm_risk.models.tsfm.cache import append_forecasts, cache_path
from tsfm_risk.models.tsfm.cached_model import build_all, forecasts_from_cache

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MODEL = "toy-model"
SERIES = "TEST"


def _make_cache(tmp_path, n=1400, sigma=0.012, days=20, seed=0, cached_horizon=20):
    """Write a cache of oracle-normal forecasts and return the matching returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    returns = pd.Series(rng.normal(0, sigma, n), index=dates)

    q = np.tile(stats.norm.ppf(LEVELS) * sigma, (n, days, 1))
    path = cache_path(MODEL, SERIES, cached_horizon, 1024, root=tmp_path)
    append_forecasts(path, list(dates), LEVELS, q)
    return returns


class TestFrameShape:
    def test_matches_the_baseline_schema(self, tmp_path):
        r = _make_cache(tmp_path)
        df = forecasts_from_cache(MODEL, SERIES, r, root=tmp_path)
        assert set(df.columns) == {
            "series", "origin", "model", "horizon", "alpha",
            "sigma2", "var", "es", "realized",
        }
        assert set(df.horizon.unique()) == {1, 5, 20}
        assert set(np.round(df.alpha.unique(), 4)) == {0.01, 0.025, 0.05}
        assert df.model.iloc[0] == f"{MODEL}-vol"

    def test_realized_alignment_is_forward_looking(self, tmp_path):
        r = _make_cache(tmp_path)
        df = forecasts_from_cache(MODEL, SERIES, r, root=tmp_path)
        row = df[(df.horizon == 5) & (df.alpha == 0.05)].iloc[10]
        t = r.index.get_loc(row.origin)
        np.testing.assert_allclose(row.realized, r.iloc[t + 1 : t + 6].sum())

    def test_missing_cache_raises(self, tmp_path):
        r = _make_cache(tmp_path)
        with pytest.raises(FileNotFoundError):
            forecasts_from_cache("absent", SERIES, r, root=tmp_path)

    def test_short_cache_rejected(self, tmp_path):
        r = _make_cache(tmp_path, days=5, cached_horizon=5)
        with pytest.raises(ValueError, match="forecast days"):
            forecasts_from_cache(MODEL, SERIES, r, root=tmp_path,
                                 cached_horizon=5, horizons=(1, 5, 20))


class TestVolPathValues:
    def test_recovers_oracle_variance(self, tmp_path):
        """Forecasts are exact normal quantiles of a known sigma, so the
        recovered one-day variance must equal sigma^2."""
        sigma = 0.012
        r = _make_cache(tmp_path, sigma=sigma)
        df = forecasts_from_cache(MODEL, SERIES, r, root=tmp_path)
        one_day = df[(df.horizon == 1) & (df.alpha == 0.05)]
        np.testing.assert_allclose(one_day.sigma2.to_numpy(), sigma**2, rtol=1e-6)

    def test_variance_scales_with_horizon(self, tmp_path):
        r = _make_cache(tmp_path)
        df = forecasts_from_cache(MODEL, SERIES, r, root=tmp_path)
        sel = df[df.alpha == 0.05].set_index(["origin", "horizon"]).sigma2
        o = df.origin.iloc[0]
        np.testing.assert_allclose(sel[(o, 5)], 5 * sel[(o, 1)], rtol=1e-9)
        np.testing.assert_allclose(sel[(o, 20)], 20 * sel[(o, 1)], rtol=1e-9)

    def test_es_never_below_var(self, tmp_path):
        r = _make_cache(tmp_path)
        df = forecasts_from_cache(MODEL, SERIES, r, root=tmp_path)
        assert (df.es >= df["var"] - 1e-12).all()

    def test_coverage_close_to_nominal_on_matched_data(self, tmp_path):
        """Returns really are normal with the forecast sigma, so breach rates
        should land near nominal — a end-to-end check of the whole chain."""
        r = _make_cache(tmp_path, n=3000, seed=5)
        df = forecasts_from_cache(MODEL, SERIES, r, root=tmp_path)
        sel = df[(df.horizon == 1) & (df.alpha == 0.05)].dropna(subset=["realized"])
        rate = (sel.realized < -sel["var"]).mean()
        assert 0.03 <= rate <= 0.07


class TestNuEstimationIsLeakageFree:
    def test_future_returns_cannot_change_past_forecasts(self, tmp_path):
        """The decisive test: corrupt every return after a cutoff and the
        forecasts at earlier origins must be bit-identical. Since nu is fitted
        on standardized residuals, a wrong pairing of returns to forecasts would
        leak tomorrow into today and break this."""
        r = _make_cache(tmp_path, n=1600, seed=11)
        cutoff = 1200

        clean = forecasts_from_cache(MODEL, SERIES, r, root=tmp_path)
        corrupted_returns = r.copy()
        corrupted_returns.iloc[cutoff:] = 0.5  # absurd +65% daily moves
        dirty = forecasts_from_cache(MODEL, SERIES, corrupted_returns, root=tmp_path)

        keep = r.index[cutoff - 1]
        a = clean[clean.origin <= keep].sort_values(["origin", "horizon", "alpha"])
        b = dirty[dirty.origin <= keep].sort_values(["origin", "horizon", "alpha"])
        for col in ("sigma2", "var", "es"):
            np.testing.assert_array_equal(
                a[col].to_numpy(), b[col].to_numpy(),
                err_msg=f"future data leaked into {col}",
            )

    def test_fat_tailed_returns_widen_the_tail(self, tmp_path):
        """nu really is being estimated: with the SAME forecast scale, Student-t
        returns must produce a wider tail than normal returns.

        Thresholds come from the closed forms, not from guesswork. At alpha=1%
        with nu=3.5 the exact ratios are VaR 1.14x and ES 1.45x — ES reacts far
        more strongly because it integrates the whole tail rather than reading
        one quantile. (That asymmetry is also the regulatory argument for
        switching from VaR to ES.) Bounds are set below the exact values to
        leave room for sampling error in the fitted nu.
        """
        sigma, n = 0.012, 3000
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2015-01-01", periods=n)
        q = np.tile(stats.norm.ppf(LEVELS) * sigma, (n, 20, 1))
        append_forecasts(cache_path(MODEL, SERIES, 20, 1024, root=tmp_path),
                         list(dates), LEVELS, q)

        normal = pd.Series(rng.normal(0, sigma, n), index=dates)
        nu_true = 3.5
        heavy = pd.Series(
            rng.standard_t(nu_true, n) * np.sqrt((nu_true - 2) / nu_true) * sigma, index=dates
        )

        def last_tail(d, col):
            return d[(d.horizon == 1) & (d.alpha == 0.01)][col].iloc[-1]

        v_norm = forecasts_from_cache(MODEL, SERIES, normal, root=tmp_path)
        v_heavy = forecasts_from_cache(MODEL, SERIES, heavy, root=tmp_path)

        var_ratio = last_tail(v_heavy, "var") / last_tail(v_norm, "var")
        es_ratio = last_tail(v_heavy, "es") / last_tail(v_norm, "es")
        assert var_ratio > 1.08, f"VaR почти не расширился: {var_ratio:.3f}"
        assert es_ratio > 1.25, f"ES почти не расширился: {es_ratio:.3f}"
        assert es_ratio > var_ratio, "ES обязан реагировать сильнее VaR"

    def test_warmup_uses_normal_innovations(self, tmp_path, caplog):
        r = _make_cache(tmp_path, n=1400)
        with caplog.at_level("INFO"):
            forecasts_from_cache(MODEL, SERIES, r, root=tmp_path, min_residuals=250)
        assert any("warm-up" in rec.message for rec in caplog.records)


class TestDirectPath:
    def test_refuses_alphas_outside_the_cached_grid(self, tmp_path):
        """Chronos-Bolt's grid stops at 0.1; asking for 1% VaR must fail loudly
        rather than return a clamped number."""
        r = _make_cache(tmp_path)
        with pytest.raises(ValueError, match="outside"):
            forecasts_from_cache(MODEL, SERIES, r, root=tmp_path,
                                 path_name="direct", horizons=(1,), alphas=(0.01,))

    def test_works_when_the_grid_reaches_alpha(self, tmp_path):
        r = _make_cache(tmp_path)
        df = forecasts_from_cache(MODEL, SERIES, r, root=tmp_path,
                                  path_name="direct", horizons=(1,), alphas=(0.1,))
        assert df.model.iloc[0] == f"{MODEL}-direct"
        assert (df["var"] > 0).all()


class TestBuildAll:
    def test_skips_series_without_cache(self, tmp_path, caplog):
        r = _make_cache(tmp_path)
        with caplog.at_level("WARNING"):
            df = build_all(MODEL, {SERIES: r, "MISSING": r}, root=tmp_path)
        assert set(df.series.unique()) == {SERIES}
        assert any("no cache" in rec.message for rec in caplog.records)
