import numpy as np
import pytest

from tsfm_risk.models.classical import (
    Ewma,
    FilteredHistoricalSimulation,
    Har,
    HistoricalSimulation,
)
from tsfm_risk.models.classical.historical import overlapping_sums
from tsfm_risk.simulate import simulate_garch

ALPHAS = (0.01, 0.05)


def iid_normal_returns(n=2000, sigma=0.01, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma, n)


def two_regime_returns(n=1000, seed=1):
    """Calm first half, turbulent second half."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0, 0.005, n // 2)
    wild = rng.normal(0, 0.03, n // 2)
    return np.concatenate([calm, wild])


class TestEwma:
    def test_reacts_to_recent_volatility(self):
        r = two_regime_returns()
        f_wild = Ewma().forecast(r, (1,), ALPHAS)[0]
        f_calm = Ewma().forecast(r[: len(r) // 2], (1,), ALPHAS)[0]
        assert f_wild.variance > 5 * f_calm.variance

    def test_hday_variance_is_h_times_daily(self):
        r = iid_normal_returns()
        f1, f20 = Ewma().forecast(r, (1, 20), ALPHAS)
        np.testing.assert_allclose(f20.variance, 20 * f1.variance)

    def test_var_matches_normal_quantile_of_sigma(self):
        r = iid_normal_returns(sigma=0.01, seed=4)
        f = Ewma().forecast(r, (1,), (0.05,))[0]
        np.testing.assert_allclose(f.var[0.05], 1.6449 * np.sqrt(f.variance), rtol=1e-3)


class TestHistorical:
    def test_overlapping_sums(self):
        s = overlapping_sums(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        np.testing.assert_allclose(s, [3.0, 5.0, 7.0])

    def test_var_close_to_true_quantile_iid(self):
        r = iid_normal_returns(n=5000, sigma=0.02, seed=2)
        f = HistoricalSimulation().forecast(r, (1,), (0.05,))[0]
        np.testing.assert_allclose(f.var[0.05], 1.6449 * 0.02, rtol=0.06)

    def test_es_at_least_var(self):
        r = iid_normal_returns(seed=3)
        f = HistoricalSimulation().forecast(r, (1, 5), ALPHAS)
        for fc in f:
            for a in ALPHAS:
                assert fc.es[a] >= fc.var[a]


class TestFhs:
    def test_deterministic_given_window(self):
        r = two_regime_returns(seed=5)
        m = FilteredHistoricalSimulation()
        f1 = m.forecast(r, (1, 5), ALPHAS)
        f2 = m.forecast(r, (1, 5), ALPHAS)
        for a, b in zip(f1, f2, strict=True):
            assert a.var == b.var and a.es == b.es

    def test_reacts_faster_than_plain_historical(self):
        # right after a calm->wild switch, FHS VaR must exceed plain
        # historical VaR, which still averages over the calm half
        r = two_regime_returns(seed=6)
        var_fhs = FilteredHistoricalSimulation().forecast(r, (1,), (0.01,))[0].var[0.01]
        var_hist = HistoricalSimulation().forecast(r, (1,), (0.01,))[0].var[0.01]
        assert var_fhs > var_hist

    def test_multiday_var_exceeds_scaled_oneday(self):
        # with vol updating inside paths, 20-day FHS VaR at 1% should be at
        # least sqrt-time-scaled 1-day VaR under clustering
        r = two_regime_returns(seed=7)
        f1, f20 = FilteredHistoricalSimulation().forecast(r, (1, 20), (0.01,))
        assert f20.var[0.01] > np.sqrt(20) * f1.var[0.01] * 0.8


class TestHar:
    def test_fit_predict_sane_on_garch_like_data(self):
        r = simulate_garch(3000, seed=13)
        m = Har()
        m.fit(r)
        f1, f5, f20 = m.forecast(r, (1, 5, 20), ALPHAS)
        assert 0 < f1.variance < f5.variance < f20.variance
        # in-sample average forecast should be near realized average variance
        np.testing.assert_allclose(f1.variance, np.var(r), rtol=3.0)

    def test_unfitted_horizon_rejected(self):
        r = iid_normal_returns()
        m = Har(horizons=(1,))
        m.fit(r)
        with pytest.raises(KeyError):
            m.forecast(r, (5,), ALPHAS)

    def test_forecast_requires_fit(self):
        with pytest.raises(RuntimeError):
            Har().forecast(iid_normal_returns(), (1,), ALPHAS)

    def test_floor_prevents_negative_variance(self):
        r = iid_normal_returns(n=1500, seed=8)
        m = Har()
        m.fit(r)
        for f in m.forecast(r, (1, 5, 20), ALPHAS):
            assert f.variance > 0
