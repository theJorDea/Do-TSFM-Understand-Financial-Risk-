"""Adapters validated against oracle forecasters whose truth is known.

An "oracle" emits the exact quantiles of a known distribution, so the adapter
must recover that distribution's VaR/ES. This separates adapter bugs from model
quality before any real TSFM is involved.
"""

import numpy as np
import pytest
from scipy import stats

from tsfm_risk.models.tsfm.adapters import (
    direct_quantile_forecast,
    fit_student_t_nu,
    scale_from_quantiles,
    vol_path_forecast,
)
from tsfm_risk.models.tsfm.base import DEFAULT_LEVELS, TSFMForecaster
from tsfm_risk.risk.measures import var_es_normal, var_es_student_t

LEVELS = np.array(DEFAULT_LEVELS)


def normal_oracle(sigma: float, horizon: int = 1, levels=LEVELS) -> np.ndarray:
    return np.tile(stats.norm.ppf(levels) * sigma, (horizon, 1))


def student_oracle(sigma: float, nu: float, horizon: int = 1, levels=LEVELS) -> np.ndarray:
    scale = sigma * np.sqrt((nu - 2) / nu)
    return np.tile(stats.t.ppf(levels, nu) * scale, (horizon, 1))


class TestScaleFromQuantiles:
    def test_recovers_normal_sigma(self):
        for sigma in (0.005, 0.02, 0.1):
            got = scale_from_quantiles(LEVELS, normal_oracle(sigma)[0])
            np.testing.assert_allclose(got, sigma, rtol=1e-6)

    def test_ignores_tail_shape(self):
        """Central-region fit must not be inflated by fat tails: a t and a
        normal with the same central spread give the same scale."""
        nu = 4.0
        t_q = student_oracle(1.0, nu)[0]
        implied = scale_from_quantiles(LEVELS, t_q)
        # the t's own Gaussian-equivalent central scale, computed independently
        central = LEVELS[(LEVELS >= 0.1) & (LEVELS <= 0.9)]
        expected = np.polyfit(
            stats.norm.ppf(central),
            stats.t.ppf(central, nu) * np.sqrt((nu - 2) / nu),
            1,
        )[0]
        np.testing.assert_allclose(implied, expected, rtol=1e-6)
        assert implied < 1.0  # t has unit variance but a NARROWER centre

    def test_location_shift_does_not_change_scale(self):
        q = normal_oracle(0.02)[0]
        np.testing.assert_allclose(
            scale_from_quantiles(LEVELS, q + 0.01), scale_from_quantiles(LEVELS, q), rtol=1e-9
        )

    def test_rejects_too_few_central_levels(self):
        with pytest.raises(ValueError, match="within"):
            scale_from_quantiles(np.array([0.01, 0.99]), np.array([-2.0, 2.0]))

    def test_rejects_decreasing_quantiles(self):
        with pytest.raises(ValueError, match="non-positive"):
            scale_from_quantiles(LEVELS, -normal_oracle(0.02)[0])


class TestFitNu:
    def test_recovers_known_nu(self):
        for nu in (4.0, 8.0):
            rng = np.random.default_rng(int(nu))
            z = rng.standard_t(nu, 200_000) * np.sqrt((nu - 2) / nu)
            np.testing.assert_allclose(fit_student_t_nu(z), nu, rtol=0.1)

    def test_normal_residuals_give_large_nu(self):
        rng = np.random.default_rng(0)
        assert fit_student_t_nu(rng.standard_normal(200_000)) > 25

    def test_rejects_small_sample(self):
        with pytest.raises(ValueError, match="residuals"):
            fit_student_t_nu(np.zeros(10))


class TestVolPath:
    def test_recovers_normal_var_es(self):
        sigma = 0.015
        f = vol_path_forecast(LEVELS, normal_oracle(sigma), (1,), (0.01, 0.05))[0]
        for a in (0.01, 0.05):
            v, e = var_es_normal(sigma, a)
            np.testing.assert_allclose(f.var[a], v, rtol=1e-5)
            np.testing.assert_allclose(f.es[a], e, rtol=1e-5)

    def test_student_t_innovations_widen_tails(self):
        q = normal_oracle(0.015, horizon=1)
        f_n = vol_path_forecast(LEVELS, q, (1,), (0.01,))[0]
        f_t = vol_path_forecast(LEVELS, q, (1,), (0.01,), nu=4.0)[0]
        assert f_t.var[0.01] > f_n.var[0.01]
        assert f_t.es[0.01] > f_n.es[0.01]
        np.testing.assert_allclose(f_t.variance, f_n.variance)  # same vol, different tail

    def test_variance_aggregates_over_horizon(self):
        sigma = 0.01
        f1, f5, f20 = vol_path_forecast(LEVELS, normal_oracle(sigma, 20), (1, 5, 20), (0.05,))
        np.testing.assert_allclose(f5.variance, 5 * f1.variance, rtol=1e-9)
        np.testing.assert_allclose(f20.variance, 20 * f1.variance, rtol=1e-9)
        np.testing.assert_allclose(f20.var[0.05], np.sqrt(20) * f1.var[0.05], rtol=1e-9)

    def test_time_varying_scale_is_summed_not_scaled(self):
        """Aggregation must use the per-day scales, not h x the first day."""
        q = np.vstack([normal_oracle(0.01)[0], normal_oracle(0.03)[0]])
        f2 = vol_path_forecast(LEVELS, q, (2,), (0.05,))[0]
        np.testing.assert_allclose(f2.variance, 0.01**2 + 0.03**2, rtol=1e-9)

    def test_rejects_short_forecast_matrix(self):
        with pytest.raises(ValueError, match="forecast days"):
            vol_path_forecast(LEVELS, normal_oracle(0.01, 2), (5,), (0.05,))


class TestDirectPath:
    def test_recovers_student_t_var(self):
        sigma, nu = 0.02, 5.0
        f = direct_quantile_forecast(LEVELS, student_oracle(sigma, nu), (0.01, 0.05))
        for a in (0.01, 0.05):
            v, _ = var_es_student_t(sigma, nu, a)
            np.testing.assert_allclose(f.var[a], v, rtol=1e-6)

    def test_es_at_least_var(self):
        f = direct_quantile_forecast(LEVELS, student_oracle(0.02, 5.0), (0.01, 0.05))
        for a in (0.01, 0.05):
            assert f.es[a] >= f.var[a]

    def test_refuses_alpha_outside_native_grid(self):
        """The whole point of the path: no extrapolation past the lowest level."""
        coarse = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        q = stats.norm.ppf(coarse) * 0.02
        with pytest.raises(ValueError, match="outside"):
            direct_quantile_forecast(coarse, q, (0.01,))

    def test_two_paths_agree_when_model_is_gaussian(self):
        """If the predictive distribution really is normal, the direct path and
        the normal vol-path must give the same VaR — any gap is tail shape."""
        sigma = 0.018
        q = normal_oracle(sigma)
        direct = direct_quantile_forecast(LEVELS, q, (0.05,))
        volp = vol_path_forecast(LEVELS, q, (1,), (0.05,))[0]
        np.testing.assert_allclose(direct.var[0.05], volp.var[0.05], rtol=1e-3)

    def test_two_paths_diverge_when_model_is_fat_tailed(self):
        """A fat-tailed predictive distribution must show up as direct > vol-path
        under normal innovations. This gap is the RQ4 measurement."""
        q = student_oracle(0.018, nu=3.5)
        direct = direct_quantile_forecast(LEVELS, q, (0.01,))
        volp = vol_path_forecast(LEVELS, q, (1,), (0.01,))[0]
        assert direct.var[0.01] > 1.2 * volp.var[0.01]


class TestForecasterContract:
    class _Toy(TSFMForecaster):
        name = "toy"

        def predict_quantiles(self, context, horizon, levels=DEFAULT_LEVELS):
            x = self._validate(context, horizon, levels)
            sigma = float(np.std(x))
            return self._enforce_monotone(np.tile(stats.norm.ppf(levels) * sigma, (horizon, 1)))

    def test_shape_and_monotonicity(self):
        m = self._Toy()
        q = m.predict_quantiles(np.random.default_rng(0).normal(0, 0.01, 500), 5)
        assert q.shape == (5, len(DEFAULT_LEVELS))
        assert np.all(np.diff(q, axis=1) >= 0)

    def test_enforce_monotone_repairs_crossing(self):
        crossed = np.array([[-0.02, -0.03, 0.01]])
        fixed = TSFMForecaster._enforce_monotone(crossed)
        np.testing.assert_allclose(fixed, [[-0.02, -0.02, 0.01]])

    def test_rejects_non_finite_context(self):
        with pytest.raises(ValueError, match="non-finite"):
            self._Toy().predict_quantiles(np.array([0.01, np.nan, 0.02]), 1)

    def test_truncate_keeps_most_recent(self):
        class Short(self._Toy):
            max_context = 10

        x = np.arange(100.0)
        np.testing.assert_array_equal(Short().truncate_context(x), np.arange(90.0, 100.0))


class TestNativeLevelGuard:
    """Chronos-Bolt, asked for the 1% quantile, returns its clamped 10%
    quantile. A number that looks like a forecast but is not one must never
    reach the risk layer, so the guard has to be explicit."""

    class _Bolt(TSFMForecaster):
        name = "bolt-like"
        native_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

        def predict_quantiles(self, context, horizon, levels=DEFAULT_LEVELS):
            self.check_native(levels)
            x = self._validate(context, horizon, levels)
            return np.tile(stats.norm.ppf(levels) * float(np.std(x)), (horizon, 1))

    def test_refuses_non_native_levels(self):
        m = self._Bolt()
        with pytest.raises(ValueError, match="refusing to fabricate"):
            m.predict_quantiles(np.random.default_rng(0).normal(0, 0.01, 300), 1)

    def test_accepts_native_levels(self):
        m = self._Bolt()
        q = m.predict_quantiles(
            np.random.default_rng(0).normal(0, 0.01, 300), 1, levels=m.native_levels
        )
        assert q.shape == (1, 9)

    def test_levels_for_request_filters(self):
        m = self._Bolt()
        assert m.levels_for_request(DEFAULT_LEVELS) == m.native_levels

    def test_vol_path_still_works_on_native_grid(self):
        """The point of the guard: small alphas stay reachable via the
        vol-path, which only needs the central levels."""
        m = self._Bolt()
        lv = np.array(m.native_levels)
        q = m.predict_quantiles(np.random.default_rng(1).normal(0, 0.012, 500), 1, tuple(lv))
        f = vol_path_forecast(lv, q, (1,), (0.01,), nu=5.0)[0]
        assert f.var[0.01] > 0

    def test_unrestricted_model_allows_any_level(self):
        class Free(self._Bolt):
            native_levels = None

        q = Free().predict_quantiles(np.random.default_rng(2).normal(0, 0.01, 300), 1)
        assert q.shape == (1, len(DEFAULT_LEVELS))
