import numpy as np
import pytest
from scipy import stats

from tsfm_risk.risk.measures import (
    var_es_empirical,
    var_es_from_quantiles,
    var_es_normal,
    var_es_student_t,
)


class TestNormal:
    def test_var_99_matches_textbook(self):
        var, _ = var_es_normal(sigma=1.0, alpha=0.01)
        np.testing.assert_allclose(var, 2.3263, atol=1e-4)

    def test_es_975_close_to_var_99_basel(self):
        # Basel FRTB calibration: ES at 2.5% roughly matches VaR at 1% for normal
        _, es = var_es_normal(sigma=1.0, alpha=0.025)
        var99, _ = var_es_normal(sigma=1.0, alpha=0.01)
        np.testing.assert_allclose(es, var99, rtol=0.01)

    def test_monte_carlo_agreement(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.001, 0.02, 2_000_000)
        var_mc, es_mc = var_es_empirical(x, 0.05)
        var, es = var_es_normal(sigma=0.02, alpha=0.05, mu=0.001)
        np.testing.assert_allclose(var, var_mc, rtol=5e-3)
        np.testing.assert_allclose(es, es_mc, rtol=5e-3)


class TestStudentT:
    def test_converges_to_normal_for_large_nu(self):
        var_t, es_t = var_es_student_t(sigma=1.0, nu=1e6, alpha=0.01)
        var_n, es_n = var_es_normal(sigma=1.0, alpha=0.01)
        np.testing.assert_allclose([var_t, es_t], [var_n, es_n], rtol=1e-3)

    def test_fat_tails_exceed_normal(self):
        var_t, es_t = var_es_student_t(sigma=1.0, nu=4.0, alpha=0.01)
        var_n, es_n = var_es_normal(sigma=1.0, alpha=0.01)
        assert var_t > var_n
        assert es_t > es_n

    def test_monte_carlo_agreement(self):
        nu, sigma = 5.0, 0.015
        rng = np.random.default_rng(11)
        x = sigma * np.sqrt((nu - 2) / nu) * rng.standard_t(nu, 4_000_000)
        var_mc, es_mc = var_es_empirical(x, 0.01)
        var, es = var_es_student_t(sigma=sigma, nu=nu, alpha=0.01)
        np.testing.assert_allclose(var, var_mc, rtol=1e-2)
        np.testing.assert_allclose(es, es_mc, rtol=1e-2)

    def test_rejects_infinite_variance(self):
        with pytest.raises(ValueError, match="nu"):
            var_es_student_t(sigma=1.0, nu=2.0, alpha=0.05)


class TestEmpirical:
    def test_es_at_least_var(self):
        rng = np.random.default_rng(3)
        x = rng.standard_t(4, 10_000) * 0.01
        var, es = var_es_empirical(x, 0.05)
        assert es >= var > 0

    def test_small_sample_rejected(self):
        with pytest.raises(ValueError, match="observations"):
            var_es_empirical(np.zeros(50), 0.05)


class TestFromQuantiles:
    def test_recovers_normal_on_dense_grid(self):
        levels = np.linspace(0.001, 0.999, 999)
        values = stats.norm.ppf(levels, scale=0.02)
        var, es = var_es_from_quantiles(levels, values, 0.05)
        var_true, es_true = var_es_normal(sigma=0.02, alpha=0.05)
        np.testing.assert_allclose(var, var_true, rtol=1e-3)
        np.testing.assert_allclose(es, es_true, rtol=2e-2)

    def test_alpha_outside_grid_rejected(self):
        levels = np.array([0.05, 0.5, 0.95])
        values = np.array([-0.03, 0.0, 0.03])
        with pytest.raises(ValueError, match="outside"):
            var_es_from_quantiles(levels, values, 0.01)

    def test_non_monotone_quantiles_rejected(self):
        levels = np.array([0.01, 0.05, 0.5])
        values = np.array([-0.02, -0.03, 0.0])
        with pytest.raises(ValueError, match="non-decreasing"):
            var_es_from_quantiles(levels, values, 0.05)
