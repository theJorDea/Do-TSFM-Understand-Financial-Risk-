"""Losses and backtests validated on synthetic data with known truth:
consistency of scoring rules at the true forecast, and Monte-Carlo
size/power of the tests roughly matching the published values.
"""

import numpy as np
import pytest

from tsfm_risk.evaluation.backtests import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
    engle_manganelli_dq,
    kupiec_pof,
)
from tsfm_risk.evaluation.losses import fz0, mse_variance, qlike
from tsfm_risk.risk.measures import var_es_normal
from tsfm_risk.simulate import simulate_garch

ALPHA = 0.05


class TestQlike:
    def test_minimized_at_true_variance(self):
        rng = np.random.default_rng(0)
        true_s2 = 4.0
        proxy = rng.normal(0, np.sqrt(true_s2), 500_000) ** 2
        candidates = np.array([2.0, 3.0, 3.8, 4.0, 4.2, 5.0, 8.0])
        avg = [qlike(np.full_like(proxy, c), proxy).mean() for c in candidates]
        assert candidates[int(np.argmin(avg))] == true_s2

    def test_defined_at_zero_proxy(self):
        out = qlike(np.array([1.0]), np.array([0.0]))
        assert np.isfinite(out).all()

    def test_rejects_nonpositive_forecast(self):
        with pytest.raises(ValueError):
            qlike(np.array([0.0]), np.array([1.0]))

    def test_mse_shape(self):
        assert mse_variance(np.ones(3), np.zeros(3)).shape == (3,)


class TestFz0:
    def test_minimized_at_true_var_es_pair(self):
        """Strict consistency: the true (VaR, ES) of a standard normal must
        beat scaled distortions of the pair in expected loss."""
        rng = np.random.default_rng(1)
        r = rng.standard_normal(1_000_000)
        var_true, es_true = var_es_normal(1.0, ALPHA)

        def avg_loss(scale: float) -> float:
            v = np.full_like(r, var_true * scale)
            e = np.full_like(r, es_true * scale)
            return float(fz0(v, e, r, ALPHA).mean())

        losses = {s: avg_loss(s) for s in (0.7, 0.85, 1.0, 1.15, 1.4)}
        assert min(losses, key=losses.get) == 1.0

    def test_penalizes_es_distortion_alone(self):
        rng = np.random.default_rng(2)
        r = rng.standard_normal(1_000_000)
        var_true, es_true = var_es_normal(1.0, ALPHA)
        v = np.full_like(r, var_true)
        good = fz0(v, np.full_like(r, es_true), r, ALPHA).mean()
        bad = fz0(v, np.full_like(r, es_true * 1.5), r, ALPHA).mean()
        assert good < bad

    def test_rejects_es_below_var(self):
        with pytest.raises(ValueError, match="ES must be >= VaR"):
            fz0(np.array([2.0]), np.array([1.0]), np.array([0.0]), ALPHA)


def _iid_normal_var_series(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n)
    var_true, _ = var_es_normal(1.0, ALPHA)
    return r, np.full(n, var_true)


class TestKupiec:
    def test_size_close_to_nominal(self):
        rejections = 0
        n_reps = 400
        for i in range(n_reps):
            r, v = _iid_normal_var_series(1000, seed=100 + i)
            if kupiec_pof(r, v, ALPHA).p_value < 0.05:
                rejections += 1
        assert 0.02 <= rejections / n_reps <= 0.09  # nominal 5%

    def test_power_against_undercoverage(self):
        rejections = 0
        n_reps = 200
        for i in range(n_reps):
            r, v = _iid_normal_var_series(1000, seed=500 + i)
            if kupiec_pof(r, v * 0.7, ALPHA).p_value < 0.05:  # VaR too small
                rejections += 1
        assert rejections / n_reps > 0.9

    def test_zero_breaches_handled(self):
        r, v = _iid_normal_var_series(500, seed=9)
        res = kupiec_pof(r, v * 100, ALPHA)  # absurdly conservative VaR
        assert res.n_breaches == 0 and res.p_value < 0.05


class TestChristoffersen:
    def test_detects_clustered_breaches(self):
        """GARCH returns against a constant VaR produce clustered breaches;
        the independence test must reject far above its size."""
        rejections = 0
        n_reps = 100
        for i in range(n_reps):
            r = simulate_garch(1500, alpha=0.15, beta=0.83, seed=700 + i) * 100
            v = np.full(r.size, -np.quantile(r, ALPHA))
            if christoffersen_independence(r, v).p_value < 0.05:
                rejections += 1
        assert rejections / n_reps > 0.5

    def test_cc_combines_pof_and_independence(self):
        r, v = _iid_normal_var_series(1000, seed=11)
        cc = christoffersen_conditional_coverage(r, v, ALPHA)
        pof = kupiec_pof(r, v, ALPHA)
        ind = christoffersen_independence(r, v)
        np.testing.assert_allclose(cc.statistic, pof.statistic + ind.statistic)


class TestDq:
    def test_size_close_to_nominal(self):
        rejections = 0
        n_reps = 300
        for i in range(n_reps):
            r, v = _iid_normal_var_series(1000, seed=900 + i)
            if engle_manganelli_dq(r, v, ALPHA).p_value < 0.05:
                rejections += 1
        assert 0.01 <= rejections / n_reps <= 0.10

    def test_power_against_garch_misspecification(self):
        rejections = 0
        n_reps = 100
        for i in range(n_reps):
            r = simulate_garch(1500, alpha=0.15, beta=0.83, seed=1300 + i) * 100
            v = np.full(r.size, -np.quantile(r, ALPHA))
            if engle_manganelli_dq(r, v, ALPHA).p_value < 0.05:
                rejections += 1
        assert rejections / n_reps > 0.6
