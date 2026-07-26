import numpy as np
import pytest

from tsfm_risk.evaluation.es_backtest import acerbi_szekely_z2, implied_nu
from tsfm_risk.risk.measures import var_es_normal, var_es_student_t

ALPHA = 0.025


class TestImpliedNu:
    def test_roundtrip_recovers_nu(self):
        for nu in (4.0, 6.0, 12.0):
            var, es = var_es_student_t(1.0, nu, ALPHA)
            got = implied_nu(var, es, ALPHA)
            np.testing.assert_allclose(got, nu, rtol=1e-4)

    def test_gaussian_ratio_maps_to_none(self):
        var, es = var_es_normal(1.0, ALPHA)
        assert implied_nu(var, es, ALPHA) is None

    def test_rejects_es_not_above_var(self):
        with pytest.raises(ValueError):
            implied_nu(2.0, 2.0, ALPHA)


class TestAcerbiSzekelyZ2:
    def _correct_forecasts(self, n: int, seed: int, nu: float = 5.0):
        rng = np.random.default_rng(seed)
        r = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu) * 0.01
        var, es = var_es_student_t(0.01, nu, ALPHA)
        return r, np.full(n, var), np.full(n, es)

    def test_correct_model_not_rejected_usually(self):
        rejections = 0
        n_reps = 40
        for i in range(n_reps):
            r, v, e = self._correct_forecasts(2000, seed=i)
            res = acerbi_szekely_z2(r, v, e, ALPHA, n_sim=500, seed=i)
            if res.p_value < 0.05:
                rejections += 1
        assert rejections / n_reps <= 0.15

    def test_understated_es_rejected(self):
        rejections = 0
        n_reps = 30
        for i in range(n_reps):
            r, v, e = self._correct_forecasts(2000, seed=200 + i)
            res = acerbi_szekely_z2(r, v * 0.75, np.maximum(e * 0.75, v * 0.76), ALPHA,
                                    n_sim=500, seed=i)
            if res.p_value < 0.05:
                rejections += 1
        assert rejections / n_reps > 0.85

    def test_z2_near_zero_for_correct_model(self):
        r, v, e = self._correct_forecasts(50_000, seed=77)
        res = acerbi_szekely_z2(r, v, e, ALPHA, n_sim=200, seed=1)
        assert abs(res.statistic) < 0.15

    def test_deterministic_given_seed(self):
        r, v, e = self._correct_forecasts(1000, seed=5)
        a = acerbi_szekely_z2(r, v, e, ALPHA, n_sim=300, seed=3)
        b = acerbi_szekely_z2(r, v, e, ALPHA, n_sim=300, seed=3)
        assert a.statistic == b.statistic and a.p_value == b.p_value
