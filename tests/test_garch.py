"""GARCH validation: synthetic parameter recovery and agreement with `arch`."""

import numpy as np
import pytest

from tsfm_risk.models.classical.garch import Garch, _filter_variance
from tsfm_risk.simulate import simulate_garch


class TestParameterRecovery:
    def test_garch_normal(self):
        m = Garch(dist="normal")
        m.fit(simulate_garch(20_000))
        p = m.params
        assert p is not None and p.converged
        np.testing.assert_allclose(p.alpha, 0.08, atol=0.015)
        np.testing.assert_allclose(p.beta, 0.90, atol=0.02)
        np.testing.assert_allclose(p.persistence, 0.98, atol=0.01)

    def test_garch_student_t_recovers_nu(self):
        m = Garch(dist="t")
        m.fit(simulate_garch(20_000, nu=6.0, seed=7))
        p = m.params
        assert p is not None and p.converged
        np.testing.assert_allclose(p.nu, 6.0, rtol=0.25)
        np.testing.assert_allclose(p.alpha, 0.08, atol=0.02)

    def test_gjr_recovers_leverage(self):
        m = Garch(dist="normal", leverage=True)
        m.fit(simulate_garch(20_000, alpha=0.03, gamma=0.10, beta=0.88, seed=3))
        p = m.params
        assert p is not None and p.converged
        np.testing.assert_allclose(p.gamma, 0.10, atol=0.03)
        assert p.gamma > p.alpha  # leverage dominates, as simulated


@pytest.fixture(scope="module")
def data():
    return simulate_garch(5_000, seed=11)


class TestAgainstArch:

    def test_loglik_not_worse_than_arch(self, data):
        from arch import arch_model

        ours = Garch(dist="normal")
        ours.fit(data)

        am = arch_model(data * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        theirs = am.fit(disp="off")

        # different variance initialization -> evaluate both parameter sets
        # under OUR recursion; our MLE must not be beaten on our own likelihood
        r = data * 100

        def our_nll(omega, alpha, beta):
            s2, _ = _filter_variance(r, omega, alpha, beta, 0.0, float(np.var(r)))
            return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + r * r / s2)

        p = ours.params
        nll_ours = our_nll(p.omega, p.alpha, p.beta)
        nll_arch = our_nll(
            theirs.params["omega"], theirs.params["alpha[1]"], theirs.params["beta[1]"]
        )
        assert nll_ours <= nll_arch + 0.5  # within numerical slack

    def test_params_close_to_arch(self, data):
        from arch import arch_model

        ours = Garch(dist="normal")
        ours.fit(data)
        theirs = arch_model(
            data * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="normal"
        ).fit(disp="off")

        np.testing.assert_allclose(ours.params.alpha, theirs.params["alpha[1]"], atol=0.01)
        np.testing.assert_allclose(ours.params.beta, theirs.params["beta[1]"], atol=0.015)

    def test_one_step_forecast_close_to_arch(self, data):
        from arch import arch_model

        ours = Garch(dist="normal")
        ours.fit(data)
        f = ours.forecast(data, horizons=(1,), alphas=(0.05,))[0]

        theirs = arch_model(
            data * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="normal"
        ).fit(disp="off")
        arch_s2 = theirs.forecast(horizon=1, reindex=False).variance.iloc[0, 0] / 100**2

        np.testing.assert_allclose(f.variance, arch_s2, rtol=0.03)


class TestForecastProperties:
    def test_multistep_variance_grows_and_mean_reverts(self):
        data = simulate_garch(3_000, seed=5)
        m = Garch(dist="normal")
        m.fit(data)
        f1, f5, f20 = m.forecast(data, horizons=(1, 5, 20), alphas=(0.01,))
        assert f1.variance < f5.variance < f20.variance  # aggregation grows
        # per-day variance approaches unconditional level
        p = m.params
        uncond_daily = p.omega / (1 - p.persistence) / 100**2
        per_day_20 = f20.variance / 20
        per_day_1 = f1.variance
        assert abs(per_day_20 - uncond_daily) <= abs(per_day_1 - uncond_daily) + 1e-12

    def test_t_var_exceeds_normal_var_at_1pct(self):
        data = simulate_garch(3_000, nu=5.0, seed=9)
        mn, mt = Garch(dist="normal"), Garch(dist="t")
        mn.fit(data)
        mt.fit(data)
        vn = mn.forecast(data, (1,), (0.01,))[0].var[0.01]
        vt = mt.forecast(data, (1,), (0.01,))[0].var[0.01]
        assert vt > vn

    def test_forecast_requires_fit(self):
        with pytest.raises(RuntimeError, match="fit"):
            Garch().forecast(np.zeros(300) + 0.001, (1,), (0.05,))
