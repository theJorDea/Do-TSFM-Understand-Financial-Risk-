import numpy as np
import pytest

from tsfm_risk.evaluation.comparison import (
    diebold_mariano,
    model_confidence_set,
    newey_west_variance,
)


class TestDieboldMariano:
    def test_size_under_equal_ability(self):
        rejections = 0
        n_reps = 400
        for i in range(n_reps):
            rng = np.random.default_rng(i)
            base = rng.standard_normal(800)
            l1 = base + rng.standard_normal(800) * 0.5
            l2 = base + rng.standard_normal(800) * 0.5
            if diebold_mariano(l1, l2).p_value < 0.05:
                rejections += 1
        assert 0.02 <= rejections / n_reps <= 0.09

    def test_detects_dominated_model(self):
        rng = np.random.default_rng(42)
        l1 = rng.standard_normal(800)
        l2 = l1 + 0.2 + rng.standard_normal(800) * 0.3  # model 2 worse
        res = diebold_mariano(l1, l2)
        assert res.p_value < 0.001
        assert res.statistic < 0 and res.mean_diff < 0

    def test_hac_handles_overlap_autocorrelation(self):
        """h-day overlapping losses are MA(h-1); naive variance overstates
        significance. Check the HAC variance grows with the lag window."""
        rng = np.random.default_rng(7)
        e = rng.standard_normal(2000)
        d = np.convolve(e, np.ones(5) / 5, mode="valid")  # MA(4) series
        v0 = newey_west_variance(d, 0)
        v4 = newey_west_variance(d, 4)
        assert v4 > 2.0 * v0

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(ValueError):
            diebold_mariano(np.zeros(200), np.zeros(300))


class TestMCS:
    def _losses(self, n=1000, seed=0, worse: dict[str, float] | None = None):
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(n)
        out = {}
        for name in ("a", "b", "c", "d"):
            shift = (worse or {}).get(name, 0.0)
            out[name] = base + rng.standard_normal(n) * 0.5 + shift
        return out

    def test_equal_models_all_survive_mostly(self):
        survived_all = 0
        n_reps = 50
        for i in range(n_reps):
            res = model_confidence_set(self._losses(seed=i), n_boot=500, seed=i)
            if len(res.included) == 4:
                survived_all += 1
        assert survived_all / n_reps > 0.75  # 90% confidence, T_max is conservative

    def test_dominated_model_eliminated(self):
        eliminated = 0
        n_reps = 30
        for i in range(n_reps):
            res = model_confidence_set(
                self._losses(seed=100 + i, worse={"d": 0.3}), n_boot=500, seed=i
            )
            if "d" not in res.included:
                eliminated += 1
        assert eliminated / n_reps > 0.9

    def test_best_model_never_eliminated_when_dominant(self):
        res = model_confidence_set(
            self._losses(seed=5, worse={"b": 0.5, "c": 0.5, "d": 0.5}),
            n_boot=1000,
        )
        assert "a" in res.included
        assert res.p_values["a"] == max(res.p_values.values())

    def test_deterministic_given_seed(self):
        r1 = model_confidence_set(self._losses(seed=3), n_boot=300, seed=9)
        r2 = model_confidence_set(self._losses(seed=3), n_boot=300, seed=9)
        assert r1.included == r2.included and r1.p_values == r2.p_values

    def test_nan_rows_dropped_consistently(self):
        losses = self._losses(seed=8)
        losses["a"][:50] = np.nan
        res = model_confidence_set(losses, n_boot=300)
        assert set(res.p_values) == {"a", "b", "c", "d"}
