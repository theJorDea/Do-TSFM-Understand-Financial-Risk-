import numpy as np
import pandas as pd
import pytest

from tsfm_risk.data.returns import (
    aggregate_returns,
    clean_returns,
    log_returns,
    squared_return_proxy,
)


def _prices(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, name="TEST")


def test_log_returns_basic():
    r = log_returns(_prices([100.0, 110.0, 99.0]))
    assert len(r) == 2
    np.testing.assert_allclose(r.iloc[0], np.log(1.1))
    np.testing.assert_allclose(r.iloc[1], np.log(99 / 110))


def test_log_returns_drops_nonpositive():
    r = log_returns(_prices([100.0, -5.0, 110.0]))
    np.testing.assert_allclose(r.iloc[0], np.log(1.1))
    assert len(r) == 1


def test_squared_proxy_is_square():
    r = log_returns(_prices([100.0, 105.0, 95.0]))
    np.testing.assert_allclose(squared_return_proxy(r).values, r.values**2)


def test_aggregate_returns_sums_forward_window():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 50), index=pd.bdate_range("2020-01-01", periods=50))
    h = 5
    agg = aggregate_returns(r, h)
    # value at origin t must equal the sum of returns over (t, t+h]
    t = r.index[10]
    np.testing.assert_allclose(agg.loc[t], r.iloc[11 : 11 + h].sum())
    # no forward window -> those dates are absent (leakage guard)
    assert r.index[-1] not in agg.index
    assert len(agg) == 50 - h


def test_aggregate_returns_h1_is_next_day_return():
    r = pd.Series([0.01, -0.02, 0.03], index=pd.bdate_range("2020-01-01", periods=3))
    agg = aggregate_returns(r, 1)
    np.testing.assert_allclose(agg.values, [-0.02, 0.03])


def test_aggregate_rejects_bad_horizon():
    r = pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1))
    with pytest.raises(ValueError):
        aggregate_returns(r, 0)


def test_clean_returns_keeps_crashes_drops_garbage():
    r = pd.Series([-0.30, 0.02, 5.0], index=pd.bdate_range("2020-01-01", periods=3))
    cleaned = clean_returns(r)
    assert -0.30 in cleaned.values  # real crash survives
    assert 5.0 not in cleaned.values  # corrupt tick removed
