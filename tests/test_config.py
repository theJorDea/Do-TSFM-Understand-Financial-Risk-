import datetime as dt
from pathlib import Path

import pytest
from pydantic import ValidationError

from tsfm_risk.config import RegimesConfig, load_regimes, load_universe

ROOT = Path(__file__).resolve().parents[1]


def test_universe_config_loads_and_validates():
    cfg = load_universe(ROOT / "configs" / "universe.yaml")
    pairs = cfg.all_series()
    assert len(pairs) >= 20
    asset_classes = {ac for ac, _ in pairs}
    assert {"equity_index", "fx", "crypto", "commodity"} <= asset_classes
    tickers = [s.ticker for _, s in pairs]
    assert len(tickers) == len(set(tickers)), "duplicate tickers in universe"


def test_regimes_config_loads_and_classifies():
    cfg = load_regimes(ROOT / "configs" / "regimes.yaml")
    assert len(cfg.crisis_windows) >= 6
    assert cfg.is_crisis(dt.date(2020, 3, 16))  # COVID crash
    assert cfg.is_crisis(dt.date(2008, 10, 10))  # GFC
    assert not cfg.is_crisis(dt.date(2017, 6, 15))  # calm 2017
    assert not cfg.is_crisis(dt.date(2013, 5, 1))


def test_overlapping_regime_windows_rejected():
    with pytest.raises(ValidationError, match="overlap"):
        RegimesConfig.model_validate(
            {
                "crisis_windows": [
                    {"name": "a", "start": "2020-01-01", "end": "2020-06-30"},
                    {"name": "b", "start": "2020-06-01", "end": "2020-12-31"},
                ]
            }
        )
