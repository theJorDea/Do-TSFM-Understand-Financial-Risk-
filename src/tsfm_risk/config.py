"""Typed loading of experiment configs.

Every experiment is fully described by YAML files under ``configs/``;
these models validate them so a typo fails loudly, not silently.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator

ASSET_CLASSES = ("equity_index", "equity_single", "fx", "crypto", "commodity", "bond_etf")


class SeriesSpec(BaseModel):
    ticker: str
    name: str


class UniverseConfig(BaseModel):
    start: dt.date
    end: dt.date
    price_field: str
    series: dict[str, list[SeriesSpec]]

    @field_validator("series")
    @classmethod
    def known_asset_classes(cls, v: dict[str, list[SeriesSpec]]) -> dict[str, list[SeriesSpec]]:
        unknown = set(v) - set(ASSET_CLASSES)
        if unknown:
            raise ValueError(f"unknown asset classes: {sorted(unknown)}")
        return v

    @model_validator(mode="after")
    def start_before_end(self) -> UniverseConfig:
        if self.start >= self.end:
            raise ValueError("universe start must precede end")
        return self

    def all_series(self) -> list[tuple[str, SeriesSpec]]:
        """Flatten to (asset_class, spec) pairs in config order."""
        return [(ac, s) for ac, specs in self.series.items() for s in specs]


class RegimeWindow(BaseModel):
    name: str
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def start_before_end(self) -> RegimeWindow:
        if self.start >= self.end:
            raise ValueError(f"regime window {self.name}: start must precede end")
        return self


class RegimesConfig(BaseModel):
    crisis_windows: list[RegimeWindow]

    @model_validator(mode="after")
    def no_overlap(self) -> RegimesConfig:
        windows = sorted(self.crisis_windows, key=lambda w: w.start)
        for prev, cur in zip(windows, windows[1:], strict=False):
            if cur.start <= prev.end:
                raise ValueError(f"regime windows overlap: {prev.name} and {cur.name}")
        return self

    def is_crisis(self, date: dt.date) -> bool:
        return any(w.start <= date <= w.end for w in self.crisis_windows)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_universe(path: str | Path = "configs/universe.yaml") -> UniverseConfig:
    return UniverseConfig.model_validate(_load_yaml(Path(path)))


def load_regimes(path: str | Path = "configs/regimes.yaml") -> RegimesConfig:
    return RegimesConfig.model_validate(_load_yaml(Path(path)))
