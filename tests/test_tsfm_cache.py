import numpy as np
import pandas as pd
import pytest

from tsfm_risk.models.tsfm.cache import (
    append_forecasts,
    cache_path,
    cached_origins,
    load_cached,
    to_cube,
    to_matrix,
)

LEVELS = (0.1, 0.5, 0.9)


def _batch(n, start="2020-01-01", seed=0):
    rng = np.random.default_rng(seed)
    origins = list(pd.bdate_range(start, periods=n))
    base = np.sort(rng.normal(0, 0.01, (n, 1, len(LEVELS))), axis=2)
    return origins, base


class TestPaths:
    def test_sanitizes_awkward_names(self, tmp_path):
        p = cache_path("amazon/chronos-bolt-base", "^GSPC", 1, 1024, root=tmp_path)
        assert "/" not in p.name and "^" not in p.name
        assert p.parent.name == "amazon_chronos-bolt-base"
        assert p.name == "_GSPC__h1__c1024.parquet"

    def test_config_is_part_of_the_path(self, tmp_path):
        """Different context lengths must not overwrite each other — the
        context-length ablation depends on it."""
        a = cache_path("m", "s", 1, 512, root=tmp_path)
        b = cache_path("m", "s", 1, 1024, root=tmp_path)
        c = cache_path("m", "s", 5, 512, root=tmp_path)
        assert len({a, b, c}) == 3


class TestRoundTrip:
    def test_write_then_read_matches(self, tmp_path):
        path = tmp_path / "x.parquet"
        origins, q = _batch(20)
        append_forecasts(path, origins, LEVELS, q)

        got_origins, got_levels, got_q = to_matrix(load_cached(path))
        assert list(got_origins) == origins
        np.testing.assert_allclose(got_levels, LEVELS)
        np.testing.assert_allclose(got_q, q[:, 0, :])

    def test_missing_file_reads_as_none(self, tmp_path):
        assert load_cached(tmp_path / "nope.parquet") is None
        assert cached_origins(tmp_path / "nope.parquet") == set()


class TestResumability:
    def test_second_batch_appends(self, tmp_path):
        path = tmp_path / "x.parquet"
        o1, q1 = _batch(10, "2020-01-01", seed=1)
        o2, q2 = _batch(10, "2020-02-01", seed=2)
        append_forecasts(path, o1, LEVELS, q1)
        append_forecasts(path, o2, LEVELS, q2)
        assert len(cached_origins(path)) == 20

    def test_rerunning_same_origins_is_a_no_op(self, tmp_path):
        """A resumed sweep re-submits origins it already has; the cache must
        not duplicate rows or change stored values."""
        path = tmp_path / "x.parquet"
        origins, q = _batch(12, seed=3)
        append_forecasts(path, origins, LEVELS, q)
        before = load_cached(path)

        append_forecasts(path, origins, LEVELS, q * 99.0)  # different values!
        after = load_cached(path)

        assert len(after) == len(before)
        pd.testing.assert_frame_equal(
            before.sort_values(["origin", "level"]).reset_index(drop=True),
            after.sort_values(["origin", "level"]).reset_index(drop=True),
        )

    def test_partial_overlap_keeps_old_and_adds_new(self, tmp_path):
        path = tmp_path / "x.parquet"
        o1, q1 = _batch(10, "2020-01-01", seed=4)
        append_forecasts(path, o1, LEVELS, q1)
        o2 = list(pd.bdate_range("2020-01-08", periods=10))  # overlaps o1 tail
        _, q2 = _batch(10, seed=5)
        append_forecasts(path, o2, LEVELS, q2)
        assert len(cached_origins(path)) == len(set(o1) | set(o2))


class TestMultiDayHorizon:
    """One run at H=20 must serve the 1-, 5- and 20-day horizons."""

    @staticmethod
    def _multiday(n=5, days=4, seed=7):
        rng = np.random.default_rng(seed)
        origins = list(pd.bdate_range("2021-01-01", periods=n))
        q = np.sort(rng.normal(0, 0.01, (n, days, len(LEVELS))), axis=2)
        return origins, q

    def test_every_day_is_stored_and_retrievable(self, tmp_path):
        path = tmp_path / "x.parquet"
        origins, q = self._multiday()
        append_forecasts(path, origins, LEVELS, q)
        df = load_cached(path)
        for d in range(1, q.shape[1] + 1):
            _, _, got = to_matrix(df, day=d)
            np.testing.assert_allclose(got, q[:, d - 1, :])

    def test_cube_reassembles_the_full_horizon(self, tmp_path):
        path = tmp_path / "x.parquet"
        origins, q = self._multiday()
        append_forecasts(path, origins, LEVELS, q)
        got_origins, got_levels, cube = to_cube(load_cached(path))
        assert list(got_origins) == origins
        np.testing.assert_allclose(got_levels, LEVELS)
        np.testing.assert_allclose(cube, q)

    def test_absent_day_is_an_explicit_error(self, tmp_path):
        path = tmp_path / "x.parquet"
        origins, q = self._multiday(days=2)
        append_forecasts(path, origins, LEVELS, q)
        with pytest.raises(ValueError, match="day 5"):
            to_matrix(load_cached(path), day=5)

    def test_row_count_covers_all_days(self, tmp_path):
        path = tmp_path / "x.parquet"
        origins, q = self._multiday(n=6, days=3)
        append_forecasts(path, origins, LEVELS, q)
        assert len(load_cached(path)) == 6 * 3 * len(LEVELS)


class TestValidation:
    def test_rejects_wrong_batch_size(self, tmp_path):
        origins, q = _batch(5)
        with pytest.raises(ValueError, match="batch size"):
            append_forecasts(tmp_path / "x.parquet", origins[:3], LEVELS, q)

    def test_rejects_wrong_grid_size(self, tmp_path):
        origins, q = _batch(5)
        with pytest.raises(ValueError, match="grid size"):
            append_forecasts(tmp_path / "x.parquet", origins, (0.1, 0.9), q)

    def test_rejects_two_dimensional_input(self, tmp_path):
        origins, q = _batch(5)
        with pytest.raises(ValueError, match="shape"):
            append_forecasts(tmp_path / "x.parquet", origins, LEVELS, q[:, 0, :])
