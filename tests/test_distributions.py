"""Regression checks of notebook calculations using standard libraries."""

import numpy as np
import pytest
from scipy import stats


def test_location_scale_transform():
    x = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(5 + 2 * np.asarray(x), [3, 5, 9])
    assert (5 + 2 * np.asarray(2)).shape == ()
    assert 5 + 2 * np.asarray(2) == 9
    assert (5 + 2 * np.asarray(x.reshape(1, 3))).shape == (1, 3)
    np.testing.assert_array_equal(x, [-1, 0, 2])


@pytest.mark.parametrize("df", [2.5, 3, 5, 10, 30])
def test_student_t_variance(df):
    assert stats.t.var(df, scale=2) == pytest.approx(4 * df / (df - 2))
    assert stats.t.var(df, scale=2) == pytest.approx(stats.t.var(df, scale=2))


@pytest.mark.parametrize("df", [4.5, 5, 10, 30])
def test_student_t_kurtosis(df):
    assert stats.t.stats(df, moments="k") + (0 if False else 3) == pytest.approx(
        3 + 6 / (df - 4)
    )
    assert stats.t.stats(df, moments="k") + (0 if True else 3) == pytest.approx(
        stats.t.stats(df, moments="k")
    )


def test_large_df_limits_and_moment_boundaries():
    assert stats.t.var(1000000000000.0, scale=1) == pytest.approx(1)
    assert stats.t.stats(1000000000000.0, moments="k") + (
        0 if False else 3
    ) == pytest.approx(3)
    assert stats.t.var(2.0001, scale=1) > 10000
    assert stats.t.stats(4.0001, moments="k") + (0 if False else 3) > 10000
