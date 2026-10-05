import numpy as np
import pytest
from scipy import stats

from finstats.distributions import (
    location_scale_transform, student_t_kurtosis, student_t_variance,
)


def test_location_scale_transform():
    x = np.array([-1., 0., 2.])
    np.testing.assert_array_equal(location_scale_transform(x, 5, 2), [3, 5, 9])
    assert location_scale_transform(2, 5, 2).shape == ()
    assert location_scale_transform(2, 5, 2) == 9
    assert location_scale_transform(x.reshape(1, 3), 5, 2).shape == (1, 3)
    np.testing.assert_array_equal(x, [-1, 0, 2])


@pytest.mark.parametrize('scale', [0, -1, np.nan, np.inf])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        location_scale_transform([1, 2], 0, scale)
    with pytest.raises(ValueError):
        student_t_variance(5, scale)


@pytest.mark.parametrize('location', [np.nan, np.inf, -np.inf])
def test_invalid_location(location):
    with pytest.raises(ValueError):
        location_scale_transform([1], location, 1)


@pytest.mark.parametrize('x', [[np.nan], [np.inf], [1j]])
def test_invalid_transform_observations(x):
    with pytest.raises(ValueError):
        location_scale_transform(x, 0, 1)


@pytest.mark.parametrize('df', [2.5, 3, 5, 10, 30])
def test_student_t_variance(df):
    assert student_t_variance(df, 2) == pytest.approx(4 * df / (df - 2))
    assert student_t_variance(df, 2) == pytest.approx(stats.t.var(df, scale=2))


@pytest.mark.parametrize('df', [4.5, 5, 10, 30])
def test_student_t_kurtosis(df):
    assert student_t_kurtosis(df) == pytest.approx(3 + 6 / (df - 4))
    assert student_t_kurtosis(df, excess=True) == pytest.approx(stats.t.stats(df, moments='k'))


@pytest.mark.parametrize('df', [-1, 0, 1, 2, np.nan, np.inf])
def test_variance_moment_existence(df):
    with pytest.raises(ValueError):
        student_t_variance(df)


@pytest.mark.parametrize('df', [-1, 0, 1, 2, 3, 4, np.nan, np.inf])
def test_kurtosis_moment_existence(df):
    with pytest.raises(ValueError):
        student_t_kurtosis(df)


def test_large_df_limits_and_moment_boundaries():
    assert student_t_variance(1e12) == pytest.approx(1)
    assert student_t_kurtosis(1e12) == pytest.approx(3)
    assert student_t_variance(2.0001) > 10000
    assert student_t_kurtosis(4.0001) > 10000
