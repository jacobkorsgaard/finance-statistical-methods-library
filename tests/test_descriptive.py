import numpy as np
import pytest
from scipy import stats

from finstats.descriptive import (
    central_moment, sample_kurtosis, sample_mean, sample_skewness,
    sample_standard_deviation, sample_variance,
)


def test_hand_computable_moments():
    x = [1, 2, 3]
    assert sample_mean(x) == 2
    assert sample_variance(x) == 1
    assert sample_variance(x, ddof=0) == pytest.approx(2 / 3)
    assert sample_variance(x, ddof=2) == 2
    assert sample_standard_deviation(x) == 1
    assert central_moment(x, 0) == 1
    assert central_moment(x, 1) == pytest.approx(0, abs=1e-15)
    assert central_moment(x, 2) == sample_variance(x, ddof=0)
    assert sample_skewness(x) == pytest.approx(0, abs=1e-15)
    assert sample_kurtosis(x) == pytest.approx(1.5)
    assert sample_kurtosis(x, excess=True) == pytest.approx(-1.5)


@pytest.mark.parametrize('ddof', [0, 1, 3])
def test_numpy_agreement(ddof):
    x = np.random.default_rng(101).normal(size=50)
    assert sample_mean(x) == pytest.approx(np.mean(x))
    assert sample_variance(x, ddof) == pytest.approx(np.var(x, ddof=ddof))
    assert sample_standard_deviation(x, ddof) == pytest.approx(np.std(x, ddof=ddof))


def test_scipy_moment_convention():
    x = np.random.default_rng(102).exponential(size=200)
    assert central_moment(x, 3) == pytest.approx(stats.moment(x, moment=3))
    assert sample_skewness(x) == pytest.approx(stats.skew(x, bias=True))
    assert sample_kurtosis(x) == pytest.approx(stats.kurtosis(x, fisher=False, bias=True))
    assert sample_kurtosis(x, excess=True) == pytest.approx(stats.kurtosis(x, bias=True))


@pytest.mark.parametrize('x', [[], [[1, 2]], [np.nan], [np.inf], [1 + 2j]])
@pytest.mark.parametrize('function', [sample_mean, sample_variance, sample_standard_deviation,
                                    sample_skewness, sample_kurtosis])
def test_invalid_observations(function, x):
    with pytest.raises(ValueError):
        function(x)


@pytest.mark.parametrize('x', [[], [[1, 2]], [np.nan], [np.inf], [1j]])
def test_invalid_central_moment_sample(x):
    with pytest.raises(ValueError):
        central_moment(x, 2)


@pytest.mark.parametrize('ddof', [-1, 1.5, True, 3, np.nan])
def test_invalid_ddof(ddof):
    for function in (sample_variance, sample_standard_deviation):
        with pytest.raises(ValueError):
            function([1, 2, 3], ddof)


@pytest.mark.parametrize('order', [-1, 1.5, True, np.nan])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        central_moment([1, 2, 3], order)


def test_singleton_and_constant_samples():
    assert sample_mean([4]) == 4
    assert sample_variance([4], ddof=0) == 0
    assert sample_standard_deviation([4], ddof=0) == 0
    assert central_moment([4], 2) == 0
    with pytest.raises(ValueError):
        sample_variance([4])
    for function in (sample_skewness, sample_kurtosis):
        with pytest.raises(ValueError):
            function([4, 4, 4])


def test_transformation_properties():
    x = np.array([-3., -1., 2., 7.])
    y = 5 + 2 * x
    assert sample_mean(y) == pytest.approx(5 + 2 * sample_mean(x))
    assert sample_variance(y) == pytest.approx(4 * sample_variance(x))
    assert sample_skewness(y) == pytest.approx(sample_skewness(x))
    assert sample_kurtosis(y) == pytest.approx(sample_kurtosis(x))
    assert sample_skewness(-x) == pytest.approx(-sample_skewness(x))
