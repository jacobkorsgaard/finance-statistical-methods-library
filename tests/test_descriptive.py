"""Regression checks of notebook calculations using standard libraries."""

import numpy as np
import pytest
from scipy import stats


def test_hand_computable_moments():
    x = [1, 2, 3]
    assert np.mean(x) == 2
    assert np.var(x, ddof=1) == 1
    assert np.var(x, ddof=0) == pytest.approx(2 / 3)
    assert np.var(x, ddof=2) == 2
    assert np.std(x, ddof=1) == 1
    assert stats.moment(x, moment=0) == 1
    assert stats.moment(x, moment=1) == pytest.approx(0, abs=1e-15)
    assert stats.moment(x, moment=2) == np.var(x, ddof=0)
    assert stats.skew(x, bias=True) == pytest.approx(0, abs=1e-15)
    assert stats.kurtosis(x, fisher=False, bias=True) == pytest.approx(1.5)
    assert stats.kurtosis(x, fisher=True, bias=True) == pytest.approx(-1.5)


@pytest.mark.parametrize("ddof", [0, 1, 3])
def test_numpy_agreement(ddof):
    x = np.random.default_rng(101).normal(size=50)
    centered = x - x.sum() / len(x)
    manual_variance = np.dot(centered, centered) / (len(x) - ddof)
    assert np.mean(x) == pytest.approx(x.sum() / len(x))
    assert np.var(x, ddof=ddof) == pytest.approx(manual_variance)
    assert np.std(x, ddof=ddof) == pytest.approx(np.sqrt(manual_variance))


def test_scipy_moment_convention():
    x = np.random.default_rng(102).exponential(size=200)
    centered = x - x.mean()
    m2, m3, m4 = [np.mean(centered**power) for power in (2, 3, 4)]
    assert stats.moment(x, moment=3) == pytest.approx(m3)
    assert stats.skew(x, bias=True) == pytest.approx(m3 / m2**1.5)
    assert stats.kurtosis(x, fisher=False, bias=True) == pytest.approx(m4 / m2**2)
    assert stats.kurtosis(x, fisher=True, bias=True) == pytest.approx(m4 / m2**2 - 3)


def test_transformation_properties():
    x = np.array([-3.0, -1.0, 2.0, 7.0])
    y = 5 + 2 * x
    assert np.mean(y) == pytest.approx(5 + 2 * np.mean(x))
    assert np.var(y, ddof=1) == pytest.approx(4 * np.var(x, ddof=1))
    assert stats.skew(y, bias=True) == pytest.approx(stats.skew(x, bias=True))
    assert stats.kurtosis(y, fisher=False, bias=True) == pytest.approx(
        stats.kurtosis(x, fisher=False, bias=True)
    )
    assert stats.skew(-x, bias=True) == pytest.approx(-stats.skew(x, bias=True))
