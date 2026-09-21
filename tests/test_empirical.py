import pytest

from finance_statistical_methods.distributions.nonparametric import ecdf, empirical_quantile


def test_ecdf():
    sample = [1, 2, 3, 4]
    assert ecdf(sample, 2) == pytest.approx(0.5)


def test_empirical_quantile_uses_inverse_ecdf():
    sample = [1, 2, 3, 4]
    assert empirical_quantile(sample, 0.75) == 3.0
