import pytest
from finstats.empirical import empirical_cdf, empirical_quantile


def test_ecdf():
    assert empirical_cdf([1, 2, 3, 4])(2) == pytest.approx(0.5)


def test_empirical_quantile_uses_inverse_ecdf():
    assert empirical_quantile([1, 2, 3, 4], 0.75) == 3
