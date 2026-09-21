import numpy as np
import pytest

from finance_statistical_methods.returns import gross_returns, log_returns, simple_returns


def test_returns_basic():
    prices = [100.0, 110.0, 99.0]
    np.testing.assert_allclose(gross_returns(prices), [1.1, 0.9])
    np.testing.assert_allclose(simple_returns(prices), [0.1, -0.1])
    np.testing.assert_allclose(log_returns(prices), np.log([1.1, 0.9]))


def test_returns_reject_nonpositive_prices():
    with pytest.raises(ValueError):
        simple_returns([100.0, 0.0])
