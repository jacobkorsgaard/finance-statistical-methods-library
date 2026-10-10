"""Return definitions used directly in Notebook 00."""

import numpy as np


def test_returns_basic():
    prices = np.array([100.0, 110.0, 99.0])
    gross = prices[1:] / prices[:-1]
    np.testing.assert_allclose(gross, [1.1, 0.9])
    np.testing.assert_allclose(gross - 1, [0.1, -0.1])
    np.testing.assert_allclose(np.log(gross), np.log([1.1, 0.9]))


def test_dividend_is_part_of_total_return():
    np.testing.assert_allclose((105 + 2) / 100 - 1, 0.07)
