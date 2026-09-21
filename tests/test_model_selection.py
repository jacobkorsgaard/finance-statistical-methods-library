import math
import pytest

from finance_statistical_methods.estimation import aic, bic


def test_aic():
    assert aic(log_likelihood=-10.0, n_params=3) == pytest.approx(26.0)


def test_bic():
    expected = 20.0 + 3.0 * math.log(100)
    assert bic(log_likelihood=-10.0, n_params=3, n_obs=100) == pytest.approx(expected)
