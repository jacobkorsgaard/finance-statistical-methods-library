import pytest

from finance_statistical_methods.risk_management import (
    empirical_expected_shortfall,
    empirical_var,
)


def test_empirical_var_and_es():
    losses = [1, 2, 3, 4, 5]
    assert empirical_var(losses, alpha=0.2) == 4.0
    assert empirical_expected_shortfall(losses, alpha=0.2) == pytest.approx(4.5)
