"""Preserve the distinction between strict-tail and inclusive sample averages."""

import numpy as np
from finstats.risk import historical_var, historical_es


def test_empirical_var_and_es_boundary_conventions():
    losses = np.array([1, 2, 3, 4, 5])
    threshold = historical_var(losses, alpha=0.2)
    assert threshold == 4
    assert historical_es(losses, alpha=0.2) == 5
    # The removed legacy wrapper included ties at VaR; keep the convention explicit.
    assert losses[losses >= threshold].mean() == 4.5
