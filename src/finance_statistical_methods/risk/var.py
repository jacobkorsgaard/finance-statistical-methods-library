"""Value at Risk measures."""

from __future__ import annotations

from numpy.typing import ArrayLike

from finance_statistical_methods.distributions.nonparametric.empirical import empirical_quantile


def empirical_var(losses: ArrayLike, alpha: float = 0.05) -> float:
    """Estimate VaR from the empirical loss distribution.

    ``alpha`` is the upper-tail probability, so 5% VaR is the 95th percentile
    of the loss distribution.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return empirical_quantile(losses, 1.0 - alpha)
