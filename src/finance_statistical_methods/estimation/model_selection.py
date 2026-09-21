"""Information criteria for fitted statistical models."""

from __future__ import annotations

import math


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike information criterion: -2 log L + 2k."""
    if n_params < 0:
        raise ValueError("n_params must be non-negative")
    return -2.0 * log_likelihood + 2.0 * n_params


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian information criterion: -2 log L + k log(n)."""
    if n_params < 0:
        raise ValueError("n_params must be non-negative")
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    return -2.0 * log_likelihood + n_params * math.log(n_obs)
