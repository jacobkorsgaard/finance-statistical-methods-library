"""Basic return transformations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_prices(prices: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1:
        raise ValueError("prices must be one-dimensional")
    if values.size < 2:
        raise ValueError("at least two prices are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("prices must contain only finite values")
    if np.any(values <= 0):
        raise ValueError("prices must be strictly positive")
    return values


def gross_returns(prices: ArrayLike) -> NDArray[np.float64]:
    """Return one-period gross returns, P_t / P_{t-1}."""
    values = _validate_prices(prices)
    return values[1:] / values[:-1]


def simple_returns(prices: ArrayLike) -> NDArray[np.float64]:
    """Return one-period simple returns, P_t / P_{t-1} - 1."""
    return gross_returns(prices) - 1.0


def log_returns(prices: ArrayLike) -> NDArray[np.float64]:
    """Return one-period continuously compounded log returns."""
    values = _validate_prices(prices)
    return np.log(values[1:] / values[:-1])
