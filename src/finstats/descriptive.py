"""Finite-sample moments, with explicit denominator conventions.

Skewness and kurtosis use m_k = mean((x - mean(x))**k), hence the
standardizing variance is m_2 (denominator n), without bias corrections.
All functions require a nonempty, finite, real, one-dimensional sample.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _sample(x: ArrayLike) -> NDArray[np.float64]:
    """Validate observations using the earlier package's sample contract."""
    if np.iscomplexobj(x):
        raise ValueError("x must contain real observations")
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a nonempty one-dimensional sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite observations")
    return values


def _integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def sample_mean(x: ArrayLike) -> float:
    """Return sum(x_i)/n for a finite observed sample."""
    values = _sample(x)
    return float(np.sum(values) / values.size)


def sample_variance(x: ArrayLike, ddof: int = 1) -> float:
    """Return sum((x_i - x_bar)**2)/(n-ddof); require 0 <= ddof < n."""
    values = _sample(x)
    _integer(ddof, "ddof")
    if ddof >= values.size:
        raise ValueError("ddof must be smaller than the sample size")
    deviations = values - sample_mean(values)
    return float(np.sum(deviations**2) / (values.size - ddof))


def sample_standard_deviation(x: ArrayLike, ddof: int = 1) -> float:
    """Return the square root of sample_variance under the same ddof."""
    return float(np.sqrt(sample_variance(x, ddof=ddof)))


def central_moment(x: ArrayLike, order: int) -> float:
    """Return m_order = n^{-1} sum((x_i-x_bar)**order), including m_0=1."""
    values = _sample(x)
    _integer(order, "order")
    return float(np.mean((values - sample_mean(values)) ** order))


def _standardized_moment(x: ArrayLike, order: int) -> float:
    values = _sample(x)
    variance = central_moment(values, 2)
    if variance == 0:
        raise ValueError("standardized moments require positive sample variance")
    # Standardize first to avoid unnecessary large powers of unscaled observations.
    z = (values - sample_mean(values)) / np.sqrt(variance)
    return float(np.mean(z**order))


def sample_skewness(x: ArrayLike) -> float:
    """Return m_3/m_2**(3/2), matching scipy.stats.skew(..., bias=True)."""
    return _standardized_moment(x, 3)


def sample_kurtosis(x: ArrayLike, excess: bool = False) -> float:
    """Return m_4/m_2**2; subtract 3 if excess=True (no bias correction)."""
    return _standardized_moment(x, 4) - (3.0 if excess else 0.0)
