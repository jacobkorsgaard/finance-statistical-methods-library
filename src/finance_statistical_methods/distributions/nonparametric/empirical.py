"""Empirical distribution functions and quantiles."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_sample(sample: ArrayLike) -> np.ndarray:
    values = np.asarray(sample, dtype=float)
    if values.ndim != 1:
        raise ValueError("sample must be one-dimensional")
    if values.size == 0:
        raise ValueError("sample cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("sample must contain only finite values")
    return values


def ecdf(sample: ArrayLike, x: float) -> float:
    """Evaluate the empirical CDF F_n(x) = n^{-1} sum 1{X_i <= x}."""
    values = _validate_sample(sample)
    return float(np.mean(values <= x))


def empirical_quantile(sample: ArrayLike, q: float) -> float:
    """Return an empirical quantile using the generalized inverse ECDF convention.

    NumPy's ``method='inverted_cdf'`` corresponds to selecting the first order
    statistic for which the ECDF reaches or exceeds q.
    """
    values = _validate_sample(sample)
    if not 0 <= q <= 1:
        raise ValueError("q must lie between 0 and 1")
    return float(np.quantile(values, q, method="inverted_cdf"))
