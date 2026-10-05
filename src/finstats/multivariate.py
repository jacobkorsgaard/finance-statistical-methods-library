"""Sample dependence and population moments of linear combinations.

Sample matrices contain observations in rows and variables in columns.
Covariances use n-ddof in the denominator (ddof=1 by default).
Population covariance matrices may be singular but must be symmetric and
positive semidefinite, up to relative floating-point tolerance 1e-12.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .descriptive import _integer, _sample, sample_mean, sample_standard_deviation


def _paired_sample(x: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray]:
    left, right = _sample(x), _sample(y)
    if left.size != right.size:
        raise ValueError("x and y must have the same number of observations")
    return left, right


def covariance(x: ArrayLike, y: ArrayLike, ddof: int = 1) -> float:
    """Return sum((x_i-x_bar)*(y_i-y_bar))/(n-ddof) for paired observations."""
    left, right = _paired_sample(x, y)
    _integer(ddof, "ddof")
    if ddof >= left.size:
        raise ValueError("ddof must be smaller than the sample size")
    return float(np.dot(left - sample_mean(left), right - sample_mean(right))
                 / (left.size - ddof))


def correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Return Pearson sample correlation; reject either constant sample.

    The common covariance/variance denominator cancels. No bias correction
    is applied. At least two paired observations are needed for nonzero spread.
    """
    left, right = _paired_sample(x, y)
    left_sd = sample_standard_deviation(left, ddof=0)
    right_sd = sample_standard_deviation(right, ddof=0)
    if left_sd == 0 or right_sd == 0:
        raise ValueError("correlation requires positive variance in both samples")
    value = covariance(left / left_sd, right / right_sd, ddof=0)
    # Roundoff can put a perfect linear relationship marginally outside [-1, 1].
    return float(np.clip(value, -1.0, 1.0))


def covariance_matrix(x: ArrayLike, ddof: int = 1) -> NDArray[np.float64]:
    """Return centered.T @ centered/(n-ddof); x has shape (n, K).

    A single variable is supported as an (n, 1) input and returns a (1, 1)
    matrix. Missing, nonfinite, complex, empty, or one-dimensional inputs
    are rejected rather than silently reshaped or deleted.
    """
    if np.iscomplexobj(x):
        raise ValueError("x must contain real observations")
    values = np.asarray(x, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("x must be a nonempty two-dimensional sample matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite observations")
    _integer(ddof, "ddof")
    if ddof >= values.shape[0]:
        raise ValueError("ddof must be smaller than the number of observations")
    means = np.array([sample_mean(column) for column in values.T])
    centered = values - means
    return centered.T @ centered / (values.shape[0] - ddof)


def linear_combination_mean(
    weights: ArrayLike, means: ArrayLike, intercept: float = 0.0
) -> float:
    """Return intercept + weights.T @ means; weights need not sum to one."""
    coefficients, population_means = _paired_sample(weights, means)
    if np.iscomplexobj(intercept) or np.ndim(intercept) != 0 or not np.isfinite(intercept):
        raise ValueError("intercept must be a finite real scalar")
    return float(intercept + np.dot(coefficients, population_means))


def linear_combination_variance(
    weights: ArrayLike, covariance_matrix: ArrayLike
) -> float:
    """Return weights.T @ Sigma @ weights for a valid population covariance.

    Sigma must have shape (K, K) matching the coefficient vector. Negative
    variances, asymmetry, and non-positive-semidefinite inputs are rejected;
    zero or singular covariance is valid. Tiny roundoff asymmetry and negative
    quadratic forms within the relative 1e-12 tolerance are normalized to zero.
    """
    coefficients = _sample(weights)
    if np.iscomplexobj(covariance_matrix):
        raise ValueError("covariance matrix must be real")
    sigma = np.asarray(covariance_matrix, dtype=float)
    if sigma.shape != (coefficients.size, coefficients.size):
        raise ValueError("covariance matrix shape must match the weight vector")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("covariance matrix must contain only finite values")
    scale = float(np.max(np.abs(sigma)))
    tolerance = 1e-12 * scale
    if not np.allclose(sigma, sigma.T, rtol=0, atol=tolerance):
        raise ValueError("covariance matrix must be symmetric")
    sigma = (sigma + sigma.T) / 2
    if np.min(np.linalg.eigvalsh(sigma)) < -tolerance:
        raise ValueError("covariance matrix must be positive semidefinite")
    value = float(coefficients @ sigma @ coefficients)
    return max(value, 0.0)
