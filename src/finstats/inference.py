"""Estimator diagnostics, exact Gaussian intervals, and elementary likelihoods.

Gaussian parameters are (mu, variance), not (mu, standard deviation).
Monte Carlo diagnostics summarize independent realized estimates; they do not
turn the fixed population parameter into a random quantity.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import xlogy, xlog1py

from .descriptive import (
    _sample, _integer, sample_mean, sample_variance, sample_standard_deviation,
)


def _finite_scalar(value: float, name: str) -> float:
    if np.iscomplexobj(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def bias(estimates: ArrayLike, true_value: float) -> float:
    """Return Monte Carlo mean(estimates)-true_value, not population bias itself."""
    return sample_mean(estimates) - _finite_scalar(true_value, "true_value")


def estimator_variance(estimates: ArrayLike, ddof: int = 1) -> float:
    """Estimate sampling variance from replications with denominator B-ddof."""
    return sample_variance(estimates, ddof=ddof)


def mean_squared_error(estimates: ArrayLike, true_value: float) -> float:
    """Return the replication average squared error, with denominator B."""
    values = _sample(estimates)
    truth = _finite_scalar(true_value, "true_value")
    return float(np.mean((values - truth)**2))


def standard_error_mean(sigma: float, n: int) -> float:
    """Return sigma/sqrt(n) for iid observations; sigma may be zero."""
    sigma = _finite_scalar(sigma, "sigma")
    _integer(n, "n")
    if sigma < 0 or n == 0:
        raise ValueError("sigma must be nonnegative and n must be positive")
    return float(sigma / np.sqrt(n))


def _interval_sample(x: ArrayLike, alpha: float) -> tuple[np.ndarray, float]:
    values = _sample(x)
    alpha = _finite_scalar(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if values.size < 2 or sample_variance(values) == 0:
        raise ValueError("Gaussian intervals require n >= 2 and positive sample variance")
    return values, alpha


def mean_confidence_interval(x: ArrayLike, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided iid Gaussian mean interval with unknown variance (Student-t).

    Reject zero sample variance: the nondegenerate Gaussian pivot is undefined.
    """
    values, alpha = _interval_sample(x, alpha)
    center = sample_mean(values)
    radius = float(stats.t.ppf(1-alpha/2, values.size-1)) * standard_error_mean(
        sample_standard_deviation(values), values.size
    )
    return center-radius, center+radius


def variance_confidence_interval(x: ArrayLike, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided iid Gaussian population-variance interval (chi-squared)."""
    values, alpha = _interval_sample(x, alpha)
    df = values.size-1
    numerator = df * sample_variance(values)
    return (float(numerator / stats.chi2.ppf(1-alpha/2, df)),
            float(numerator / stats.chi2.ppf(alpha/2, df)))


def bernoulli_loglikelihood(theta: float, y: ArrayLike) -> float:
    """Log-likelihood of an ordered binary sample, theta in [0,1].

    Compatible boundary samples have log-likelihood zero; impossible ones
    return -inf. No binomial counting factor is included for ordered observations.
    """
    theta = _finite_scalar(theta, "theta")
    values = _sample(y)
    if not 0 <= theta <= 1 or not np.all((values == 0) | (values == 1)):
        raise ValueError("theta must be in [0,1] and y must contain only zero or one")
    successes = float(np.sum(values))
    return float(xlogy(successes, theta) + xlog1py(values.size-successes, -theta))


def gaussian_loglikelihood(params: ArrayLike, y: ArrayLike) -> float:
    """Gaussian iid sample log-likelihood at params=(mu, sigma_squared)."""
    parameters = _sample(params)
    if parameters.size != 2 or parameters[1] <= 0:
        raise ValueError("params must be (finite mu, strictly positive finite variance)")
    values = _sample(y)
    mu, variance = parameters
    return float(-values.size/2 * (np.log(2*np.pi)+np.log(variance))
                 - np.sum((values-mu)**2)/(2*variance))


def gaussian_mle(y: ArrayLike) -> tuple[float, float]:
    """Return (sample mean, denominator-n variance) for an iid Gaussian sample.

    Reject constant samples, including n=1: with sigma_squared > 0 their
    likelihood is unbounded as variance tends to zero, so no finite MLE exists.
    """
    values = _sample(y)
    variance = sample_variance(values, ddof=0)
    if variance == 0:
        raise ValueError("a nonconstant sample is required for a finite Gaussian MLE")
    return sample_mean(values), variance


def wald_statistic(estimate: float, null_value: float, variance: float) -> float:
    """Return scalar squared Wald statistic (estimate-null_value)**2/variance.

    variance is the estimated sampling variance of the estimator. This returns
    W=Z**2, with asymptotic chi-squared_1 calibration, not the signed Z statistic.
    """
    estimate = _finite_scalar(estimate, "estimate")
    null_value = _finite_scalar(null_value, "null_value")
    variance = _finite_scalar(variance, "variance")
    if variance <= 0:
        raise ValueError("variance must be strictly positive")
    return float((estimate-null_value)**2 / variance)
