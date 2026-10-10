"""Explicit AR/MA polynomial conventions and known-parameter AR(1) forecasts."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ._validation import _integer, _finite_scalar, _positive_size


def _stationary_phi(phi: float) -> float:
    phi = _finite_scalar(phi, "phi")
    if abs(phi) >= 1:
        raise ValueError("a causal stationary AR(1) requires |phi| < 1")
    return phi


def ar1_unconditional_variance(phi: float, sigma2: float) -> float:
    """Population variance sigma2/(1-phi**2) for the causal stationary solution."""
    phi = _stationary_phi(phi)
    sigma2 = _finite_scalar(sigma2, "sigma2")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    return sigma2 / (1 - phi**2)


def ar1_acf(phi: float, max_lag: int) -> NDArray:
    """Stationary population autocorrelations phi**j, including lag zero."""
    phi = _stationary_phi(phi)
    _integer(max_lag, "max_lag")
    return np.asarray(phi ** np.arange(max_lag + 1), dtype=float)


def _coefficients(coefficients: ArrayLike) -> NDArray:
    if np.iscomplexobj(coefficients):
        raise ValueError("coefficients must be real")
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be a finite one-dimensional vector")
    return values


def ar_roots(phi_coefficients: ArrayLike) -> NDArray:
    """Roots in z of 1-phi_1*z-...; an empty/zero AR polynomial has no roots."""
    coefficients = _coefficients(phi_coefficients)
    return np.polynomial.polynomial.polyroots(np.r_[1.0, -coefficients])


def is_stationary_ar(phi_coefficients: ArrayLike) -> bool:
    """Whether all AR roots lie strictly outside the unit circle (causal solution)."""
    return bool(np.all(np.abs(ar_roots(phi_coefficients)) > 1))


def ar1_forecast(last_value: float, phi: float, mu: float, horizon: int) -> NDArray:
    """Mean forecasts for horizons 1..horizon under known stationary AR(1)."""
    phi = _stationary_phi(phi)
    _positive_size(horizon)
    last_value = _finite_scalar(last_value, "last_value")
    mu = _finite_scalar(mu, "mu")
    return mu + phi ** np.arange(1, horizon + 1) * (last_value - mu)


def ma_roots(theta: ArrayLike) -> NDArray:
    """Roots of 1+theta_1*z+...; empty/zero MA polynomials have no roots."""
    return np.polynomial.polynomial.polyroots(np.r_[1.0, _coefficients(theta)])


def is_invertible_ma(theta: ArrayLike) -> bool:
    """Whether every MA root lies strictly outside the unit circle."""
    return bool(np.all(np.abs(ma_roots(theta)) > 1))


def arma_roots(ar_coefficients, ma_coefficients):
    """Return (AR roots, MA roots) for the course minus-AR/plus-MA convention."""
    return ar_roots(ar_coefficients), ma_roots(ma_coefficients)
