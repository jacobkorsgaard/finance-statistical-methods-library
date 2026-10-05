"""Finite Gaussian moving averages and invertibility roots, plus-sign convention."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..inference import _finite_scalar
from .ar import _coefficients, _positive_size


def ma_roots(theta: ArrayLike) -> NDArray:
    """Roots of 1+theta_1*z+...; empty/zero MA polynomials have no roots."""
    return np.polynomial.polynomial.polyroots(np.r_[1., _coefficients(theta)])


def is_invertible_ma(theta: ArrayLike) -> bool:
    """Whether every MA root lies strictly outside the unit circle."""
    return bool(np.all(np.abs(ma_roots(theta)) > 1))


def simulate_ma(theta: ArrayLike, sigma: float, n: int, mu: float = 0., rng=None) -> NDArray:
    """n observations of mu+epsilon_t+sum theta_j*epsilon_(t-j).

    Draw pre-sample Gaussian innovations too, making every returned observation
    follow the stationary finite-MA law. Non-invertible MA processes are valid.
    """
    theta = _coefficients(theta)
    _positive_size(n)
    sigma, mu = _finite_scalar(sigma, "sigma"), _finite_scalar(mu, "mu")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng(rng)
    innovations = rng.normal(0,sigma,size=n+len(theta))
    return mu+np.convolve(innovations, np.r_[1.,theta], mode="valid")
