"""Reproducible process simulations with explicit initialization and indexing."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ._validation import _sample, _integer, _finite_scalar, _positive_size
from .linear import _stationary_phi, _coefficients, ar1_unconditional_variance
from .volatility import _parameters, garch11_unconditional_variance


def simulate_ar1(
    phi: float, sigma: float, n: int, mu: float = 0.0, rng=None
) -> NDArray:
    """n Gaussian AR(1) observations, initialized from the exact stationary law.

    sigma is innovation SD, not process SD. Supply a Generator for reproducibility.
    No truncated burn-in approximation is needed for this Gaussian model.
    """
    phi = _stationary_phi(phi)
    _positive_size(n)
    mu = _finite_scalar(mu, "mu")
    variance = ar1_unconditional_variance(phi, _finite_scalar(sigma, "sigma") ** 2)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng(rng)
    previous = rng.normal(mu, np.sqrt(variance))
    innovations = rng.normal(0, sigma, size=n)
    values = np.empty(n)
    for t in range(n):
        previous = mu + phi * (previous - mu) + innovations[t]
        values[t] = previous
    return values


def simulate_ma(
    theta: ArrayLike, sigma: float, n: int, mu: float = 0.0, rng=None
) -> NDArray:
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
    innovations = rng.normal(0, sigma, size=n + len(theta))
    return mu + np.convolve(innovations, np.r_[1.0, theta], mode="valid")


def simulate_random_walk(
    n: int, sigma: float = 1.0, initial: float = 0.0, rng=None
) -> NDArray:
    """Return (Y_0,...,Y_n), so n increments yield n+1 observations.

    Y_0 is the fixed initial value. differencing returns exactly the n generated
    innovations. Variance at index t is t*sigma**2.
    """
    _positive_size(n)
    sigma, initial = _finite_scalar(sigma, "sigma"), _finite_scalar(initial, "initial")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    innovations = np.random.default_rng(rng).normal(0, sigma, size=n)
    return np.r_[initial, initial + np.cumsum(innovations)]


def simulate_garch11(
    omega: float,
    alpha: float,
    beta: float,
    n: int,
    rng=None,
    *,
    df: float | None = None,
) -> tuple[NDArray, NDArray]:
    """GARCH(1,1), with population-variance initialization and 1,000 burn-in steps.

    Innovations are Gaussian by default. Optional df>2 selects Student-t
    innovations scaled by sqrt((df-2)/df) to have unit variance.

    Burn-in approximates stationary initialization; it is not an exact draw from
    the stationary joint law. Higher moment existence is not assumed.
    """
    omega, alpha, beta = _parameters(omega, alpha, beta, stationary=True)
    _positive_size(n)
    rng = np.random.default_rng(rng)
    total = n + 1000
    if df is None:
        innovations = rng.normal(size=total)
    else:
        df = _finite_scalar(df, "df")
        if df <= 2:
            raise ValueError("df must exceed 2 for unit-variance Student-t innovations")
        innovations = rng.standard_t(df, size=total) * np.sqrt((df - 2) / df)
    h = np.empty(total)
    a = np.empty(total)
    h[0] = garch11_unconditional_variance(omega, alpha, beta)
    a[0] = np.sqrt(h[0]) * innovations[0]
    for t in range(1, total):
        h[t] = omega + alpha * a[t - 1] ** 2 + beta * h[t - 1]
        a[t] = np.sqrt(h[t]) * innovations[t]
    return a[1000:], h[1000:]


def simulate_arch1(
    omega: float, alpha: float, n: int, rng=None
) -> tuple[NDArray, NDArray]:
    """Gaussian ARCH(1) as the beta=0 variance recursion; see simulate_garch11."""
    return simulate_garch11(omega, alpha, 0.0, n, rng)


def simulate_ar1_garch11(
    phi: float,
    omega: float,
    alpha: float,
    beta: float,
    n: int,
    mu: float = 0.0,
    rng=None,
    *,
    df: float | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Return aligned (observations, innovations, conditional variances).

    X_t = mu + phi*(X_(t-1)-mu) + a_t, with Gaussian (default) or unit-variance Student-t (df>2) standardized
    GARCH(1,1) shocks. mu is the unconditional mean, not the recursion
    intercept. Require |phi|<1 and alpha+beta<1 for finite stationary
    second moments. Output variance belongs to a_t conditional on the past,
    rather than to the unconditional X_t distribution.

    Reuse the GARCH simulator's 1,000-step variance burn-in, then discard
    1,000 further observations while initializing the mean at mu. This
    approximates stationary initialization, rather than drawing its exact law.
    """
    phi = _stationary_phi(phi)
    _positive_size(n)
    mu = _finite_scalar(mu, "mu")
    burn = 1000
    shocks, variances = simulate_garch11(omega, alpha, beta, n + burn, rng=rng, df=df)
    observations = np.empty(n + burn)
    previous = mu
    for t, shock in enumerate(shocks):
        previous = mu + phi * (previous - mu) + shock
        observations[t] = previous
    return observations[burn:], shocks[burn:], variances[burn:]
