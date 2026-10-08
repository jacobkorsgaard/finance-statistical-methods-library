"""Simulation of a stationary AR mean with GARCH innovation variance."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..inference import _finite_scalar
from .ar import _positive_size, _stationary_phi
from .volatility import simulate_garch11


def simulate_ar1_garch11(
    phi: float, omega: float, alpha: float, beta: float, n: int,
    mu: float = 0., rng=None, *, df: float | None = None,
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
