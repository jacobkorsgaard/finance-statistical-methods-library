"""Known-parameter volatility recursions; use arch for standard estimation.

Outputs are conditional variances, not standard deviations. Initialization is
explicit and need not equal a fitted model's backcast.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ._validation import _sample, _finite_scalar, _positive_size


def _parameters(omega, alpha, beta=0.0, stationary=False):
    omega = _finite_scalar(omega, "omega")
    alpha = _finite_scalar(alpha, "alpha")
    beta = _finite_scalar(beta, "beta")
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("omega must be positive; alpha and beta must be nonnegative")
    if stationary and alpha + beta >= 1:
        raise ValueError("finite stationary variance requires alpha+beta < 1")
    return omega, alpha, beta


def arch1_unconditional_variance(omega: float, alpha: float) -> float:
    """Stationary variance omega/(1-alpha), requiring 0<=alpha<1."""
    return garch11_unconditional_variance(omega, alpha, 0.0)


def garch11_unconditional_variance(omega: float, alpha: float, beta: float) -> float:
    """Finite stationary variance omega/(1-alpha-beta), unit-variance innovations."""
    omega, alpha, beta = _parameters(omega, alpha, beta, stationary=True)
    return omega / (1 - alpha - beta)


def garch11_variance_path(
    residuals: ArrayLike,
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float | None = None,
) -> NDArray:
    """Variance aligned with residuals: h[0]=initial, h[t]=omega+alpha*a[t-1]^2+beta*h[t-1].

    Initial variance defaults to stationary population variance. At/above the
    unit persistence boundary, a strictly positive finite initial value is needed.
    This explicit initialization need not equal arch's estimated backcast.
    """
    omega, alpha, beta = _parameters(omega, alpha, beta)
    residuals = _sample(residuals)
    initial = (
        garch11_unconditional_variance(omega, alpha, beta)
        if initial_variance is None
        else _finite_scalar(initial_variance, "initial_variance")
    )
    if initial <= 0:
        raise ValueError("initial_variance must be positive")
    h = np.empty(residuals.size)
    h[0] = initial
    for t in range(1, residuals.size):
        h[t] = omega + alpha * residuals[t - 1] ** 2 + beta * h[t - 1]
    return h


def garch11_forecast(
    last_residual: float,
    last_variance: float,
    omega: float,
    alpha: float,
    beta: float,
    horizon: int = 1,
) -> NDArray:
    """Forecast conditional VARIANCES for horizons 1..horizon, with known parameters.

    First step uses the last observed shock; later steps replace future squared
    shocks by their conditional expectations. For persistence<1 forecasts revert
    to stationary variance. Finite horizons also support IGARCH/nonstationary
    parameters without claiming a finite long-run variance.
    """
    omega, alpha, beta = _parameters(omega, alpha, beta)
    _positive_size(horizon)
    last_residual = _finite_scalar(last_residual, "last_residual")
    last_variance = _finite_scalar(last_variance, "last_variance")
    if last_variance <= 0:
        raise ValueError("last_variance must be positive")
    forecasts = np.empty(horizon)
    forecasts[0] = omega + alpha * last_residual**2 + beta * last_variance
    for h in range(1, horizon):
        forecasts[h] = omega + (alpha + beta) * forecasts[h - 1]
    return forecasts
