"""ARCH/GARCH variance recursions; production estimation uses arch.

Simulation returns (observations, conditional VARIANCES), not standard deviations.
Finite-horizon filtering/forecasting permits persistence >=1 when an initial
variance is supplied; stationary moments and simulations require persistence <1.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..descriptive import _sample
from ..inference import _finite_scalar
from .ar import _positive_size


def _parameters(omega, alpha, beta=0., stationary=False):
    omega = _finite_scalar(omega,"omega")
    alpha = _finite_scalar(alpha,"alpha")
    beta = _finite_scalar(beta,"beta")
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("omega must be positive; alpha and beta must be nonnegative")
    if stationary and alpha+beta >= 1:
        raise ValueError("finite stationary variance requires alpha+beta < 1")
    return omega, alpha, beta


def arch1_unconditional_variance(omega: float, alpha: float) -> float:
    """Stationary variance omega/(1-alpha), requiring 0<=alpha<1."""
    return garch11_unconditional_variance(omega,alpha,0.)


def garch11_unconditional_variance(omega: float, alpha: float, beta: float) -> float:
    """Finite stationary variance omega/(1-alpha-beta), unit-variance innovations."""
    omega,alpha,beta = _parameters(omega,alpha,beta,stationary=True)
    return omega/(1-alpha-beta)


def garch11_variance_path(residuals: ArrayLike, omega: float, alpha: float, beta: float,
                          initial_variance: float | None = None) -> NDArray:
    """Variance aligned with residuals: h[0]=initial, h[t]=omega+alpha*a[t-1]^2+beta*h[t-1].

    Initial variance defaults to stationary population variance. At/above the
    unit persistence boundary, a strictly positive finite initial value is needed.
    This explicit initialization need not equal arch's estimated backcast.
    """
    omega,alpha,beta = _parameters(omega,alpha,beta)
    residuals = _sample(residuals)
    initial = (garch11_unconditional_variance(omega,alpha,beta) if initial_variance is None
               else _finite_scalar(initial_variance,"initial_variance"))
    if initial <= 0:
        raise ValueError("initial_variance must be positive")
    h = np.empty(residuals.size)
    h[0] = initial
    for t in range(1,residuals.size):
        h[t] = omega+alpha*residuals[t-1]**2+beta*h[t-1]
    return h


def simulate_garch11(omega: float, alpha: float, beta: float, n: int, rng=None, *, df: float | None = None) -> tuple[NDArray, NDArray]:
    """GARCH(1,1), with population-variance initialization and 1,000 burn-in steps.

    Innovations are Gaussian by default. Optional df>2 selects Student-t
    innovations scaled by sqrt((df-2)/df) to have unit variance.

    Burn-in approximates stationary initialization; it is not an exact draw from
    the stationary joint law. Higher moment existence is not assumed.
    """
    omega,alpha,beta = _parameters(omega,alpha,beta,stationary=True)
    _positive_size(n)
    rng = np.random.default_rng(rng)
    total = n+1000
    if df is None:
        innovations = rng.normal(size=total)
    else:
        df = _finite_scalar(df, "df")
        if df <= 2:
            raise ValueError("df must exceed 2 for unit-variance Student-t innovations")
        innovations = rng.standard_t(df, size=total)*np.sqrt((df-2)/df)
    h = np.empty(total)
    a = np.empty(total)
    h[0] = garch11_unconditional_variance(omega,alpha,beta)
    a[0] = np.sqrt(h[0])*innovations[0]
    for t in range(1,total):
        h[t] = omega+alpha*a[t-1]**2+beta*h[t-1]
        a[t] = np.sqrt(h[t])*innovations[t]
    return a[1000:],h[1000:]


def simulate_arch1(omega: float, alpha: float, n: int, rng=None) -> tuple[NDArray, NDArray]:
    """Gaussian ARCH(1) as the beta=0 variance recursion; see simulate_garch11."""
    return simulate_garch11(omega,alpha,0.,n,rng)


def garch11_forecast(last_residual: float, last_variance: float, omega: float,
                     alpha: float, beta: float, horizon: int = 1) -> NDArray:
    """Forecast conditional VARIANCES for horizons 1..horizon, with known parameters.

    First step uses the last observed shock; later steps replace future squared
    shocks by their conditional expectations. For persistence<1 forecasts revert
    to stationary variance. Finite horizons also support IGARCH/nonstationary
    parameters without claiming a finite long-run variance.
    """
    omega,alpha,beta = _parameters(omega,alpha,beta)
    _positive_size(horizon)
    last_residual = _finite_scalar(last_residual,"last_residual")
    last_variance = _finite_scalar(last_variance,"last_variance")
    if last_variance <= 0:
        raise ValueError("last_variance must be positive")
    forecasts = np.empty(horizon)
    forecasts[0] = omega+alpha*last_residual**2+beta*last_variance
    for h in range(1,horizon):
        forecasts[h] = omega+(alpha+beta)*forecasts[h-1]
    return forecasts
