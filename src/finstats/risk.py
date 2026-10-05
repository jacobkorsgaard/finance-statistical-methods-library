"""Upper-tail loss measures; alpha is tail probability, not confidence level.

Losses are L=-R. Historical ES uses STRICT exceedances of the inverse-ECDF
quantile as specified in Notebook 05, unlike the legacy inclusive-tail API.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from .descriptive import _sample
from .inference import _finite_scalar
from .nonparametric import empirical_quantile


def _alpha(alpha: float) -> float:
    alpha = _finite_scalar(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return alpha


def _location_scale(location: float, scale: float) -> tuple[float, float]:
    location = _finite_scalar(location, "location")
    scale = _finite_scalar(scale, "scale")
    if scale <= 0:
        raise ValueError("scale must be strictly positive")
    return location, scale


def historical_var(losses: ArrayLike, alpha: float = .05) -> float:
    """Return the (1-alpha) generalized-inverse ECDF loss quantile."""
    return empirical_quantile(losses, 1-_alpha(alpha))


def historical_es(losses: ArrayLike, alpha: float = .05) -> float:
    """Mean observed loss STRICTLY above VaR; reject an empty exceedance tail.

    This course estimator is not fractional-mass empirical integrated ES and
    need not average exactly alpha*n observations, particularly with ties.
    """
    values = _sample(losses)
    threshold = historical_var(values, alpha)
    tail = values[values > threshold]
    if tail.size == 0:
        raise ValueError("no observed losses strictly exceed the VaR threshold")
    return float(np.mean(tail))


def gaussian_var(mu: float, sigma: float, alpha: float = .05) -> float:
    """Gaussian LOSS quantile mu+sigma*Phi^{-1}(1-alpha)."""
    mu, sigma = _location_scale(mu, sigma)
    return float(mu+sigma*stats.norm.isf(_alpha(alpha)))


def gaussian_es(mu: float, sigma: float, alpha: float = .05) -> float:
    """Gaussian mean loss conditional on exceeding its VaR."""
    mu, sigma = _location_scale(mu, sigma)
    alpha = _alpha(alpha)
    return float(mu+sigma*stats.norm.pdf(stats.norm.isf(alpha))/alpha)


def student_t_var(location: float, scale: float, df: float, alpha: float = .05) -> float:
    """LOSS quantile for location+scale*T_df; scale is not standard deviation."""
    location, scale = _location_scale(location, scale)
    df = _finite_scalar(df, "df")
    if df <= 0:
        raise ValueError("df must be strictly positive")
    return float(location+scale*stats.t.isf(_alpha(alpha), df))


def student_t_es(location: float, scale: float, df: float, alpha: float = .05) -> float:
    """LOSS ES for location+scale*T_df, requiring df>1 (finite tail mean)."""
    location, scale = _location_scale(location, scale)
    df = _finite_scalar(df, "df")
    if df <= 1:
        raise ValueError("finite Student-t ES requires df > 1")
    alpha = _alpha(alpha)
    q = stats.t.isf(alpha, df)
    return float(location+scale*(df+q*q)/(df-1)*stats.t.pdf(q,df)/alpha)
