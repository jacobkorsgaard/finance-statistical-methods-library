"""Univariate Laplace distribution.

Parameterization
----------------
mu    : location
scale : positive scale parameter

The distribution is symmetric around mu and has heavier tails
than the Normal distribution.

Functions
---------
pdf
cdf
quantile
sample
fit
"""

import numpy as np
from scipy.stats import laplace

from .fit_statistics import fit_result


def pdf(
    x,
    mu=0.0,
    scale=1.0,
):
    """Laplace probability density function."""
    return laplace.pdf(
        x,
        loc=mu,
        scale=scale,
    )


def cdf(
    x,
    mu=0.0,
    scale=1.0,
):
    """Laplace cumulative distribution function."""
    return laplace.cdf(
        x,
        loc=mu,
        scale=scale,
    )


def quantile(
    q,
    mu=0.0,
    scale=1.0,
):
    """Laplace quantile function."""
    return laplace.ppf(
        q,
        loc=mu,
        scale=scale,
    )


def sample(
    n,
    mu=0.0,
    scale=1.0,
    random_state=None,
):
    """Generate random observations from a Laplace distribution."""
    return laplace.rvs(
        loc=mu,
        scale=scale,
        size=n,
        random_state=random_state,
    )


def fit(data):
    """Fit a Laplace distribution by maximum likelihood."""

    data = np.asarray(data)

    mu_hat, scale_hat = laplace.fit(
        data
    )

    log_likelihood = np.sum(
        laplace.logpdf(
            data,
            loc=mu_hat,
            scale=scale_hat,
        )
    )

    params = {
        "mu": mu_hat,
        "scale": scale_hat,
    }

    return fit_result(
        params,
        log_likelihood,
        len(data),
    )