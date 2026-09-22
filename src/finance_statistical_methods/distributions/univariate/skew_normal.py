"""Univariate skew-normal distribution.

Parameterization
----------------
mu    : location
scale : positive scale parameter
alpha : shape/skewness parameter
alpha = 0 reduces to the Normal distribution.

Functions
---------
pdf
cdf
quantile
sample
fit
"""

import numpy as np
from scipy.stats import skewnorm
from .fit_statistics import fit_result


def pdf(
    x,
    mu=0.0,
    scale=1.0,
    alpha=0.0,
):
    """Skew-normal probability density function."""
    return skewnorm.pdf(
        x,
        a=alpha,
        loc=mu,
        scale=scale,
    )


def cdf(
    x,
    mu=0.0,
    scale=1.0,
    alpha=0.0,
):
    """Skew-normal cumulative distribution function."""
    return skewnorm.cdf(
        x,
        a=alpha,
        loc=mu,
        scale=scale,
    )


def quantile(
    q,
    mu=0.0,
    scale=1.0,
    alpha=0.0,
):
    """Skew-normal quantile function."""
    return skewnorm.ppf(
        q,
        a=alpha,
        loc=mu,
        scale=scale,
    )


def sample(
    n,
    mu=0.0,
    scale=1.0,
    alpha=0.0,
    random_state=None,
):
    """Generate random observations from a skew-normal distribution."""
    return skewnorm.rvs(
        a=alpha,
        loc=mu,
        scale=scale,
        size=n,
        random_state=random_state,
    )


def fit(data):
    """Fit skew-normal shape, location and scale by MLE."""

    data = np.asarray(data)

    alpha_hat, mu_hat, scale_hat = skewnorm.fit(
        data
    )

    log_likelihood = np.sum(
        skewnorm.logpdf(
            data,
            a=alpha_hat,
            loc=mu_hat,
            scale=scale_hat,
        )
    )

    params = {
        "mu": mu_hat,
        "scale": scale_hat,
        "alpha": alpha_hat,
    }

    return fit_result(
        params,
        log_likelihood,
        len(data),
    )