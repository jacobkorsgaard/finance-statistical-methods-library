"""Univariate Normal distribution.

Parameterization
----------------
X ~ N(mu, sigma^2)

Functions
---------
pdf
cdf
quantile
sample
fit
"""

import numpy as np
from scipy.stats import norm

from .fit_statistics import fit_result


def pdf(x, mu=0.0, sigma=1.0):
    """Evaluate the Normal probability density function."""
    return norm.pdf(
        x,
        loc=mu,
        scale=sigma,
    )


def cdf(x, mu=0.0, sigma=1.0):
    """Evaluate the Normal cumulative distribution function."""
    return norm.cdf(
        x,
        loc=mu,
        scale=sigma,
    )


def quantile(q, mu=0.0, sigma=1.0):
    """Evaluate the Normal quantile function."""
    return norm.ppf(
        q,
        loc=mu,
        scale=sigma,
    )


def sample(
    n,
    mu=0.0,
    sigma=1.0,
    random_state=None,
):
    """Generate observations from a Normal distribution."""
    return norm.rvs(
        loc=mu,
        scale=sigma,
        size=n,
        random_state=random_state,
    )


def fit(data):
    """Estimate mu and sigma by maximum likelihood."""

    data = np.asarray(data)

    mu_hat, sigma_hat = norm.fit(data)

    log_likelihood = np.sum(
        norm.logpdf(
            data,
            loc=mu_hat,
            scale=sigma_hat,
        )
    )

    params = {
        "mu": mu_hat,
        "sigma": sigma_hat,
    }

    return fit_result(
        params,
        log_likelihood,
        len(data),
    )