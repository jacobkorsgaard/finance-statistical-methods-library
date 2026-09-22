"""Univariate Lognormal distribution.

Parameterization
----------------
log(X) ~ N(mu, sigma^2)

Hence mu and sigma refer to the distribution of log(X), not X.
SciPy uses s=sigma and scale=exp(mu).

Functions
---------
pdf
cdf
quantile
sample
fit
"""

import numpy as np
from scipy.stats import lognorm

from .fit_statistics import fit_result


def pdf(x, mu=0.0, sigma=1.0):
    """Lognormal probability density function."""
    return lognorm.pdf(
        x,
        s=sigma,
        loc=0.0,
        scale=np.exp(mu),
    )


def cdf(x, mu=0.0, sigma=1.0):
    """Lognormal cumulative distribution function."""
    return lognorm.cdf(
        x,
        s=sigma,
        loc=0.0,
        scale=np.exp(mu),
    )


def quantile(q, mu=0.0, sigma=1.0):
    """Lognormal quantile function."""
    return lognorm.ppf(
        q,
        s=sigma,
        loc=0.0,
        scale=np.exp(mu),
    )


def sample(
    n,
    mu=0.0,
    sigma=1.0,
    random_state=None,
):
    """Generate random observations from a Lognormal distribution."""
    return lognorm.rvs(
        s=sigma,
        loc=0.0,
        scale=np.exp(mu),
        size=n,
        random_state=random_state,
    )


def fit(data):
    """Fit a two-parameter Lognormal distribution by MLE."""

    data = np.asarray(data)

    if np.any(data <= 0):
        raise ValueError(
            "Lognormal data must be strictly positive."
        )

    sigma_hat, _, scale_hat = lognorm.fit(
        data,
        floc=0.0,
    )

    mu_hat = np.log(scale_hat)

    log_likelihood = np.sum(
        lognorm.logpdf(
            data,
            s=sigma_hat,
            loc=0.0,
            scale=np.exp(mu_hat),
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