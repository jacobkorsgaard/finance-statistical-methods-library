"""Univariate location-scale Student-t distribution.

Parameterization
----------------
X ~ t(mu, scale, df)

Note that scale is not generally the standard deviation.
For df > 2:

    Var(X) = scale^2 * df / (df - 2)

Functions
---------
pdf
cdf
quantile
sample
fit
"""

import numpy as np
from scipy.stats import t

from .fit_statistics import fit_result


def pdf(
    x,
    mu=0.0,
    scale=1.0,
    df=5.0,
):
    """Student-t probability density function."""
    return t.pdf(
        x,
        df=df,
        loc=mu,
        scale=scale,
    )


def cdf(
    x,
    mu=0.0,
    scale=1.0,
    df=5.0,
):
    """Student-t cumulative distribution function."""
    return t.cdf(
        x,
        df=df,
        loc=mu,
        scale=scale,
    )


def quantile(
    q,
    mu=0.0,
    scale=1.0,
    df=5.0,
):
    """Student-t quantile function."""
    return t.ppf(
        q,
        df=df,
        loc=mu,
        scale=scale,
    )


def sample(
    n,
    mu=0.0,
    scale=1.0,
    df=5.0,
    random_state=None,
):
    """Generate random observations from a Student-t distribution."""
    return t.rvs(
        df=df,
        loc=mu,
        scale=scale,
        size=n,
        random_state=random_state,
    )


def fit(data):
    """Fit Student-t location, scale and df by MLE."""

    data = np.asarray(data)

    df_hat, mu_hat, scale_hat = t.fit(
        data
    )

    log_likelihood = np.sum(
        t.logpdf(
            data,
            df=df_hat,
            loc=mu_hat,
            scale=scale_hat,
        )
    )

    params = {
        "mu": mu_hat,
        "scale": scale_hat,
        "df": df_hat,
    }

    return fit_result(
        params,
        log_likelihood,
        len(data),
    )