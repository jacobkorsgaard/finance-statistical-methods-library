import numpy as np


def aic(log_likelihood, n_params):
    """Akaike Information Criterion."""
    return -2.0 * log_likelihood + 2.0 * n_params


def bic(log_likelihood, n_params, n_obs):
    """Bayesian Information Criterion."""
    return (
        -2.0 * log_likelihood
        + np.log(n_obs) * n_params
    )


def fit_result(params, log_likelihood, n_obs):
    """Create a common result dictionary for fitted distributions."""

    n_params = len(params)

    return {
        "params": params,
        "log_likelihood": float(log_likelihood),
        "aic": float(
            aic(log_likelihood, n_params)
        ),
        "bic": float(
            bic(log_likelihood, n_params, n_obs)
        ),
    }