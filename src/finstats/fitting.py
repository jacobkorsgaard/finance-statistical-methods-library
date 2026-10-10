"""Comparable univariate fit summaries using SciPy's existing estimators.

No density, quantile, sampler or optimizer is reimplemented. The result keeps
course parameter labels and includes a frozen SciPy distribution for evaluation.
"""

import numpy as np
from scipy import stats
from ._validation import _sample


def fit_distribution(data, distribution):
    """Fit norm, t, laplace, lognorm (loc=0), or skewnorm using SciPy MLE.

    distribution is one of these SciPy generators, or its name as a string.
    The returned dict contains params, distribution, log_likelihood, aic and bic.
    The lognormal params describe log(X); Student-t scale is not its SD.
    Numerical SciPy fits inherit starting-value/local-maximum limitations.
    """
    x = _sample(data)
    if isinstance(distribution, str):
        if distribution not in {"norm", "t", "laplace", "lognorm", "skewnorm"}:
            raise ValueError("unsupported distribution")
        distribution = getattr(stats, distribution)
    name = getattr(distribution, "name", None)
    if name not in {"norm", "t", "laplace", "lognorm", "skewnorm"}:
        raise ValueError("unsupported distribution")
    if name == "lognorm":
        if np.any(x <= 0):
            raise ValueError("lognormal observations must be strictly positive")
        shape, _, scale = distribution.fit(x, floc=0)
        params = {"mu": np.log(scale), "sigma": shape}
        frozen = distribution(shape, loc=0, scale=scale)
    elif name == "t":
        shape, location, scale = distribution.fit(x)
        params = {"mu": location, "scale": scale, "df": shape}
        frozen = distribution(shape, loc=location, scale=scale)
    elif name == "skewnorm":
        shape, location, scale = distribution.fit(x)
        params = {"mu": location, "scale": scale, "alpha": shape}
        frozen = distribution(shape, loc=location, scale=scale)
    else:
        location, scale = distribution.fit(x)
        params = {"mu": location, "sigma" if name == "norm" else "scale": scale}
        frozen = distribution(loc=location, scale=scale)
    ll = float(frozen.logpdf(x).sum())
    count = len(params)
    return {
        "params": params,
        "distribution": frozen,
        "log_likelihood": ll,
        "aic": float(-2 * ll + 2 * count),
        "bic": float(-2 * ll + np.log(len(x)) * count),
    }
