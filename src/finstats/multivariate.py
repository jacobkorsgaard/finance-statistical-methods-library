"""Custom multivariate Laplace and skew-normal densities, moments and estimation.

Rows are observations. A scale matrix is not generally a covariance matrix.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats


def bivariate_symmetric_laplace_pdf(
    x: ArrayLike, location: ArrayLike, covariance: ArrayLike
) -> NDArray[np.float64]:
    """Density of location + sqrt(W)*Z, W~Exp(1), Z~N_2(0, covariance).

    W and Z are independent. The specified matrix is the actual covariance,
    not the scale of the univariate Laplace marginals. Points have shape
    (..., 2); the return shape is (...). All inputs must be finite and real,
    and covariance must be symmetric positive definite. The density is +inf
    at its location, an integrable singularity rather than a point mass.
    """
    from scipy.special import k0

    if any(np.iscomplexobj(value) for value in (x, location, covariance)):
        raise ValueError("points, location and covariance must be real")
    points = np.asarray(x, dtype=float)
    center = np.asarray(location, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 2 or points.size == 0:
        raise ValueError("points must have nonempty shape (..., 2)")
    if center.shape != (2,) or sigma.shape != (2, 2):
        raise ValueError("location and covariance must have shapes (2,) and (2, 2)")
    if not all(np.all(np.isfinite(value)) for value in (points, center, sigma)):
        raise ValueError("all inputs must be finite")
    tolerance = 1e-12 * float(np.max(np.abs(sigma)))
    if not np.allclose(sigma, sigma.T, rtol=0, atol=tolerance):
        raise ValueError("covariance must be symmetric")
    sigma = sigma / 2 + sigma.T / 2
    try:
        factor = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error
    standardized = np.linalg.solve(factor, (points - center).reshape(-1, 2).T)
    radius = np.sqrt(2 * np.sum(standardized**2, axis=0))
    density = k0(radius) / (np.pi * np.prod(np.diag(factor)))
    return np.asarray(density.reshape(points.shape[:-1]), dtype=float)


def _parameters(location, scale, shape):
    if any(np.iscomplexobj(v) for v in (location, scale, shape)):
        raise ValueError("parameters must be real")
    mu, matrix, alpha = (np.asarray(v, dtype=float) for v in (location, scale, shape))
    if (
        mu.ndim != 1
        or not mu.size
        or alpha.shape != mu.shape
        or matrix.shape != (mu.size, mu.size)
    ):
        raise ValueError("incompatible parameter dimensions")
    if not all(np.isfinite(v).all() for v in (mu, matrix, alpha)):
        raise ValueError("parameters must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0):
        raise ValueError("scale must be symmetric positive definite")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("scale must be symmetric positive definite") from error
    sd = np.sqrt(np.diag(matrix))
    omega = matrix / np.outer(sd, sd)
    delta = omega @ alpha / np.sqrt(1 + alpha @ omega @ alpha)
    return mu, matrix, alpha, sd, delta


def multivariate_skew_normal_logpdf(x, location, scale, shape):
    """Log density for finite real points shaped (..., K); returns shape (...)."""
    mu, matrix, alpha, sd, _ = _parameters(location, scale, shape)
    if np.iscomplexobj(x):
        raise ValueError("points must be real")
    points = np.asarray(x, dtype=float)
    if (
        points.ndim == 0
        or points.shape[-1] != mu.size
        or not points.size
        or not np.isfinite(points).all()
    ):
        raise ValueError("points must be finite with shape (..., K)")
    return (
        np.log(2)
        + stats.multivariate_normal.logpdf(points, mean=mu, cov=matrix)
        + stats.norm.logcdf(((points - mu) / sd) @ alpha)
    )


def multivariate_skew_normal_moments(location, scale, shape):
    """Return population mean and covariance, distinct from location and scale."""
    mu, matrix, _, sd, delta = _parameters(location, scale, shape)
    shift = sd * delta
    return mu + np.sqrt(2 / np.pi) * shift, matrix - (2 / np.pi) * np.outer(
        shift, shift
    )


def multivariate_skew_normal_rvs(location, scale, shape, size, random_state):
    """Draw (size, K) observations via the latent half-normal representation.

    random_state is a NumPy Generator or a seed accepted by default_rng.
    """
    mu, matrix, _, sd, delta = _parameters(location, scale, shape)
    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 1
    ):
        raise ValueError("size must be a positive integer")
    rng = np.random.default_rng(random_state)
    d = sd * delta
    residual_cov = matrix - np.outer(d, d)
    residual = rng.multivariate_normal(np.zeros(mu.size), residual_cov, size=size)
    return mu + np.abs(rng.standard_normal(size))[:, None] * d + residual


def multivariate_skew_normal_mle(data):
    """Multi-start numerical fit; returns location, scale, shape and diagnostics.

    Positive definite scale uses a Cholesky factor. For stable finite numerical
    fits, shape entries are bounded to [-30, 30], and log Cholesky diagonals
    to [-10, 10] in standardized coordinates. Starts reduce local-optimum risk
    without guaranteeing a global solution. Bound hits are reported explicitly.
    """
    if np.iscomplexobj(data):
        raise ValueError("data must be real")
    x = np.asarray(data, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] < 2
        or len(x) <= x.shape[1]
        or not np.isfinite(x).all()
    ):
        raise ValueError(
            "need finite observations, more rows than variables, and at least two variables"
        )
    center, sd = x.mean(axis=0), x.std(axis=0)
    if np.any(sd == 0):
        raise ValueError("sample covariance must be positive definite")
    z = (x - center) / sd
    covariance = np.cov(z, rowvar=False, ddof=0)
    if np.linalg.eigvalsh(covariance).min() <= 1e-12:
        raise ValueError("sample covariance must be positive definite")
    k = x.shape[1]
    rows, cols = np.tril_indices(k)
    diag = rows == cols

    def unpack(p):
        factor = np.zeros((k, k))
        entries = p[k:-k].copy()
        entries[diag] = np.exp(entries[diag])
        factor[rows, cols] = entries
        return p[:k], factor @ factor.T, p[-k:]

    def objective(p):
        mu, matrix, alpha = unpack(p)
        return -multivariate_skew_normal_logpdf(z, mu, matrix, alpha).sum()

    bounds = (
        [(None, None)] * k
        + [(-10, 10) if d else (None, None) for d in diag]
        + [(-30, 30)] * k
    )
    starts = [np.zeros(k)]
    for j in range(k):
        for sign in [-1, 1]:
            alpha = np.zeros(k)
            alpha[j] = 2 * sign
            starts.append(alpha)
    diagnostics, candidates = [], []
    for alpha in starts:
        # Match the initial model moments to sample moments, while introducing skewness.
        omega = covariance / np.outer(
            np.sqrt(np.diag(covariance)), np.sqrt(np.diag(covariance))
        )
        delta = omega @ alpha / np.sqrt(1 + alpha @ omega @ alpha)
        d = np.sqrt(np.diag(covariance)) * delta
        matrix = covariance + (2 / np.pi) * np.outer(d, d)
        factor = np.linalg.cholesky(matrix)
        entries = factor[rows, cols].copy()
        entries[diag] = np.log(entries[diag])
        initial = np.r_[-np.sqrt(2 / np.pi) * d, entries, alpha]
        fit = optimize.minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-11},
        )
        ll = -fit.fun - len(x) * np.log(sd).sum()
        diagnostics.append(
            {
                "start_shape": alpha.tolist(),
                "success": bool(fit.success),
                "log_likelihood": float(ll),
                "message": str(fit.message),
            }
        )
        if fit.success and np.isfinite(ll):
            candidates.append((ll, fit.x))
    if not candidates:
        raise RuntimeError("all skew-normal optimization starts failed")
    ll, p = max(candidates, key=lambda item: item[0])
    mu, matrix, alpha = unpack(p)
    return {
        "location": center + sd * mu,
        "scale": matrix * np.outer(sd, sd),
        "shape": alpha,
        "log_likelihood": float(ll),
        "starts": diagnostics,
        "shape_bound_hit": bool(np.any(np.abs(alpha) >= 29.99)),
    }
