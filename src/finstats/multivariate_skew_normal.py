"""Location/scale multivariate skew-normal density, moments, sampling and MLE.

Uses 2 phi_K(x; location, scale) Phi(alpha' D^-1 (x-location)),
where D contains marginal scale standard deviations. Scale is not covariance.
This extends the standard skew-normal construction used in Notebook 02.
"""

import numpy as np
from scipy import optimize, stats


def _parameters(location, scale, shape):
    if any(np.iscomplexobj(v) for v in (location, scale, shape)):
        raise ValueError("parameters must be real")
    mu, matrix, alpha = (np.asarray(v, dtype=float) for v in (location, scale, shape))
    if mu.ndim != 1 or not mu.size or alpha.shape != mu.shape or matrix.shape != (mu.size, mu.size):
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
    omega = matrix/np.outer(sd, sd)
    delta = omega@alpha/np.sqrt(1+alpha@omega@alpha)
    return mu, matrix, alpha, sd, delta


def multivariate_skew_normal_logpdf(x, location, scale, shape):
    """Log density for finite real points shaped (..., K); returns shape (...)."""
    mu, matrix, alpha, sd, _ = _parameters(location, scale, shape)
    if np.iscomplexobj(x):
        raise ValueError("points must be real")
    points = np.asarray(x, dtype=float)
    if points.ndim == 0 or points.shape[-1] != mu.size or not points.size or not np.isfinite(points).all():
        raise ValueError("points must be finite with shape (..., K)")
    return np.log(2)+stats.multivariate_normal.logpdf(points, mean=mu, cov=matrix)+stats.norm.logcdf(((points-mu)/sd)@alpha)


def multivariate_skew_normal_moments(location, scale, shape):
    """Return population mean and covariance, distinct from location and scale."""
    mu, matrix, _, sd, delta = _parameters(location, scale, shape)
    shift = sd*delta
    return mu+np.sqrt(2/np.pi)*shift, matrix-(2/np.pi)*np.outer(shift, shift)


def multivariate_skew_normal_rvs(location, scale, shape, size, random_state):
    """Draw (size, K) observations via the latent half-normal representation.

    random_state is a NumPy Generator or a seed accepted by default_rng.
    """
    mu, matrix, _, sd, delta = _parameters(location, scale, shape)
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    rng = np.random.default_rng(random_state)
    d = sd*delta
    residual_cov = matrix-np.outer(d,d)
    residual = rng.multivariate_normal(np.zeros(mu.size), residual_cov, size=size)
    return mu+np.abs(rng.standard_normal(size))[:,None]*d+residual


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
    if x.ndim != 2 or x.shape[1] < 2 or len(x) <= x.shape[1] or not np.isfinite(x).all():
        raise ValueError("need finite observations, more rows than variables, and at least two variables")
    center, sd = x.mean(axis=0), x.std(axis=0)
    if np.any(sd == 0):
        raise ValueError("sample covariance must be positive definite")
    z = (x-center)/sd
    covariance = np.cov(z, rowvar=False, ddof=0)
    if np.linalg.eigvalsh(covariance).min() <= 1e-12:
        raise ValueError("sample covariance must be positive definite")
    k = x.shape[1]
    rows, cols = np.tril_indices(k)
    diag = rows == cols
    def unpack(p):
        factor = np.zeros((k,k))
        entries = p[k:-k].copy()
        entries[diag] = np.exp(entries[diag])
        factor[rows,cols] = entries
        return p[:k], factor@factor.T, p[-k:]
    def objective(p):
        mu, matrix, alpha = unpack(p)
        return -multivariate_skew_normal_logpdf(z, mu, matrix, alpha).sum()
    bounds = [(None,None)]*k+[(-10,10) if d else (None,None) for d in diag]+[(-30,30)]*k
    starts = [np.zeros(k)]
    for j in range(k):
        for sign in [-1,1]:
            alpha = np.zeros(k)
            alpha[j] = 2*sign
            starts.append(alpha)
    diagnostics, candidates = [], []
    for alpha in starts:
        # Match the initial model moments to sample moments, while introducing skewness.
        omega = covariance/np.outer(np.sqrt(np.diag(covariance)), np.sqrt(np.diag(covariance)))
        delta = omega@alpha/np.sqrt(1+alpha@omega@alpha)
        d = np.sqrt(np.diag(covariance))*delta
        matrix = covariance+(2/np.pi)*np.outer(d,d)
        factor = np.linalg.cholesky(matrix)
        entries = factor[rows,cols].copy()
        entries[diag] = np.log(entries[diag])
        initial = np.r_[-np.sqrt(2/np.pi)*d, entries, alpha]
        fit = optimize.minimize(objective, initial, method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": 1500, "ftol": 1e-11})
        ll = -fit.fun-len(x)*np.log(sd).sum()
        diagnostics.append({"start_shape": alpha.tolist(), "success": bool(fit.success),
                            "log_likelihood": float(ll), "message": str(fit.message)})
        if fit.success and np.isfinite(ll):
            candidates.append((ll,fit.x))
    if not candidates:
        raise RuntimeError("all skew-normal optimization starts failed")
    ll, p = max(candidates, key=lambda item:item[0])
    mu, matrix, alpha = unpack(p)
    return {"location": center+sd*mu, "scale": matrix*np.outer(sd,sd), "shape": alpha,
            "log_likelihood": float(ll), "starts": diagnostics,
            "shape_bound_hit": bool(np.any(np.abs(alpha)>=29.99))}
