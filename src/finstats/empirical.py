"""Empirical distribution objects with explicit order-statistic conventions."""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from ._validation import _sample


def empirical_cdf(x: ArrayLike) -> Callable[[ArrayLike], NDArray[np.float64]]:
    """Return a right-continuous ECDF evaluator, including ties and +/- infinity.

    Sorting once makes repeated evaluation efficient. Scalar evaluation returns
    a zero-dimensional array. NaN/complex evaluation points are rejected.
    """
    ordered = np.sort(_sample(x)).copy()

    def evaluate(points: ArrayLike) -> NDArray[np.float64]:
        if np.iscomplexobj(points):
            raise ValueError("evaluation points must be real")
        points = np.asarray(points, dtype=float)
        if np.any(np.isnan(points)):
            raise ValueError("evaluation points must not contain NaN")
        return np.asarray(np.searchsorted(ordered, points, side="right") / ordered.size)

    return evaluate


def empirical_quantile(x: ArrayLike, q: float) -> float:
    """Generalized-inverse ECDF quantile (ceil(n*q) order statistic).

    Endpoints return the sample minimum and
    maximum. This differs from NumPy's default interpolated quantile.
    """
    values = _sample(x)
    if np.ndim(q) != 0 or np.iscomplexobj(q) or not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be a finite scalar in [0,1]")
    return float(np.quantile(values, q, method="inverted_cdf"))


def qq_data(x: ArrayLike, distribution=stats.norm) -> tuple[NDArray, NDArray]:
    """Return (population quantiles, sorted sample), probabilities (i-.5)/n.

    distribution is a SciPy distribution or frozen distribution exposing ppf.
    Its parameters must match the reference distribution intended by the reader.
    """
    values = np.sort(_sample(x))
    probabilities = (np.arange(1, values.size + 1) - 0.5) / values.size
    theoretical = np.asarray(distribution.ppf(probabilities), dtype=float)
    if theoretical.shape != values.shape or not np.all(np.isfinite(theoretical)):
        raise ValueError(
            "reference quantiles must be finite and match the sample shape"
        )
    return theoretical, values
