"""Differencing and Gaussian random walks with explicit initial-observation indexing."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..descriptive import _sample, _integer
from ..inference import _finite_scalar
from .ar import _positive_size


def difference(x: ArrayLike, order: int = 1) -> NDArray:
    """Repeated adjacent differences; order zero copies, order<T is required."""
    values = _sample(x)
    _integer(order, "order")
    if order >= values.size:
        raise ValueError("order must be smaller than series length")
    return np.diff(values, n=order)


def simulate_random_walk(n: int, sigma: float = 1., initial: float = 0., rng=None) -> NDArray:
    """Return (Y_0,...,Y_n), so n increments yield n+1 observations.

    Y_0 is the fixed initial value. differencing returns exactly the n generated
    innovations. Variance at index t is t*sigma**2.
    """
    _positive_size(n)
    sigma, initial = _finite_scalar(sigma,"sigma"), _finite_scalar(initial,"initial")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    innovations = np.random.default_rng(rng).normal(0,sigma,size=n)
    return np.r_[initial, initial+np.cumsum(innovations)]
