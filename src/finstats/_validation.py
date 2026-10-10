"""Private input contracts shared by custom statistical implementations."""

from numbers import Integral
import numpy as np
from numpy.typing import ArrayLike, NDArray


def _sample(x: ArrayLike) -> NDArray[np.float64]:
    """Validate observations using the earlier package's sample contract."""
    if np.iscomplexobj(x):
        raise ValueError("x must contain real observations")
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a nonempty one-dimensional sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite observations")
    return values


def _integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite_scalar(value: float, name: str) -> float:
    if np.iscomplexobj(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_size(n: int) -> None:
    _integer(n, "n")
    if n == 0:
        raise ValueError("n must be positive")
