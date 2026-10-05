"""Location-scale transformations and Student-t population moment restrictions.

PDF/CDF evaluation and sampling use mature numerical libraries directly.
The Student-t scale lambda is not its standard deviation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_scale(scale: float) -> None:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and strictly positive")


def location_scale_transform(
    x: ArrayLike, location: float, scale: float
) -> NDArray[np.float64]:
    """Return location + scale*x, with finite real x and positive finite scale.

    Any input shape is preserved; scalar input returns a zero-dimensional array.
    """
    _positive_scale(scale)
    if not np.isfinite(location):
        raise ValueError("location must be finite")
    if np.iscomplexobj(x):
        raise ValueError("x must be real")
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain finite values")
    return np.asarray(location + scale * values, dtype=float)


def student_t_variance(df: float, scale: float = 1.0) -> float:
    """Return lambda**2*nu/(nu-2); reject nu <= 2 (no finite variance)."""
    _positive_scale(scale)
    if not np.isfinite(df) or df <= 2:
        raise ValueError("finite Student-t variance requires finite df > 2")
    return float(scale**2 * (1.0 + 2.0 / (df - 2.0)))


def student_t_kurtosis(df: float, excess: bool = False) -> float:
    """Return 3+6/(nu-4), or excess 6/(nu-4); require finite nu > 4."""
    if not np.isfinite(df) or df <= 4:
        raise ValueError("finite Student-t kurtosis requires finite df > 4")
    return float(6.0 / (df - 4.0) + (0.0 if excess else 3.0))
