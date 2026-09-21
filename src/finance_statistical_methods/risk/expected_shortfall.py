"""Expected Shortfall measures."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .var import empirical_var


def empirical_expected_shortfall(losses: ArrayLike, alpha: float = 0.05) -> float:
    """Estimate Expected Shortfall from observed losses."""
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1:
        raise ValueError("losses must be one-dimensional")
    if values.size == 0:
        raise ValueError("losses cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("losses must contain only finite values")

    var = empirical_var(values, alpha)
    tail = values[values >= var]
    return float(np.mean(tail))
