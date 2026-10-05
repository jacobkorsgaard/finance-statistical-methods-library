"""Matched out-of-sample forecast errors in the original observation units."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..descriptive import _sample


def forecast_errors(actual: ArrayLike, forecast: ArrayLike) -> NDArray:
    """Return actual-forecast; require matched finite one-dimensional arrays."""
    actual, forecast = _sample(actual), _sample(forecast)
    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have the same shape")
    return actual-forecast


def forecast_mse(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Average squared forecast error; units are observation-units squared."""
    return float(np.mean(forecast_errors(actual,forecast)**2))


def forecast_rmse(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Square root of forecast MSE, in observation units."""
    return float(np.sqrt(forecast_mse(actual,forecast)))


def forecast_mae(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Average absolute forecast error, in observation units."""
    return float(np.mean(np.abs(forecast_errors(actual,forecast))))
