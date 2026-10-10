"""Time-series sample dependence with explicit denominator conventions."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ._validation import _sample, _integer
from statsmodels.tsa.stattools import acf as _reference_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def sample_autocovariance(x: ArrayLike, lag: int, adjusted: bool = True) -> float:
    """Centered lag product sum divided by T-lag (course), or T if adjusted=False.

    The full sample mean centers both segments. Lag zero uses denominator T.
    """
    values = _sample(x)
    _integer(lag, "lag")
    if lag >= values.size:
        raise ValueError("lag must be smaller than sample length")
    centered = values - np.mean(values)
    denominator = values.size - lag if adjusted else values.size
    return float(np.dot(centered[lag:], centered[: values.size - lag]) / denominator)


def sample_autocorrelation(x: ArrayLike, lag: int, adjusted: bool = True) -> float:
    """Return sample lag covariance / lag-zero covariance; reject constants."""
    variance = sample_autocovariance(x, 0)
    if variance == 0:
        raise ValueError("autocorrelation requires a nonconstant series")
    return sample_autocovariance(x, lag, adjusted=adjusted) / variance


def acf(x: ArrayLike, max_lag: int, adjusted: bool = True) -> NDArray[np.float64]:
    """Return sample ACF at lags 0..max_lag, using the stated denominator."""
    values = _sample(x)
    _integer(max_lag, "max_lag")
    if max_lag >= values.size:
        raise ValueError("max_lag must be smaller than sample length")
    if sample_autocovariance(values, 0) == 0:
        raise ValueError("autocorrelation requires a nonconstant series")
    return _reference_acf(values, nlags=max_lag, adjusted=adjusted, fft=False)


def ljung_box(x: ArrayLike, lags: int, model_df: int = 0) -> dict[str, float | int]:
    """Single-horizon Ljung-Box Q and chi-square p-value, df=lags-model_df.

    Uses denominator-T sample autocovariances to match standard production
    implementations, unlike the default course-adjusted sample ACF. Parameters
    fitted in ARMA dependence consume p+q degrees of freedom. Calibration is
    approximate and does not establish independence or account for all forms
    of conditional heteroskedasticity.
    """
    values = _sample(x)
    _integer(lags, "lags")
    _integer(model_df, "model_df")
    if lags <= model_df or lags >= values.size:
        raise ValueError("require model_df < lags < sample length")
    if sample_autocovariance(values, 0) == 0:
        raise ValueError("autocorrelation requires a nonconstant series")
    result = acorr_ljungbox(values, lags=[lags], model_df=model_df).iloc[0]
    return {
        "statistic": float(result.lb_stat),
        "p_value": float(result.lb_pvalue),
        "df": lags - model_df,
        "lags": lags,
        "nobs": len(values),
    }


def mcleod_li(residuals: ArrayLike, lags: int) -> dict[str, float | int]:
    """Squared-residual Ljung-Box statistic with zero mean-model df deduction.

    This portmanteau screen follows McLeod-Li logic; after fitting a variance
    model its plain chi-square p-value is a descriptive approximate diagnostic,
    not an exact parameter-adjusted specification test. Conventional squared-
    series calibration also requires finite fourth moments of residuals.
    """
    return ljung_box(_sample(residuals) ** 2, lags, model_df=0)


def standardized_residuals(
    residuals: ArrayLike, conditional_volatility: ArrayLike
) -> NDArray:
    """Divide matched finite residuals by strictly positive conditional SDs."""
    residuals = _sample(residuals)
    volatility = _sample(conditional_volatility)
    if residuals.shape != volatility.shape or np.any(volatility <= 0):
        raise ValueError("volatility must be positive and match residual length")
    return residuals / volatility
