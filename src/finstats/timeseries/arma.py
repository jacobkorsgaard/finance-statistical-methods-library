"""Root checks reuse AR and MA conventions; estimation belongs to statsmodels."""
from .ar import ar_roots, is_stationary_ar as is_stationary
from .ma import ma_roots, is_invertible_ma as is_invertible


def arma_roots(ar_coefficients, ma_coefficients):
    """Return (AR roots, MA roots) for the course minus-AR/plus-MA convention."""
    return ar_roots(ar_coefficients), ma_roots(ma_coefficients)
