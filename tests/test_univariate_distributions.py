import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from finance_statistical_methods.distributions.univariate import (
    normal,
    lognormal,
    laplace,
    student_t,
    skew_normal,
)

from finance_statistical_methods.distributions.univariate.fit_statistics import (
    aic,
    bic,
)

from finance_statistical_methods.distributions.univariate.diagnostics import (
    plot_distribution,
    plot_cdf,
    qq_plot,
)


# ---------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------

def test_normal_reference_values():
    """Check well-known values of the standard Normal distribution."""

    assert normal.pdf(0.0) == pytest.approx(
        1 / np.sqrt(2 * np.pi)
    )

    assert normal.cdf(0.0) == pytest.approx(
        0.5
    )

    assert normal.quantile(0.5) == pytest.approx(
        0.0
    )


def test_normal_quantile_inverse_and_fit():
    """Check inverse CDF and Normal MLE."""

    q = np.array([0.1, 0.5, 0.9])

    x = normal.quantile(
        q,
        mu=1.0,
        sigma=2.0,
    )

    recovered_q = normal.cdf(
        x,
        mu=1.0,
        sigma=2.0,
    )

    assert np.allclose(
        recovered_q,
        q,
    )

    data = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        8.0,
    ])

    fit = normal.fit(data)

    assert fit["params"]["mu"] == pytest.approx(
        np.mean(data)
    )

    # Normal MLE uses n in the denominator: ddof=0.
    assert fit["params"]["sigma"] == pytest.approx(
        np.std(data, ddof=0)
    )


# ---------------------------------------------------------------------
# Lognormal
# ---------------------------------------------------------------------

def test_lognormal():
    """Check Lognormal reference values, MLE and support."""

    assert lognormal.cdf(
        1.0,
        mu=0.0,
        sigma=1.0,
    ) == pytest.approx(0.5)

    assert lognormal.quantile(
        0.5,
        mu=0.0,
        sigma=1.0,
    ) == pytest.approx(1.0)

    data = np.exp(
        np.array([
            -0.5,
            0.0,
            0.2,
            0.7,
            1.1,
        ])
    )

    fit = lognormal.fit(data)

    log_data = np.log(data)

    assert fit["params"]["mu"] == pytest.approx(
        np.mean(log_data)
    )

    assert fit["params"]["sigma"] == pytest.approx(
        np.std(log_data, ddof=0)
    )

    with pytest.raises(ValueError):
        lognormal.fit([
            1.0,
            0.0,
            2.0,
        ])


# ---------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------

def test_laplace():
    """Check Laplace reference values and closed-form MLE."""

    assert laplace.pdf(
        2.0,
        mu=2.0,
        scale=3.0,
    ) == pytest.approx(
        1 / 6
    )

    assert laplace.cdf(
        2.0,
        mu=2.0,
        scale=3.0,
    ) == pytest.approx(
        0.5
    )

    data = np.array([
        -3.0,
        -1.0,
        0.0,
        0.5,
        5.0,
    ])

    fit = laplace.fit(data)

    mu_hat = np.median(data)

    scale_hat = np.mean(
        np.abs(data - mu_hat)
    )

    assert fit["params"]["mu"] == pytest.approx(
        mu_hat
    )

    assert fit["params"]["scale"] == pytest.approx(
        scale_hat
    )


# ---------------------------------------------------------------------
# Student-t
# ---------------------------------------------------------------------

def test_student_t():
    """Check Student-t symmetry, inverse CDF and fitted parameters."""

    assert student_t.cdf(
        2.0,
        mu=2.0,
        scale=1.5,
        df=5.0,
    ) == pytest.approx(
        0.5
    )

    q = np.array([
        0.1,
        0.5,
        0.9,
    ])

    x = student_t.quantile(
        q,
        mu=0.2,
        scale=1.4,
        df=7.0,
    )

    assert np.allclose(
        student_t.cdf(
            x,
            mu=0.2,
            scale=1.4,
            df=7.0,
        ),
        q,
    )

    data = student_t.sample(
        400,
        mu=0.3,
        scale=1.2,
        df=6.0,
        random_state=123,
    )

    fit = student_t.fit(data)

    assert fit["params"]["scale"] > 0
    assert fit["params"]["df"] > 0
    assert np.isfinite(
        fit["log_likelihood"]
    )


# ---------------------------------------------------------------------
# Skew-normal
# ---------------------------------------------------------------------

def test_skew_normal():
    """Check alpha=0 reduction to Normal and fitted parameters."""

    x = np.linspace(
        -3.0,
        3.0,
        21,
    )

    assert np.allclose(
        skew_normal.pdf(
            x,
            mu=1.0,
            scale=2.0,
            alpha=0.0,
        ),
        normal.pdf(
            x,
            mu=1.0,
            sigma=2.0,
        ),
    )

    data = skew_normal.sample(
        400,
        mu=-0.2,
        scale=1.3,
        alpha=4.0,
        random_state=123,
    )

    fit = skew_normal.fit(data)

    assert fit["params"]["scale"] > 0

    assert np.isfinite(
        fit["params"]["alpha"]
    )

    assert np.isfinite(
        fit["log_likelihood"]
    )


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def test_sample_sizes():
    """All distributions should generate the requested sample size."""

    assert len(
        normal.sample(
            25,
            random_state=1,
        )
    ) == 25

    assert len(
        lognormal.sample(
            25,
            random_state=1,
        )
    ) == 25

    assert len(
        laplace.sample(
            25,
            random_state=1,
        )
    ) == 25

    assert len(
        student_t.sample(
            25,
            random_state=1,
        )
    ) == 25

    assert len(
        skew_normal.sample(
            25,
            random_state=1,
        )
    ) == 25


# ---------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------

def test_information_criteria():
    """Check AIC and BIC against their definitions."""

    assert aic(
        -100.0,
        3,
    ) == pytest.approx(
        206.0
    )

    expected_bic = (
        200.0
        + 3 * np.log(50)
    )

    assert bic(
        -100.0,
        3,
        50,
    ) == pytest.approx(
        expected_bic
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def test_diagnostics_return_axes():
    """Diagnostic functions should return Matplotlib Axes objects."""

    data = normal.sample(
        100,
        random_state=123,
    )

    fit = normal.fit(data)

    ax1 = plot_distribution(
        data,
        normal,
        fit,
    )

    ax2 = plot_cdf(
        data,
        normal,
        fit,
    )

    ax3 = qq_plot(
        data,
        normal,
        fit,
    )

    assert ax1 is not None
    assert ax2 is not None
    assert ax3 is not None

    plt.close("all")