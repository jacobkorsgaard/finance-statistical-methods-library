"""Graphical diagnostics for fitted univariate distributions.

The functions in this module compare an observed sample with a fitted
parametric distribution.

Diagnostics
-----------
plot_distribution
    Compares the empirical distribution with the fitted probability
    density function using a histogram and, optionally, a kernel
    density estimate.

plot_cdf
    Compares the empirical cumulative distribution function (ECDF)
    with the fitted parametric CDF.

qq_plot
    Compares sample quantiles with theoretical quantiles from the
    fitted distribution. A close-to-linear relationship indicates
    that the fitted distribution provides a reasonable description
    of the sample distribution.

Notes
-----
These diagnostics are graphical tools rather than formal hypothesis
tests. They are intended to reveal differences in location, scale,
skewness, tail behaviour, and overall distributional shape.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_distribution(
    data,
    distribution,
    fit,
    bins=30,
    show_kde=True,
    ax=None,
):
    """
    Plot histogram, optional KDE, and fitted PDF.
    """

    data = np.asarray(data)

    if ax is None:
        _, ax = plt.subplots()

    lower = np.min(data)
    upper = np.max(data)

    padding = 0.05 * (
        upper - lower
    )

    grid = np.linspace(
        lower - padding,
        upper + padding,
        500,
    )

    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.3,
        label="Histogram",
    )

    if show_kde:
        kde = gaussian_kde(data)

        ax.plot(
            grid,
            kde(grid),
            label="KDE",
        )

    fitted_pdf = distribution.pdf(
        grid,
        **fit["params"],
    )

    ax.plot(
        grid,
        fitted_pdf,
        label="Fitted PDF",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()

    return ax

def qq_plot(
    data,
    distribution,
    fit,
    ax=None,
):
    """
    Compare sample quantiles with fitted theoretical quantiles.
    """

    data = np.asarray(data)

    if ax is None:
        _, ax = plt.subplots()

    sample_quantiles = np.sort(data)

    n = len(data)

    probabilities = (
        np.arange(1, n + 1) - 0.5
    ) / n

    theoretical_quantiles = distribution.quantile(
        probabilities,
        **fit["params"],
    )

    ax.scatter(
        theoretical_quantiles,
        sample_quantiles,
    )

    lower = min(
        np.min(theoretical_quantiles),
        np.min(sample_quantiles),
    )

    upper = max(
        np.max(theoretical_quantiles),
        np.max(sample_quantiles),
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
    )

    ax.set_xlabel(
        "Theoretical quantiles"
    )

    ax.set_ylabel(
        "Sample quantiles"
    )

    ax.set_title("Q-Q plot")

    return ax

def plot_cdf(
    data,
    distribution,
    fit,
    ax=None,
):
    """
    Compare the empirical CDF with the fitted parametric CDF.
    """

    data = np.asarray(data)

    if ax is None:
        _, ax = plt.subplots()

    sorted_data = np.sort(data)

    empirical_cdf = (
        np.arange(1, len(data) + 1)
        / len(data)
    )

    lower = np.min(data)
    upper = np.max(data)

    padding = 0.05 * (
        upper - lower
    )

    grid = np.linspace(
        lower - padding,
        upper + padding,
        500,
    )

    fitted_cdf = distribution.cdf(
        grid,
        **fit["params"],
    )

    ax.step(
        sorted_data,
        empirical_cdf,
        where="post",
        label="Empirical CDF",
    )

    ax.plot(
        grid,
        fitted_cdf,
        label="Fitted CDF",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Probability")
    ax.legend()

    return ax

