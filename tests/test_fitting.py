"""Fit summaries retain SciPy estimates and likelihood/criterion conventions."""

import numpy as np
import pytest
from scipy import stats
from finstats.fitting import fit_distribution


@pytest.mark.parametrize("family", ["norm", "t", "laplace", "lognorm", "skewnorm"])
def test_summary_matches_scipy_likelihood_and_information_criteria(family):
    x = np.exp(np.random.default_rng(82).normal(size=60))
    result = fit_distribution(x, family)
    ll = result["distribution"].logpdf(x).sum()
    k = len(result["params"])
    assert result["log_likelihood"] == pytest.approx(ll)
    assert result["aic"] == pytest.approx(-2 * ll + 2 * k)
    assert result["bic"] == pytest.approx(-2 * ll + np.log(len(x)) * k)
    direct = getattr(stats, family).fit(
        x, **({"floc": 0} if family == "lognorm" else {})
    )
    np.testing.assert_allclose(
        result["distribution"].ppf([0.05, 0.95]),
        getattr(stats, family)(*direct).ppf([0.05, 0.95]),
    )


@pytest.mark.parametrize("sample", [[], [[1, 2]], [1, np.nan], [1, np.inf], [1j]])
def test_invalid_fit_sample(sample):
    with pytest.raises(ValueError):
        fit_distribution(sample, stats.norm)


def test_unsupported_distribution():
    for family in ["unknown", stats.expon]:
        with pytest.raises(ValueError):
            fit_distribution([1, 2, 3], family)
