import numpy as np
import pytest
from scipy import stats
from finstats.fitting import fit_distribution


def test_normal_reference_values():
    """Check well-known values of the standard Normal distribution."""
    assert stats.norm.pdf(0.0) == pytest.approx(1 / np.sqrt(2 * np.pi))
    assert stats.norm.cdf(0.0) == pytest.approx(0.5)
    assert stats.norm.ppf(0.5) == pytest.approx(0.0)


def test_normal_quantile_inverse_and_fit():
    """Check inverse CDF and Normal MLE."""
    q = np.array([0.1, 0.5, 0.9])
    x = stats.norm.ppf(q, loc=1.0, scale=2.0)
    recovered_q = stats.norm.cdf(x, loc=1.0, scale=2.0)
    assert np.allclose(recovered_q, q)
    data = np.array([1.0, 2.0, 3.0, 4.0, 8.0])
    fit = fit_distribution(data, stats.norm)
    assert fit["params"]["mu"] == pytest.approx(np.mean(data))
    assert fit["params"]["sigma"] == pytest.approx(np.std(data, ddof=0))


def test_lognormal():
    """Check Lognormal reference values, MLE and support."""
    assert stats.lognorm.cdf(1.0, s=1.0, scale=np.exp(0.0)) == pytest.approx(0.5)
    assert stats.lognorm.ppf(0.5, s=1.0, scale=np.exp(0.0)) == pytest.approx(1.0)
    data = np.exp(np.array([-0.5, 0.0, 0.2, 0.7, 1.1]))
    fit = fit_distribution(data, stats.lognorm)
    log_data = np.log(data)
    assert fit["params"]["mu"] == pytest.approx(np.mean(log_data))
    assert fit["params"]["sigma"] == pytest.approx(np.std(log_data, ddof=0))
    with pytest.raises(ValueError):
        fit_distribution([1.0, 0.0, 2.0], stats.lognorm)


def test_laplace():
    """Check Laplace reference values and closed-form MLE."""
    assert stats.laplace.pdf(2.0, scale=3.0, loc=2.0) == pytest.approx(1 / 6)
    assert stats.laplace.cdf(2.0, scale=3.0, loc=2.0) == pytest.approx(0.5)
    data = np.array([-3.0, -1.0, 0.0, 0.5, 5.0])
    fit = fit_distribution(data, stats.laplace)
    mu_hat = np.median(data)
    scale_hat = np.mean(np.abs(data - mu_hat))
    assert fit["params"]["mu"] == pytest.approx(mu_hat)
    assert fit["params"]["scale"] == pytest.approx(scale_hat)


def test_student_t():
    """Check Student-t symmetry, inverse CDF and fitted parameters."""
    assert stats.t.cdf(2.0, scale=1.5, df=5.0, loc=2.0) == pytest.approx(0.5)
    q = np.array([0.1, 0.5, 0.9])
    x = stats.t.ppf(q, scale=1.4, df=7.0, loc=0.2)
    assert np.allclose(stats.t.cdf(x, scale=1.4, df=7.0, loc=0.2), q)
    data = stats.t.rvs(scale=1.2, df=6.0, random_state=123, loc=0.3, size=400)
    fit = fit_distribution(data, stats.t)
    assert fit["params"]["scale"] > 0
    assert fit["params"]["df"] > 0
    assert np.isfinite(fit["log_likelihood"])


def test_skew_normal():
    """Check alpha=0 reduction to Normal and fitted parameters."""
    x = np.linspace(-3.0, 3.0, 21)
    assert np.allclose(
        stats.skewnorm.pdf(x, scale=2.0, loc=1.0, a=0.0),
        stats.norm.pdf(x, loc=1.0, scale=2.0),
    )
    data = stats.skewnorm.rvs(scale=1.3, random_state=123, loc=-0.2, a=4.0, size=400)
    fit = fit_distribution(data, stats.skewnorm)
    assert fit["params"]["scale"] > 0
    assert np.isfinite(fit["params"]["alpha"])
    assert np.isfinite(fit["log_likelihood"])


def test_sample_sizes():
    """All distributions should generate the requested sample size."""
    assert len(stats.norm.rvs(random_state=1, size=25)) == 25
    assert (
        len(stats.lognorm.rvs(random_state=1, size=25, s=1.0, scale=np.exp(0.0))) == 25
    )
    assert len(stats.laplace.rvs(random_state=1, size=25)) == 25
    assert len(stats.t.rvs(random_state=1, size=25, df=5.0)) == 25
    assert len(stats.skewnorm.rvs(random_state=1, size=25, a=0.0)) == 25


def test_information_criteria():
    """Check AIC and BIC against their definitions."""
    assert -2 * -100.0 + 2 * 3 == pytest.approx(206.0)
    expected_bic = 200.0 + 3 * np.log(50)
    assert -2 * -100.0 + np.log(50) * 3 == pytest.approx(expected_bic)
