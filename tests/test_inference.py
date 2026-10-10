"""Regression checks of notebook calculations using standard libraries."""

import numpy as np
import pytest
from scipy import stats


def test_replication_diagnostics_and_mse_identity():
    x = [1, 2, 3]
    assert np.mean(x) - 1 == 1
    assert np.var(x, ddof=1) == 1
    assert np.var(x, ddof=0) == pytest.approx(2 / 3)
    assert np.mean((np.asarray(x) - 1) ** 2) == pytest.approx(5 / 3)
    assert np.mean((np.asarray(x) - 1) ** 2) == pytest.approx(
        np.var(x, ddof=0) + (np.mean(x) - 1) ** 2
    )
    assert np.mean([2]) - 2 == 0
    assert np.mean((np.asarray([2]) - 2) ** 2) == 0


@pytest.mark.parametrize("sigma,n,expected", [(2, 100, 0.2), (0, 3, 0), (1, 1, 1)])
def test_standard_error_mean(sigma, n, expected):
    assert sigma / np.sqrt(n) == expected


def test_gaussian_intervals_hand_example():
    x = [1, 2, 3]
    q = stats.t.ppf(0.975, 2)
    assert stats.t.interval(
        1 - 0.05, len(x) - 1, loc=np.mean(x), scale=np.std(x, ddof=1) / np.sqrt(len(x))
    ) == pytest.approx((2 - q / np.sqrt(3), 2 + q / np.sqrt(3)))
    assert (len(x) - 1) * np.var(x, ddof=1) / stats.chi2.ppf(
        [1 - 0.05 / 2, 0.05 / 2], len(x) - 1
    ) == pytest.approx((2 / stats.chi2.ppf(0.975, 2), 2 / stats.chi2.ppf(0.025, 2)))
    assert stats.t.interval(
        1 - 0.1, len(x) - 1, loc=np.mean(x), scale=np.std(x, ddof=1) / np.sqrt(len(x))
    ) == pytest.approx(stats.t.interval(0.9, 2, loc=2, scale=1 / np.sqrt(3)))


def test_interval_affine_behavior_and_confidence_width():
    x = np.array([-2.0, 0.0, 1.0, 4.0])
    assert stats.t.interval(
        1 - 0.05,
        len(3 + 2 * x) - 1,
        loc=np.mean(3 + 2 * x),
        scale=np.std(3 + 2 * x, ddof=1) / np.sqrt(len(3 + 2 * x)),
    ) == pytest.approx(
        3
        + 2
        * np.array(
            stats.t.interval(
                1 - 0.05,
                len(x) - 1,
                loc=np.mean(x),
                scale=np.std(x, ddof=1) / np.sqrt(len(x)),
            )
        )
    )
    assert (len(3 + 2 * x) - 1) * np.var(3 + 2 * x, ddof=1) / stats.chi2.ppf(
        [1 - 0.05 / 2, 0.05 / 2], len(3 + 2 * x) - 1
    ) == pytest.approx(
        4
        * np.array(
            (len(x) - 1)
            * np.var(x, ddof=1)
            / stats.chi2.ppf([1 - 0.05 / 2, 0.05 / 2], len(x) - 1)
        )
    )
    narrow = stats.t.interval(0.9, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    wide = stats.t.interval(0.99, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


def test_bernoulli_likelihood_and_boundaries():
    y = [1, 0, 1]
    assert stats.bernoulli.logpmf(y, 0.6).sum() == pytest.approx(
        2 * np.log(0.6) + np.log(0.4)
    )
    assert stats.bernoulli.logpmf(y, 0.6).sum() == pytest.approx(
        stats.bernoulli.logpmf(y, 0.6).sum()
    )
    assert stats.bernoulli.logpmf([0, 0], 0).sum() == 0
    assert stats.bernoulli.logpmf([1, 1], 1).sum() == 0
    assert stats.bernoulli.logpmf([1], 0).sum() == -np.inf
    assert stats.bernoulli.logpmf([0], 1).sum() == -np.inf
    assert stats.bernoulli.logpmf(y, 2 / 3).sum() > stats.bernoulli.logpmf(y, 0.5).sum()


def test_gaussian_mle_and_loglikelihood():
    y = np.array([1.0, 2.0, 3.0])
    assert (np.mean(y), np.var(y, ddof=0)) == pytest.approx((2, 2 / 3))
    assert stats.norm.logpdf(
        y, loc=[2, 1][0], scale=np.sqrt([2, 1][1])
    ).sum() == pytest.approx(-1.5 * np.log(2 * np.pi) - 1)
    assert stats.norm.logpdf(
        y, loc=[2, 1][0], scale=np.sqrt([2, 1][1])
    ).sum() == pytest.approx(stats.norm.logpdf(y, loc=2, scale=1).sum())
    mu, sd = stats.norm.fit(y)
    assert (np.mean(y), np.var(y, ddof=0)) == pytest.approx((mu, sd**2))
    assert (
        stats.norm.logpdf(
            y,
            loc=(np.mean(y), np.var(y, ddof=0))[0],
            scale=np.sqrt((np.mean(y), np.var(y, ddof=0))[1]),
        ).sum()
        > stats.norm.logpdf(y, loc=[2, 1][0], scale=np.sqrt([2, 1][1])).sum()
    )


def test_wald_hand_example():
    assert (0.005 - 0) ** 2 / (0.02**2 / 100) == pytest.approx(6.25)
    assert (2 - 2) ** 2 / 1 == 0
    assert (1 - 2) ** 2 / 0.5 == (2 - 1) ** 2 / 0.5


def test_gaussian_interval_coverage_fixed_seed():
    draws = np.random.default_rng(303).normal(loc=2, scale=3, size=(2000, 20))
    mean_intervals = np.array(
        [
            stats.t.interval(
                1 - 0.05,
                len(x) - 1,
                loc=np.mean(x),
                scale=np.std(x, ddof=1) / np.sqrt(len(x)),
            )
            for x in draws
        ]
    )
    variance_intervals = np.array(
        [
            (len(x) - 1)
            * np.var(x, ddof=1)
            / stats.chi2.ppf([1 - 0.05 / 2, 0.05 / 2], len(x) - 1)
            for x in draws
        ]
    )
    for intervals, truth in [(mean_intervals, 2), (variance_intervals, 9)]:
        coverage = np.mean((intervals[:, 0] <= truth) & (truth <= intervals[:, 1]))
        assert abs(coverage - 0.95) < 0.02
