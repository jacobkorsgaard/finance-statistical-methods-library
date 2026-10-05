import numpy as np
import pytest
from scipy import stats

from finstats.descriptive import sample_variance
from finstats.multivariate import (
    correlation, covariance, covariance_matrix,
    linear_combination_mean, linear_combination_variance,
)


def test_covariance_hand_calculation_and_affine_properties():
    x, y = np.array([1., 2., 3.]), np.array([2., 1., 6.])
    assert covariance(x, y) == 2
    assert covariance(x, y, ddof=0) == pytest.approx(4 / 3)
    assert covariance(x, y, ddof=2) == 4
    assert covariance(x + 10, y - 7) == pytest.approx(2)
    assert covariance(-2 * x + 10, 3 * y - 7) == pytest.approx(-12)
    assert covariance(x, x) == sample_variance(x)
    assert covariance(x, [5, 5, 5]) == 0


@pytest.mark.parametrize('ddof', [0, 1, 3])
def test_covariance_and_matrix_numpy_agreement(ddof):
    values = np.random.default_rng(201).normal(size=(40, 3))
    expected = np.cov(values, rowvar=False, ddof=ddof)
    observed = covariance_matrix(values, ddof)
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(observed, observed.T)
    for j in range(3):
        assert observed[j, j] == pytest.approx(sample_variance(values[:, j], ddof))
    assert covariance(values[:, 0], values[:, 1], ddof) == pytest.approx(expected[0, 1])
    assert np.min(np.linalg.eigvalsh(observed)) >= -1e-14


def test_correlation_linear_relations_and_scipy_agreement():
    x = np.array([-2., 0., 1., 7.])
    assert correlation(x, 3 * x + 5) == pytest.approx(1)
    assert correlation(x, -2 * x + 8) == pytest.approx(-1)
    y = np.array([2., -3., 6., 1.])
    assert correlation(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert correlation(x, y) == pytest.approx(stats.pearsonr(x, y).statistic)
    assert correlation(x + 4, y * 7 - 2) == pytest.approx(correlation(x, y))


def test_covariance_matrix_hand_calculation_and_single_variable():
    x = np.array([[1., 2.], [2., 1.], [3., 6.]])
    np.testing.assert_allclose(covariance_matrix(x), [[1, 2], [2, 7]])
    np.testing.assert_allclose(covariance_matrix(x + [5, -2]), [[1, 2], [2, 7]])
    one_column = covariance_matrix(x[:, :1])
    assert one_column.shape == (1, 1)
    assert one_column[0, 0] == 1
    np.testing.assert_array_equal(covariance_matrix([[1, 2]], ddof=0), np.zeros((2, 2)))


def test_linear_combination_hand_calculation():
    weights, means, sigma = [2, -1], [1, 3], [[4, 1], [1, 9]]
    assert linear_combination_mean(weights, means) == -1
    assert linear_combination_mean(weights, means, intercept=5) == 4
    assert linear_combination_variance(weights, sigma) == 21
    assert linear_combination_variance([0, 0], sigma) == 0
    assert linear_combination_mean([2], [3], 1) == 7
    assert linear_combination_variance([2], [[3]]) == 12


def test_linear_combination_of_realized_sample():
    values = np.random.default_rng(202).normal(size=(50, 3))
    weights = np.array([0.3, -0.8, 2.])
    realized = values @ weights
    assert linear_combination_variance(weights, covariance_matrix(values)) == pytest.approx(
        sample_variance(realized)
    )


def test_singular_covariance_and_roundoff_tolerance():
    assert linear_combination_variance([1, -1], [[1, 1], [1, 1]]) == 0
    assert linear_combination_variance([1, 2], np.zeros((2, 2))) == 0
    assert linear_combination_variance([1, -1], [[1, 1 + 1e-14], [1, 1]]) == 0
    # Tolerance follows covariance units, including small financial variances.
    with pytest.raises(ValueError):
        linear_combination_variance([1, 1], [[1e-16, 2e-16], [2e-16, 1e-16]])


@pytest.mark.parametrize('bad', [[], [[1, 2]], [np.nan], [np.inf], [1j]])
@pytest.mark.parametrize('function', [covariance, correlation])
def test_invalid_paired_samples(function, bad):
    with pytest.raises(ValueError):
        function(bad, [1, 2])
    with pytest.raises(ValueError):
        function([1, 2], bad)


@pytest.mark.parametrize('function', [covariance, correlation, linear_combination_mean])
def test_mismatched_vector_lengths(function):
    with pytest.raises(ValueError):
        function([1, 2], [1, 2, 3])


@pytest.mark.parametrize('ddof', [-1, 1.5, True, 3, np.nan])
def test_invalid_ddof(ddof):
    with pytest.raises(ValueError):
        covariance([1, 2, 3], [3, 2, 1], ddof)
    with pytest.raises(ValueError):
        covariance_matrix([[1], [2], [3]], ddof)


@pytest.mark.parametrize('x,y', [([1], [2]), ([1, 1], [2, 3]), ([1, 2], [3, 3])])
def test_undefined_correlation(x, y):
    with pytest.raises(ValueError):
        correlation(x, y)


@pytest.mark.parametrize('bad', [[], [1, 2], [[]], [[np.nan]], [[np.inf]], [[1j]]])
def test_invalid_sample_matrix(bad):
    with pytest.raises(ValueError):
        covariance_matrix(bad)


def test_singleton_covariance_requires_ddof_zero():
    with pytest.raises(ValueError):
        covariance([1], [2])
    with pytest.raises(ValueError):
        covariance_matrix([[1, 2]])
    assert covariance([1], [2], ddof=0) == 0


@pytest.mark.parametrize('bad', [[], [[1, 2]], [np.nan], [np.inf], [1j]])
def test_invalid_linear_combination_vectors(bad):
    with pytest.raises(ValueError):
        linear_combination_mean(bad, [1, 2])
    with pytest.raises(ValueError):
        linear_combination_mean([1, 2], bad)
    with pytest.raises(ValueError):
        linear_combination_variance(bad, np.eye(2))


@pytest.mark.parametrize('intercept', [np.nan, np.inf, 1j, [1, 2]])
def test_invalid_intercept(intercept):
    with pytest.raises(ValueError):
        linear_combination_mean([1], [2], intercept)


@pytest.mark.parametrize('sigma', [
    [1, 2], [[1]], [[1, 0, 0], [0, 1, 0]],
    [[1, 0.5], [0, 1]], [[1, 2], [2, 1]], [[-1, 0], [0, 1]],
    [[np.nan, 0], [0, 1]], [[np.inf, 0], [0, 1]], [[1j, 0], [0, 1]],
])
def test_invalid_population_covariance(sigma):
    with pytest.raises(ValueError):
        linear_combination_variance([1, 2], sigma)
