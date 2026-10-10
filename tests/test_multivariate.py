"""Regression checks of notebook calculations using standard libraries."""

import numpy as np
import pytest
from scipy import stats


def test_covariance_hand_calculation_and_affine_properties():
    x, y = (np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 6.0]))
    assert np.cov(x, y, ddof=1)[0, 1] == 2
    assert np.cov(x, y, ddof=0)[0, 1] == pytest.approx(4 / 3)
    assert np.cov(x, y, ddof=2)[0, 1] == 4
    assert np.cov(x + 10, y - 7, ddof=1)[0, 1] == pytest.approx(2)
    assert np.cov(-2 * x + 10, 3 * y - 7, ddof=1)[0, 1] == pytest.approx(-12)
    assert np.cov(x, x, ddof=1)[0, 1] == np.var(x, ddof=1)
    assert np.cov(x, [5, 5, 5], ddof=1)[0, 1] == 0


@pytest.mark.parametrize("ddof", [0, 1, 3])
def test_covariance_and_matrix_numpy_agreement(ddof):
    values = np.random.default_rng(201).normal(size=(40, 3))
    centered = values - values.mean(axis=0)
    expected = centered.T @ centered / (len(values) - ddof)
    observed = np.atleast_2d(np.cov(values, rowvar=False, ddof=ddof))
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(observed, observed.T)
    for j in range(3):
        assert observed[j, j] == pytest.approx(np.var(values[:, j], ddof=ddof))
    assert np.cov(values[:, 0], values[:, 1], ddof=ddof)[0, 1] == pytest.approx(
        expected[0, 1]
    )
    assert np.min(np.linalg.eigvalsh(observed)) >= -1e-14


def test_correlation_linear_relations_and_scipy_agreement():
    x = np.array([-2.0, 0.0, 1.0, 7.0])
    assert np.corrcoef(x, 3 * x + 5)[0, 1] == pytest.approx(1)
    assert np.corrcoef(x, -2 * x + 8)[0, 1] == pytest.approx(-1)
    y = np.array([2.0, -3.0, 6.0, 1.0])
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(stats.pearsonr(x, y).statistic)
    assert np.corrcoef(x + 4, y * 7 - 2)[0, 1] == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_covariance_matrix_hand_calculation_and_single_variable():
    x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 6.0]])
    np.testing.assert_allclose(
        np.atleast_2d(np.cov(x, rowvar=False, ddof=1)), [[1, 2], [2, 7]]
    )
    np.testing.assert_allclose(
        np.atleast_2d(np.cov(x + [5, -2], rowvar=False, ddof=1)), [[1, 2], [2, 7]]
    )
    one_column = np.atleast_2d(np.cov(x[:, :1], rowvar=False, ddof=1))
    assert one_column.shape == (1, 1)
    assert one_column[0, 0] == 1
    np.testing.assert_array_equal(
        (np.array([[1, 2]]) - np.array([[1, 2]]).mean(axis=0)).T
        @ (np.array([[1, 2]]) - np.array([[1, 2]]).mean(axis=0)),
        np.zeros((2, 2)),
    )


def test_linear_combination_hand_calculation():
    weights, means, sigma = ([2, -1], [1, 3], [[4, 1], [1, 9]])
    assert 0 + np.asarray(weights) @ np.asarray(means) == -1
    assert 5 + np.asarray(weights) @ np.asarray(means) == 4
    assert np.asarray(weights) @ np.asarray(sigma) @ np.asarray(weights) == 21
    assert np.asarray([0, 0]) @ np.asarray(sigma) @ np.asarray([0, 0]) == 0
    assert 1 + np.asarray([2]) @ np.asarray([3]) == 7
    assert np.asarray([2]) @ np.asarray([[3]]) @ np.asarray([2]) == 12


def test_linear_combination_of_realized_sample():
    values = np.random.default_rng(202).normal(size=(50, 3))
    weights = np.array([0.3, -0.8, 2.0])
    realized = values @ weights
    assert np.asarray(weights) @ np.asarray(
        np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    ) @ np.asarray(weights) == pytest.approx(np.var(realized, ddof=1))
