"""Mixture, marginal and transformation identities for the bivariate Laplace PDF."""
import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

from finstats.multivariate import bivariate_symmetric_laplace_pdf


def test_density_matches_normal_exponential_mixture():
    center = np.array([.2, -.4])
    sigma = np.array([[1., .6], [.6, 2.]])
    for point in ([.5, .2], [2., -1.]):
        expected = quad(lambda w: stats.multivariate_normal.pdf(
            point, mean=center, cov=w*sigma) * np.exp(-w), 0, np.inf,
            epsabs=1e-10)[0]
        assert bivariate_symmetric_laplace_pdf(point, center, sigma) == pytest.approx(expected, rel=1e-8)


def test_radial_mass_and_population_second_moment():
    density = lambda r: float(bivariate_symmetric_laplace_pdf([r, 0], [0, 0], np.eye(2)))
    mass = quad(lambda r: 2*np.pi*r*density(r), 0, np.inf)[0]
    radius_squared = quad(lambda r: 2*np.pi*r**3*density(r), 0, np.inf)[0]
    assert mass == pytest.approx(1, abs=1e-8)
    assert radius_squared == pytest.approx(2, abs=1e-8)


def test_univariate_laplace_marginal():
    center, sigma = [.2, -.4], [[2., .6], [.6, 1.]]
    x1 = .7
    marginal = quad(lambda x2: float(bivariate_symmetric_laplace_pdf(
        [x1, x2], center, sigma)), -np.inf, np.inf)[0]
    assert marginal == pytest.approx(stats.laplace.pdf(x1, loc=.2, scale=1.), rel=1e-8)


def test_affine_change_of_variables():
    center = np.array([.2, -.4])
    sigma = np.array([[1., .6], [.6, 2.]])
    A, b = np.array([[1.5, .8], [0., .7]]), np.array([2., -1.])
    points = np.array([[-1., 1.], [0., .5], [2., -1.]])
    original = bivariate_symmetric_laplace_pdf(points, center, sigma)
    transformed = bivariate_symmetric_laplace_pdf(points@A.T+b, A@center+b, A@sigma@A.T)
    np.testing.assert_allclose(transformed, original/abs(np.linalg.det(A)))


def test_vectorization_and_integrable_central_singularity():
    points = np.array([[[0., 0.], [.5, 1.]], [[-1., 2.], [3., -.5]]])
    result = bivariate_symmetric_laplace_pdf(points, [0, 0], np.eye(2))
    assert result.shape == (2, 2)
    assert np.isposinf(result[0, 0])
    for index in [(0, 1), (1, 0), (1, 1)]:
        assert result[index] == bivariate_symmetric_laplace_pdf(points[index], [0, 0], np.eye(2))
    assert bivariate_symmetric_laplace_pdf([1, 2], [0, 0], np.eye(2)).shape == ()


@pytest.mark.parametrize('points', [[], [1], [1, 2, 3], 1, [[1, np.nan]], [1, np.inf], [1j, 0]])
def test_invalid_points(points):
    with pytest.raises(ValueError):
        bivariate_symmetric_laplace_pdf(points, [0, 0], np.eye(2))


@pytest.mark.parametrize('center', [[0], [0, 0, 0], [np.nan, 0], [1j, 0]])
def test_invalid_location(center):
    with pytest.raises(ValueError):
        bivariate_symmetric_laplace_pdf([1, 2], center, np.eye(2))


@pytest.mark.parametrize('sigma', [
    [[1]], [[1, 0], [0, 0]], [[1, 2], [2, 1]], [[-1, 0], [0, 1]],
    [[1, .2], [0, 1]], [[np.nan, 0], [0, 1]], [[np.inf, 0], [0, 1]],
    [[1j, 0], [0, 1]],
])
def test_invalid_covariance(sigma):
    with pytest.raises(ValueError):
        bivariate_symmetric_laplace_pdf([1, 2], [0, 0], sigma)
