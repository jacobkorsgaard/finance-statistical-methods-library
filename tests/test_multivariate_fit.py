import numpy as np
import pytest
from scipy import stats
from finstats.multivariate_fit import multivariate_t_mle


def test_student_t_fit_recovers_population_and_improves_likelihood():
    mean = np.array([.5, -.3])
    scale = np.array([[1., .3], [.3, .6]])
    x = stats.multivariate_t.rvs(loc=mean, shape=scale, df=5, size=2500,
                                random_state=np.random.default_rng(606))
    fit = multivariate_t_mle(x)
    np.testing.assert_allclose(fit['location'], mean, atol=.08)
    np.testing.assert_allclose(fit['scale'], scale, atol=.12)
    assert 3.5 < fit['df'] < 7.5
    assert fit['log_likelihood'] >= stats.multivariate_t.logpdf(x, loc=mean, shape=scale, df=5).sum()
    assert fit['log_likelihood'] == pytest.approx(stats.multivariate_t.logpdf(
        x, loc=fit['location'], shape=fit['scale'], df=fit['df']).sum())
    assert np.linalg.eigvalsh(fit['scale']).min() > 0
    assert any(row['success'] for row in fit['starts'])
    transformed = multivariate_t_mle(x*np.array([3., .2])+[4., -1.])
    np.testing.assert_allclose(transformed['location'], fit['location']*[3., .2]+[4., -1.], atol=1e-4)
    np.testing.assert_allclose(transformed['scale'], fit['scale']*np.outer([3., .2], [3., .2]), rtol=1e-4)


@pytest.mark.parametrize('x', [[], [1,2,3], [[1,2],[3,4]], [[1,2],[1,3],[1,4]],
                              [[1,2],[2,4],[3,6]], [[1,2],[3,np.nan],[4,5]],
                              [[1j,2],[3,4],[5,6]]])
def test_invalid_samples(x):
    with pytest.raises(ValueError):
        multivariate_t_mle(x)


@pytest.mark.parametrize('starts', [[], [1], [201], [np.nan], [[4]]])
def test_invalid_starts(starts):
    with pytest.raises(ValueError):
        multivariate_t_mle([[0,1],[1,0],[2,3],[3,1]], starts)


def test_all_failed_starts_raise(monkeypatch):
    from types import SimpleNamespace
    import finstats.multivariate_fit as module
    monkeypatch.setattr(module.optimize, 'minimize', lambda objective, initial, **kwargs:
                        SimpleNamespace(fun=objective(initial), x=initial,
                                        success=False, message='failed'))
    with pytest.raises(RuntimeError, match='all multivariate'):
        multivariate_t_mle([[0,1],[1,0],[2,3],[3,1]])
