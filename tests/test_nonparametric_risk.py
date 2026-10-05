import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from finstats.nonparametric import empirical_cdf, empirical_quantile, qq_data
from finstats.risk import (
    historical_var, historical_es, gaussian_var, gaussian_es, student_t_var, student_t_es,
)


def test_ecdf_right_continuity_and_ties():
    x = np.array([3.,1.,1.,2.])
    f = empirical_cdf(x)
    x[:] = 0  # evaluator owns its sorted observations
    np.testing.assert_array_equal(f([-np.inf,.9,1,1.9,2,3,np.inf]),[0,0,.5,.5,.75,1,1])
    assert float(f(1)) == .5
    assert f(np.zeros((2,2))).shape == (2,2)


@pytest.mark.parametrize('q,expected', [(0,1),(.25,1),(.5,1),(.51,2),(1,3)])
def test_inverse_quantile(q,expected):
    assert empirical_quantile([3,1,1,2],q) == expected


def test_qq_quantiles():
    theoretical, observed = qq_data([3,1,2])
    np.testing.assert_array_equal(observed,[1,2,3])
    np.testing.assert_allclose(theoretical,stats.norm.ppf([1/6,.5,5/6]))
    theoretical, _ = qq_data([1,2],stats.t(5))
    np.testing.assert_allclose(theoretical,stats.t.ppf([.25,.75],5))


def test_historical_loss_and_strict_tail():
    losses=[0,1,2,3,4]
    assert historical_var(losses,.4) == 2
    assert historical_es(losses,.4) == 3.5
    assert historical_var([-5,-3,-1],.5) == -3  # VaR need not be positive
    with pytest.raises(ValueError):
        historical_es([1,1,1],.05)
    with pytest.raises(ValueError):
        historical_es(losses,.01)  # quantile is the maximum


def test_parametric_loss_formulas_and_integrals():
    for alpha in [.5,.05,.005]:
        q=stats.norm.isf(alpha)
        assert gaussian_var(2,3,alpha) == pytest.approx(2+3*q)
        expected=quad(lambda z:z*stats.norm.pdf(z),q,np.inf)[0]/alpha
        assert gaussian_es(2,3,alpha) == pytest.approx(2+3*expected)
        for df in [2,5,20]:
            q=stats.t.isf(alpha,df)
            assert student_t_var(5,2,df,alpha) == pytest.approx(stats.t.isf(alpha,df,loc=5,scale=2))
            expected=quad(lambda z:z*stats.t.pdf(z,df),q,np.inf)[0]/alpha
            assert student_t_es(5,2,df,alpha) == pytest.approx(5+2*expected)
            assert student_t_es(5,2,df,alpha)>student_t_var(5,2,df,alpha)


@pytest.mark.parametrize('x',[[],[[1,2]],[np.nan],[np.inf],[1j]])
def test_invalid_samples(x):
    for f in [empirical_cdf,qq_data,historical_var,historical_es]:
        with pytest.raises(ValueError):
            f(x)
    with pytest.raises(ValueError):
        empirical_quantile(x,.5)


@pytest.mark.parametrize('q',[-1,1.1,np.nan,np.inf,[.5],1j])
def test_invalid_quantile(q):
    with pytest.raises(ValueError):
        empirical_quantile([1,2],q)


@pytest.mark.parametrize('alpha',[0,1,-.1,np.nan,np.inf,[.05],1j])
def test_invalid_alpha(alpha):
    for f,args in [(historical_var,([1,2],)),(historical_es,([1,2],)),
                   (gaussian_var,(0,1)),(gaussian_es,(0,1)),
                   (student_t_var,(0,1,5)),(student_t_es,(0,1,5))]:
        with pytest.raises(ValueError):
            f(*args,alpha=alpha)


@pytest.mark.parametrize('scale',[0,-1,np.nan,np.inf])
def test_invalid_scales(scale):
    for f,args in [(gaussian_var,(0,scale)),(gaussian_es,(0,scale)),
                   (student_t_var,(0,scale,5)),(student_t_es,(0,scale,5))]:
        with pytest.raises(ValueError):
            f(*args)


def test_t_moment_restrictions_and_ecdf_points():
    assert np.isfinite(student_t_var(0,1,.5))
    for df in [0,-1,np.nan,np.inf]:
        with pytest.raises(ValueError):
            student_t_var(0,1,df)
    for df in [.5,1,np.nan,np.inf]:
        with pytest.raises(ValueError):
            student_t_es(0,1,df)
    for points in [np.nan,1j]:
        with pytest.raises(ValueError):
            empirical_cdf([1,2])(points)


def test_ecdf_monte_carlo_moments():
    samples=np.random.default_rng(404).standard_t(1,size=(4000,100))
    estimates=np.array([float(empirical_cdf(x)(0)) for x in samples])
    assert abs(estimates.mean()-.5)<.004
    assert abs(estimates.var()-.25/100)<.0002
