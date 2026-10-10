import numpy as np
import pytest
from scipy import integrate, stats
from finstats.multivariate import (
    multivariate_skew_normal_logpdf as logpdf,
    multivariate_skew_normal_moments as moments,
    multivariate_skew_normal_rvs as rvs,
    multivariate_skew_normal_mle as mle,
)


def test_gaussian_limit_and_independent_scalar_skew_normal():
    x = np.array([[0.,1.],[-1.,2.],[2.,-.5]])
    mu, scale = np.array([.2,-.3]), np.diag([4.,.25])
    np.testing.assert_allclose(logpdf(x,mu,scale,[0,0]), stats.multivariate_normal.logpdf(x,mu,scale))
    expected = stats.skewnorm.logpdf(x[:,0],3,loc=.2,scale=2)+stats.norm.logpdf(x[:,1],loc=-.3,scale=.5)
    np.testing.assert_allclose(logpdf(x,mu,scale,[3,0]), expected)
    m,c = moments(mu,scale,[0,0])
    np.testing.assert_allclose(m,mu)
    np.testing.assert_allclose(c,scale)


def test_normalization_and_marginal_shape_with_correlation():
    scale = np.array([[1.,.4],[.4,2.]])
    alpha = np.array([2.,-1.])
    sd = np.sqrt(np.diag(scale))
    omega = scale/np.outer(sd,sd)
    delta = omega@alpha/np.sqrt(1+alpha@omega@alpha)
    for x in [-1.,0.,1.]:
        actual = integrate.quad(lambda y: np.exp(logpdf([x,y],[0,0],scale,alpha)), -np.inf,np.inf)[0]
        expected = stats.skewnorm.pdf(x,delta[0]/np.sqrt(1-delta[0]**2),scale=sd[0])
        assert actual == pytest.approx(expected,rel=1e-7)
    integral = integrate.quad(lambda x: stats.skewnorm.pdf(x,delta[0]/np.sqrt(1-delta[0]**2)), -np.inf,np.inf)[0]
    assert integral == pytest.approx(1.)


def test_sampling_reproduces_moments_and_seed():
    mu,scale,alpha = [.2,-.4], [[1.,.3],[.3,.7]], [3.,-2.]
    x = rvs(mu,scale,alpha,100_000,1006)
    m,c = moments(mu,scale,alpha)
    np.testing.assert_allclose(x.mean(axis=0),m,atol=.012)
    np.testing.assert_allclose(np.cov(x,rowvar=False),c,atol=.015)
    np.testing.assert_array_equal(rvs(mu,scale,alpha,20,7),rvs(mu,scale,alpha,20,7))


def test_mle_improves_gaussian_and_recovers_moments():
    x = rvs([.5,-.2],[[1.,.2],[.2,.7]],[3.,-1.],1600,606)
    fit = mle(x)
    ll = logpdf(x,fit['location'],fit['scale'],fit['shape']).sum()
    assert fit['log_likelihood'] == pytest.approx(ll)
    assert ll >= stats.multivariate_normal.logpdf(x,x.mean(axis=0),np.cov(x,rowvar=False,ddof=0)).sum()-1e-6
    m,c = moments(fit['location'],fit['scale'],fit['shape'])
    truth_m,truth_c = moments([.5,-.2],[[1.,.2],[.2,.7]],[3.,-1.])
    np.testing.assert_allclose(m,truth_m,atol=.08)
    np.testing.assert_allclose(c,truth_c,atol=.09)
    assert any(s['success'] for s in fit['starts'])
    # Positive diagonal changes of units preserve the density including its Jacobian.
    units = np.array([3.,.2])
    np.testing.assert_allclose(logpdf(x*units+2,fit['location']*units+2,
        fit['scale']*np.outer(units,units),fit['shape']),logpdf(x,fit['location'],fit['scale'],fit['shape'])-np.log(units).sum())


@pytest.mark.parametrize('scale,shape',[([[1,2],[2,1]],[0,0]),([[1,.1],[0,1]],[0,0]),
    ([[1,0],[0,1]],[np.nan,0]),([[1,0],[0,1]],[1j,0]),([[1,0],[0,1]],[0])])
def test_parameter_validation(scale,shape):
    with pytest.raises(ValueError): logpdf([0,0],[0,0],scale,shape)


@pytest.mark.parametrize('x',[[],[1,2],[[1,2],[2,4],[3,6]],[[1,np.nan],[2,3],[4,5]],[[1j,2],[2,3],[4,5]]])
def test_mle_invalid_data(x):
    with pytest.raises(ValueError): mle(x)


@pytest.mark.parametrize('x',[[],[0],[[0,np.nan]],[1j,0]])
def test_point_validation(x):
    with pytest.raises(ValueError): logpdf(x,[0,0],np.eye(2),[0,0])


@pytest.mark.parametrize('size',[0,-1,1.5,True])
def test_sampling_size_validation(size):
    with pytest.raises(ValueError): rvs([0,0],np.eye(2),[0,0],size,1)


def test_failed_starts_raise(monkeypatch):
    from types import SimpleNamespace
    import finstats.multivariate as module
    monkeypatch.setattr(module.optimize,'minimize',lambda fun,x,**kwargs:
        SimpleNamespace(fun=fun(x),x=x,success=False,message='failed'))
    with pytest.raises(RuntimeError,match='all skew-normal'):
        mle([[0,1],[1,0],[2,3],[3,1]])
