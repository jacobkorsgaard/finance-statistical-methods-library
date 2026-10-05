"""Analytical checks and domain contracts for Notebook 03 computations."""
import numpy as np
import pytest
from scipy import stats
from finstats.inference import (
    bias, estimator_variance, mean_squared_error, standard_error_mean,
    mean_confidence_interval, variance_confidence_interval,
    bernoulli_loglikelihood, gaussian_loglikelihood, gaussian_mle, wald_statistic,
)


def test_replication_diagnostics_and_mse_identity():
    x = [1, 2, 3]
    assert bias(x, 1) == 1
    assert estimator_variance(x) == 1
    assert estimator_variance(x, ddof=0) == pytest.approx(2/3)
    assert mean_squared_error(x, 1) == pytest.approx(5/3)
    assert mean_squared_error(x, 1) == pytest.approx(estimator_variance(x, 0)+bias(x, 1)**2)
    assert bias([2], 2) == 0
    assert mean_squared_error([2], 2) == 0


@pytest.mark.parametrize('sigma,n,expected', [(2,100,.2),(0,3,0),(1,1,1)])
def test_standard_error_mean(sigma,n,expected):
    assert standard_error_mean(sigma,n) == expected


@pytest.mark.parametrize('sigma,n', [(-1,2),(np.inf,2),(1,0),(1,1.5),(1,True),(1,-2),(1,2+0j)])
def test_invalid_standard_error(sigma,n):
    with pytest.raises(ValueError):
        standard_error_mean(sigma,n)


def test_gaussian_intervals_hand_example():
    x = [1,2,3]  # mean=2, S^2=1, n=3
    q = stats.t.ppf(.975,2)
    assert mean_confidence_interval(x) == pytest.approx((2-q/np.sqrt(3),2+q/np.sqrt(3)))
    assert variance_confidence_interval(x) == pytest.approx((2/stats.chi2.ppf(.975,2),2/stats.chi2.ppf(.025,2)))
    assert mean_confidence_interval(x, .1) == pytest.approx(stats.t.interval(.9,2,loc=2,scale=1/np.sqrt(3)))


def test_interval_affine_behavior_and_confidence_width():
    x=np.array([-2.,0.,1.,4.])
    assert mean_confidence_interval(3+2*x) == pytest.approx(3+2*np.array(mean_confidence_interval(x)))
    assert variance_confidence_interval(3+2*x) == pytest.approx(4*np.array(variance_confidence_interval(x)))
    for f in (mean_confidence_interval,variance_confidence_interval):
        narrow=f(x,.1); wide=f(x,.01)
        assert wide[0]<narrow[0]<narrow[1]<wide[1]


@pytest.mark.parametrize('f',[mean_confidence_interval,variance_confidence_interval])
@pytest.mark.parametrize('alpha',[0,1,-.1,np.nan,np.inf,[.05],.05+0j])
def test_interval_invalid_alpha(f,alpha):
    with pytest.raises(ValueError):
        f([1,2,3],alpha)


@pytest.mark.parametrize('f',[mean_confidence_interval,variance_confidence_interval])
@pytest.mark.parametrize('x',[[1],[1,1],[],[1,np.nan],[[1,2]],[1,2j]])
def test_interval_invalid_sample(f,x):
    with pytest.raises(ValueError):
        f(x)


def test_bernoulli_likelihood_and_boundaries():
    y=[1,0,1]
    assert bernoulli_loglikelihood(.6,y)==pytest.approx(2*np.log(.6)+np.log(.4))
    assert bernoulli_loglikelihood(.6,y)==pytest.approx(stats.bernoulli.logpmf(y,.6).sum())
    assert bernoulli_loglikelihood(0,[0,0])==0
    assert bernoulli_loglikelihood(1,[1,1])==0
    assert bernoulli_loglikelihood(0,[1])==-np.inf
    assert bernoulli_loglikelihood(1,[0])==-np.inf
    assert bernoulli_loglikelihood(2/3,y)>bernoulli_loglikelihood(.5,y)


@pytest.mark.parametrize('theta,y',[(-.1,[0]),(1.1,[1]),(np.nan,[1]),(.5,[.2]),(.5,[]),(.5,[np.inf]),(.5,[[0,1]]),(1j,[1])])
def test_bernoulli_invalid(theta,y):
    with pytest.raises(ValueError):
        bernoulli_loglikelihood(theta,y)


def test_gaussian_mle_and_loglikelihood():
    y=np.array([1.,2.,3.])
    assert gaussian_mle(y)==pytest.approx((2,2/3))
    assert gaussian_loglikelihood([2,1],y)==pytest.approx(-1.5*np.log(2*np.pi)-1)
    assert gaussian_loglikelihood([2,1],y)==pytest.approx(stats.norm.logpdf(y,loc=2,scale=1).sum())
    mu,sd=stats.norm.fit(y)
    assert gaussian_mle(y)==pytest.approx((mu,sd**2))
    assert gaussian_loglikelihood(gaussian_mle(y),y)>gaussian_loglikelihood([2,1],y)


@pytest.mark.parametrize('params',[[0,0],[0,-1],[0,np.inf],[np.nan,1],[0],[0,1,2],[[0,1]],[1j,1]])
def test_gaussian_invalid_params(params):
    with pytest.raises(ValueError):
        gaussian_loglikelihood(params,[1,2])


@pytest.mark.parametrize('y',[[],[1],[1,1],[np.nan,1],[[1,2]],[1j,2]])
def test_gaussian_mle_invalid(y):
    with pytest.raises(ValueError):
        gaussian_mle(y)


def test_wald_hand_example():
    assert wald_statistic(.005,0,.02**2/100)==pytest.approx(6.25)
    assert wald_statistic(2,2,1)==0
    assert wald_statistic(1,2,.5)==wald_statistic(2,1,.5)


@pytest.mark.parametrize('args',[(1,0,0),(1,0,-1),(1,0,np.inf),(np.nan,0,1),(1,np.nan,1),([1],0,1),(1,0,1j)])
def test_wald_invalid(args):
    with pytest.raises(ValueError):
        wald_statistic(*args)


@pytest.mark.parametrize('f',[bias,mean_squared_error])
@pytest.mark.parametrize('truth',[np.nan,np.inf,[1],1j])
def test_invalid_truth(f,truth):
    with pytest.raises(ValueError):
        f([1,2],truth)


@pytest.mark.parametrize('f',[bias,estimator_variance,mean_squared_error])
@pytest.mark.parametrize('x',[[],[np.nan],[[1,2]],[1j]])
def test_invalid_replications(f,x):
    with pytest.raises(ValueError):
        if f is estimator_variance:
            f(x)
        else:
            f(x, 0)


def test_replication_variance_domain():
    for ddof in (-1,True,1.2,3):
        with pytest.raises(ValueError):
            estimator_variance([1,2,3],ddof)


def test_gaussian_interval_coverage_fixed_seed():
    # A statistical check of the complete interval computation, not its formula alone.
    draws=np.random.default_rng(303).normal(loc=2,scale=3,size=(2000,20))
    mean_intervals=np.array([mean_confidence_interval(x) for x in draws])
    variance_intervals=np.array([variance_confidence_interval(x) for x in draws])
    for intervals,truth in [(mean_intervals,2),(variance_intervals,9)]:
        coverage=np.mean((intervals[:,0]<=truth)&(truth<=intervals[:,1]))
        assert abs(coverage-.95)<.02


@pytest.mark.parametrize('y', [[], [np.nan], [np.inf], [[1, 2]], [1j]])
def test_gaussian_loglikelihood_invalid_observations(y):
    with pytest.raises(ValueError):
        gaussian_loglikelihood([0, 1], y)
