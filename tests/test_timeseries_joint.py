import numpy as np
import pytest
from scipy import stats
from scipy.signal import lfilter
from finstats.timeseries.joint import filter_arma_garch, fit_arma_garch, JointARMAGARCHFit


def test_filter_matches_hand_recursion_and_is_causal():
    x=np.array([1.,2.,-.5,3.,1.])
    m,e,h=filter_arma_garch(x,.2,[.4],[.3],.1,.2,.5,2.)
    expected_e=[];expected_h=[];previous_x=.2;previous_e=0.;previous_h=2.
    for i,value in enumerate(x):
        mean=.2+.4*(previous_x-.2)+.3*previous_e
        variance=2. if i==0 else .1+.2*previous_e**2+.5*previous_h
        expected_e.append(value-mean);expected_h.append(variance)
        previous_x,previous_e,previous_h=value,value-mean,variance
    np.testing.assert_allclose(e,expected_e);np.testing.assert_allclose(h,expected_h)
    changed=x.copy();changed[-1]=100.
    m2,e2,h2=filter_arma_garch(changed,.2,[.4],[.3],.1,.2,.5,2.)
    np.testing.assert_allclose(m,m2);np.testing.assert_allclose(h,h2)
    np.testing.assert_allclose(e[:-1],e2[:-1])


def test_predictive_distribution_has_correct_one_step_mean_and_variance():
    x=np.array([.1,-.2,.4,.8,-.3])
    fit=JointARMAGARCHFit(.1,np.array([.4]),np.array([.2]),.1,.2,.5,5.,x,1.,1,-10.,[])
    m,e,h=fit.filter()
    means,variances,paths=fit.forecast(3,simulations=100000,seed=42)
    assert means[0]==pytest.approx(.1+.4*(x[-1]-.1)+.2*e[-1])
    assert variances[0]==pytest.approx(.1+.2*e[-1]**2+.5*h[-1])
    assert variances[1]==pytest.approx(.1+.7*variances[0])
    assert paths[:,0].mean()==pytest.approx(means[0],abs=.01)
    assert paths[:,0].var()==pytest.approx(variances[0],rel=.04)
    np.testing.assert_allclose(paths.mean(axis=0),means,atol=.015)
    np.testing.assert_array_equal(fit.forecast(3,10,42)[2],fit.forecast(3,10,42)[2])
    assert fit.aic==pytest.approx(20+2*len(fit.params))
    assert fit.bic==pytest.approx(20+np.log(4)*len(fit.params))


@pytest.mark.parametrize("distribution", ["normal", "t"])
def test_conditional_mle_likelihood_matches_independent_density(distribution):
    rng=np.random.default_rng(21)
    x=lfilter([1,.2],[1,-.4],rng.normal(size=500))+.1
    fit=fit_arma_garch(x,[.4],[.2],.1,n_starts=1,distribution=distribution)
    _,e,h=fit.filter()
    expected=(stats.norm.logpdf(e[2:],scale=np.sqrt(h[2:])).sum() if distribution=="normal" else
              stats.t.logpdf(e[2:],fit.df,scale=np.sqrt(h[2:]*(fit.df-2)/fit.df)).sum())
    assert fit.loglikelihood==pytest.approx(expected)
    assert fit.starts[0]['converged']
    assert fit.nobs==498


@pytest.mark.parametrize('parameters',[(.1,.2,.8,1.),(-.1,.2,.5,1.),(.1,.2,.5,0.)])
def test_invalid_variance_parameters(parameters):
    with pytest.raises(ValueError):filter_arma_garch([1.,2.,3.],0.,[.4],[.2],*parameters)


@pytest.mark.parametrize('alpha,beta', [([.1],[.4,.2]),([.1,.15],[.4]),([.1,.15],[.3,.2])])
def test_higher_order_variance_recursion_and_forecast(alpha,beta):
    x=np.array([1.,-.5,2.,-.7,.3,1.2])
    initial=2.
    _,e,h=filter_arma_garch(x,0.,[.2],[.1],.1,alpha,beta,initial)
    for t in range(1,len(x)):
        expected=.1+sum(a*(e[t-j]**2 if t>=j else initial) for j,a in enumerate(alpha,1))
        expected+=sum(b*(h[t-j] if t>=j else initial) for j,b in enumerate(beta,1))
        assert h[t]==pytest.approx(expected)
    fit=JointARMAGARCHFit(0.,np.array([.2]),np.array([.1]),.1,np.array(alpha),
                         np.array(beta),None,x,initial,2,-10.,[])
    means,variances,paths=fit.forecast(4,simulations=100000,seed=83)
    assert fit.volatility_order==(len(beta),len(alpha))
    assert len(fit.params)==1+1+1+1+len(alpha)+len(beta)
    assert variances[0]==pytest.approx(.1+sum(a*e[-j]**2 for j,a in enumerate(alpha,1))+
                                     sum(b*h[-j] for j,b in enumerate(beta,1)))
    squares=list(e**2);history=list(h)
    for value in variances:
        expected=.1+sum(a*squares[-j] for j,a in enumerate(alpha,1))
        expected+=sum(b*history[-j] for j,b in enumerate(beta,1))
        assert value==pytest.approx(expected)
        squares.append(value);history.append(value)
    np.testing.assert_allclose(paths.mean(axis=0),means,atol=.012)
    impulse_weights=lfilter([1,.1],[1,-.2],[1,0,0,0])
    np.testing.assert_allclose(paths.var(axis=0),np.convolve(variances,impulse_weights**2)[:4],rtol=.025)


def test_higher_order_joint_mle_and_order_validation():
    rng=np.random.default_rng(26)
    x=lfilter([1,.2],[1,-.4],rng.normal(size=400))
    fit=fit_arma_garch(x,[.4],[.2],0.,garch_order=(2,2),n_starts=1)
    assert fit.volatility_order==(2,2)
    _,e,h=fit.filter()
    assert fit.loglikelihood==pytest.approx(stats.norm.logpdf(e[2:],scale=np.sqrt(h[2:])).sum())
    for order in [(0,1),(1,0),(1,3)]:
        with pytest.raises(ValueError):fit_arma_garch(x,[.4],[.2],0.,garch_order=order)


def test_sandwich_standard_errors_match_analytical_gaussian_scores():
    from finstats.timeseries.joint import _wald_inference
    x=np.random.default_rng(9).normal(.4,1.3,600)
    mu=x.mean();sigma=np.sqrt(np.mean((x-mu)**2));centered=x-mu
    errors,pvalues,status=_wald_inference(lambda theta:stats.norm.logpdf(x,theta[0],theta[1]),[mu,sigma])
    scores=np.column_stack((centered/sigma**2,-1/sigma+centered**2/sigma**3))
    inverse=np.diag([sigma**2/len(x),sigma**2/(2*len(x))])
    expected=np.sqrt(np.diag(inverse@(scores.T@scores)@inverse))
    np.testing.assert_allclose(errors,expected,rtol=1e-5)
    assert pvalues[0]==pytest.approx(2*stats.norm.sf(abs(mu/expected[0])),rel=1e-4,abs=0)
    assert pvalues[1]<1e-200
    assert status=='Computed'


def test_singular_inference_is_unavailable():
    from finstats.timeseries.joint import _wald_inference
    errors,pvalues,status=_wald_inference(lambda theta:np.ones(10),[1.,2.])
    assert np.isnan(errors).all() and np.isnan(pvalues).all()
    assert status.startswith('Unavailable')
