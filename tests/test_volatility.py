import numpy as np
import pytest
from statsmodels.stats.diagnostic import acorr_ljungbox
from finstats.timeseries.diagnostics import mcleod_li, standardized_residuals
from finstats.timeseries.volatility import (
    simulate_arch1, arch1_unconditional_variance, simulate_garch11,
    garch11_unconditional_variance, garch11_variance_path, garch11_forecast,
)


def test_population_variances():
    assert arch1_unconditional_variance(2,.8)==pytest.approx(10)
    assert garch11_unconditional_variance(1,.2,.5)==pytest.approx(10/3)


def test_variance_alignment_hand_recursion():
    h=garch11_variance_path([2,3,-1],1,.2,.5,initial_variance=4)
    np.testing.assert_allclose(h,[4,3.8,4.7])
    np.testing.assert_allclose(garch11_forecast(-1,4.7,1,.2,.5,3),[3.55,3.485,3.4395])
    assert garch11_variance_path([1],1,.2,.5)[0]==pytest.approx(10/3)


def test_igarch_finite_paths_have_no_stationary_variance():
    with pytest.raises(ValueError): garch11_unconditional_variance(1,.2,.8)
    with pytest.raises(ValueError): garch11_variance_path([1,2],1,.2,.8)
    np.testing.assert_allclose(garch11_variance_path([1,2],1,.2,.8,4),[4,4.4])
    np.testing.assert_allclose(garch11_forecast(1,4,1,.2,.8,3),[4.4,5.4,6.4])


def test_forecast_convergence():
    values=garch11_forecast(10,2,1,.2,.5,100)
    assert values[-1]==pytest.approx(10/3)
    assert values[0]>values[1]>values[-1]


def test_simulation_reproducibility_and_recursion():
    for function,args in [(simulate_arch1,(2,.4,100)),(simulate_garch11,(1,.1,.8,100))]:
        a,h=function(*args,rng=606)
        a2,h2=function(*args,rng=606)
        np.testing.assert_array_equal(a,a2);np.testing.assert_array_equal(h,h2)
        assert len(a)==100 and np.all(h>0)
        omega,alpha=args[:2];beta=0 if function is simulate_arch1 else args[2]
        np.testing.assert_allclose(h[1:],omega+alpha*a[:-1]**2+beta*h[:-1])
    a,h=simulate_garch11(1,0,0,100,rng=606)
    assert np.all(h==1)


def test_simulated_population_variance_low_persistence():
    a,h=simulate_arch1(2,.2,50000,rng=604)
    assert abs(np.var(a)-2.5)<.08
    assert abs(h.mean()-2.5)<.05


def test_standardization_and_squared_portmanteau():
    np.testing.assert_allclose(standardized_residuals([2,-3],[2,1]),[1,-3])
    x=np.random.default_rng(608).normal(size=100)
    ours=mcleod_li(x,10)
    reference=acorr_ljungbox(x*x,lags=[10]).iloc[0]
    assert ours['statistic']==pytest.approx(reference.lb_stat)
    assert ours['p_value']==pytest.approx(reference.lb_pvalue)


@pytest.mark.parametrize('omega,alpha,beta',[(0,.1,.8),(-1,.1,.8),(1,-.1,.8),(1,.1,-.1),(1,.5,.5),(1,.2,.9),(np.nan,.1,.8),(1,np.inf,.8)])
def test_invalid_stationary_parameters(omega,alpha,beta):
    with pytest.raises(ValueError): garch11_unconditional_variance(omega,alpha,beta)
    with pytest.raises(ValueError): simulate_garch11(omega,alpha,beta,10)


@pytest.mark.parametrize('omega,alpha',[(0,.2),(1,-.1),(1,1),(1,np.nan)])
def test_invalid_arch_parameters(omega,alpha):
    with pytest.raises(ValueError): arch1_unconditional_variance(omega,alpha)
    with pytest.raises(ValueError): simulate_arch1(omega,alpha,10)


@pytest.mark.parametrize('bad',[0,-1,1.5,True])
def test_invalid_forecast_horizons_and_lengths(bad):
    with pytest.raises(ValueError): garch11_forecast(1,1,1,.1,.8,bad)
    with pytest.raises(ValueError): simulate_garch11(1,.1,.8,bad)


@pytest.mark.parametrize('a,h',[([1],[0]),([1],[-1]),([1],[np.nan]),([1,2],[1]),([],[]),([1j],[1])])
def test_invalid_standardization(a,h):
    with pytest.raises(ValueError): standardized_residuals(a,h)


@pytest.mark.parametrize('initial',[0,-1,np.nan,np.inf])
def test_invalid_initial_variance(initial):
    with pytest.raises(ValueError): garch11_variance_path([1,2],1,.1,.8,initial)
    with pytest.raises(ValueError): garch11_forecast(1,initial,1,.1,.8)


def test_path_invalid_residuals_and_parameters():
    for x in [[],[np.nan],[[1,2]],[1j]]:
        with pytest.raises(ValueError): garch11_variance_path(x,1,.1,.8)
    for args in [(0,.1,.8),(1,-.1,.8),(1,.1,-.1)]:
        with pytest.raises(ValueError): garch11_forecast(1,1,*args)
        with pytest.raises(ValueError): garch11_variance_path([1],*args,initial_variance=1)


def test_student_innovations_are_standardized_and_aligned():
    a, h = simulate_garch11(1, 0, 0, 80, rng=42, df=5)
    expected = np.random.default_rng(42).standard_t(5, 1080)[1000:] * np.sqrt(3/5)
    np.testing.assert_array_equal(a, expected)
    np.testing.assert_array_equal(h, np.ones(80))
    a, h = simulate_garch11(.05, .05, .9, 80, rng=42, df=5)
    np.testing.assert_allclose(h[1:], .05+.05*a[:-1]**2+.9*h[:-1])


@pytest.mark.parametrize("df", [1, 2, np.nan, np.inf])
def test_student_simulation_requires_finite_variance(df):
    with pytest.raises(ValueError):
        simulate_garch11(1, 0, 0, 10, df=df)
