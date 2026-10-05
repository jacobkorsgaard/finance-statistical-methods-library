import numpy as np
import pytest
from statsmodels.tsa.stattools import acf as reference_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from finstats.timeseries.diagnostics import sample_autocovariance, sample_autocorrelation, acf, ljung_box
from finstats.timeseries.ar import simulate_ar1, ar1_unconditional_variance, ar1_acf, ar_roots, is_stationary_ar, ar1_forecast
from finstats.timeseries.ma import simulate_ma, ma_roots, is_invertible_ma
from finstats.timeseries.arma import arma_roots, is_stationary, is_invertible
from finstats.timeseries.nonstationary import difference, simulate_random_walk
from finstats.timeseries.forecasting import forecast_errors, forecast_mse, forecast_rmse, forecast_mae


def test_lag_moments_hand_example():
    x=[1,2,3]
    assert sample_autocovariance(x,0)==pytest.approx(2/3)
    assert sample_autocovariance(x,1)==0
    assert sample_autocovariance(x,2)==-1
    assert sample_autocorrelation(x,2)==pytest.approx(-1.5)
    assert sample_autocovariance(x,2,adjusted=False)==pytest.approx(-1/3)
    np.testing.assert_allclose(acf(x,2),[1,0,-1.5])


@pytest.mark.parametrize('adjusted',[True,False])
def test_acf_reference_conventions(adjusted):
    x=np.random.default_rng(501).normal(size=100)
    np.testing.assert_allclose(acf(x,10,adjusted),reference_acf(x,nlags=10,adjusted=adjusted,fft=False))


def test_ljung_box_reference():
    x=np.random.default_rng(502).normal(size=150)
    for df in [0,2]:
        ours=ljung_box(x,10,df)
        trusted=acorr_ljungbox(x,lags=[10],model_df=df).iloc[0]
        assert ours['statistic']==pytest.approx(trusted.lb_stat)
        assert ours['p_value']==pytest.approx(trusted.lb_pvalue)
        assert ours['df']==10-df


def test_ar1_moments_forecasts_roots():
    assert ar1_unconditional_variance(.5,3)==4
    np.testing.assert_allclose(ar1_acf(-.5,3),[1,-.5,.25,-.125])
    np.testing.assert_allclose(ar1_forecast(10,.5,2,3),[6,4,3])
    assert ar_roots([.5])[0]==2
    assert is_stationary_ar([1,-.75])
    assert not is_stationary_ar([1])
    assert not is_stationary_ar([1.1])
    assert is_stationary_ar([]) and is_stationary_ar([0,0])
    assert np.all(np.abs(ar_roots([1,-.75]))>1)


def test_ma_roots_and_reproducibility():
    assert ma_roots([.5])[0]==-2
    assert is_invertible_ma([.5])
    assert not is_invertible_ma([1])
    assert not is_invertible_ma([2])
    assert is_invertible_ma([])
    expected=arma_roots([.5],[.5])
    np.testing.assert_allclose(expected[0],[2]);np.testing.assert_allclose(expected[1],[-2])
    assert is_stationary([.5]) and is_invertible([.5])
    a=simulate_ma([.5],2,5,mu=3,rng=np.random.default_rng(10))
    innovations=np.random.default_rng(10).normal(0,2,6)
    np.testing.assert_allclose(a,3+innovations[1:]+.5*innovations[:-1])


def test_ar1_exact_seed_and_population_moments():
    a=simulate_ar1(.7,1,20,rng=503)
    np.testing.assert_array_equal(a,simulate_ar1(.7,1,20,rng=503))
    # Test initialization, not just a long burn-in convergence claim.
    first=np.array([simulate_ar1(.7,1,1,mu=2,rng=i)[0] for i in range(4000)])
    assert abs(first.mean()-2)<.08
    assert abs(first.var()-1/(1-.49))<.12


def test_walk_indexing_and_difference():
    innovations=np.random.default_rng(4).normal(0,2,5)
    path=simulate_random_walk(5,2,initial=3,rng=4)
    assert path.size==6 and path[0]==3
    np.testing.assert_allclose(difference(path),innovations)
    np.testing.assert_allclose(difference([1,4,9,16],2),[2,2])
    np.testing.assert_array_equal(difference([1,2],0),[1,2])


def test_forecast_metrics():
    np.testing.assert_array_equal(forecast_errors([1,4],[2,2]),[-1,2])
    assert forecast_mse([1,4],[2,2])==2.5
    assert forecast_rmse([1,4],[2,2])==np.sqrt(2.5)
    assert forecast_mae([1,4],[2,2])==1.5


@pytest.mark.parametrize('bad',[[],[np.nan],[np.inf],[[1,2]],[1j]])
def test_invalid_series(bad):
    for f,args in [(sample_autocovariance,(bad,0)),(sample_autocorrelation,(bad,0)),
                   (acf,(bad,1)),(ljung_box,(bad,1)),(difference,(bad,))]:
        with pytest.raises(ValueError): f(*args)


@pytest.mark.parametrize('bad',[-1,1.5,True,3,np.nan])
def test_invalid_lags(bad):
    for f in [sample_autocovariance,sample_autocorrelation,acf]:
        with pytest.raises(ValueError): f([1,2,3],bad)


@pytest.mark.parametrize('phi',[-1,1,1.1,np.nan,np.inf])
def test_nonstationary_ar1(phi):
    for f,args in [(ar1_unconditional_variance,(phi,1)),(ar1_acf,(phi,5)),
                   (simulate_ar1,(phi,1,5)),(ar1_forecast,(2,phi,0,3))]:
        with pytest.raises(ValueError): f(*args)


@pytest.mark.parametrize('bad',[0,-1,1.5,True])
def test_invalid_sizes(bad):
    for f,args in [(simulate_ar1,(.5,1,bad)),(simulate_ma,([.5],1,bad)),
                   (simulate_random_walk,(bad,)),(ar1_forecast,(1,.5,0,bad))]:
        with pytest.raises(ValueError): f(*args)


@pytest.mark.parametrize('coeff',[[np.nan],[[.5]],[1j]])
def test_invalid_coefficients(coeff):
    for f in [ar_roots,is_stationary_ar,ma_roots,is_invertible_ma]:
        with pytest.raises(ValueError): f(coeff)


def test_diagnostic_and_forecast_restrictions():
    with pytest.raises(ValueError): acf([1,1],1)
    for args in [(0,0),(3,0),(2,2),(2,-1)]:
        with pytest.raises(ValueError): ljung_box([1,2,4],args[0],args[1])
    for order in [-1,3,True]:
        with pytest.raises(ValueError): difference([1,2,3],order)
    for f in [forecast_errors,forecast_mse,forecast_rmse,forecast_mae]:
        with pytest.raises(ValueError): f([1,2],[1])
        with pytest.raises(ValueError): f([np.nan],[1])
    for f,args in [(simulate_ar1,(.5,-1,10)),(simulate_ma,([.5],0,10)),
                   (simulate_random_walk,(10,0)),(ar1_unconditional_variance,(.5,0))]:
        with pytest.raises(ValueError): f(*args)
