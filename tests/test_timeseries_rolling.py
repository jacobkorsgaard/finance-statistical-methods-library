import numpy as np
import pandas as pd
from scipy.signal import lfilter
from finstats.joint import fit_arma_garch
from finstats.rolling import expanding_arma_garch_forecasts


def test_expanding_refits_are_causal_and_cached_records_match(tmp_path):
    rng=np.random.default_rng(14)
    values=lfilter([1,.2],[1,-.3],rng.normal(size=164))
    initial=fit_arma_garch(values[:160],[.3],[.2],0.,n_starts=1,compute_inference=False)
    path=tmp_path/'forecasts.csv'
    forecasts=expanding_arma_garch_forecasts(values,initial,n_starts=1,cache_path=path)
    assert len(forecasts)==4
    np.testing.assert_array_equal(forecasts.target_index,[160,161,162,163])
    assert (forecasts.variance>0).all() and (forecasts.ES>=forecasts.VaR).all()
    assert (forecasts.converged_starts>=1).all()
    cached=expanding_arma_garch_forecasts(values,initial,n_starts=1,cache_path=path)
    forecasts["df"]=forecasts["df"].astype(float)
    cached["df"]=cached["df"].astype(float)
    pd.testing.assert_frame_equal(forecasts,cached,check_dtype=False)
    changed=values.copy();changed[-1]=100.
    revised=expanding_arma_garch_forecasts(changed,initial,n_starts=1)
    # Even the last forecast is made before seeing the changed target.
    np.testing.assert_array_equal(forecasts[['mean','variance','VaR','ES']],revised[['mean','variance','VaR','ES']])
    # Input changes invalidate the saved calculation rather than retaining stale rows.
    invalidated=expanding_arma_garch_forecasts(changed,initial,n_starts=1,cache_path=path)
    revised["df"]=revised["df"].astype(float)
    invalidated["df"]=invalidated["df"].astype(float)
    pd.testing.assert_frame_equal(revised,invalidated,check_dtype=False)


def test_warm_fit_retains_specification_and_skips_inference():
    values=np.random.default_rng(12).standard_t(6,220)*np.sqrt(4/6)
    initial=fit_arma_garch(values[:210],[.2],[.1],0.,distribution='t',n_starts=1)
    warm=fit_arma_garch(values[:211],initial.ar,initial.ma,initial.mu,
        distribution='t',start_fit=initial,n_starts=2,compute_inference=False)
    assert warm.inference is None
    assert len(warm.starts)==2 and any(row['converged'] for row in warm.starts)
    assert warm.df>2 and warm.nobs==209
