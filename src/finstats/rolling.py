"""Expanding-window joint-model refits and one-step forecasts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from .risk import gaussian_var, gaussian_es, student_t_var, student_t_es
from .joint import JointARMAGARCHFit, fit_arma_garch, _series


def _refit_signature(values, initial_fit, n_starts, seed):
    digest=hashlib.sha256()
    digest.update(np.asarray(values,dtype='<f8').tobytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(Path(__file__).with_name('joint.py').read_bytes())
    settings={'training_size':len(initial_fit.values),'initial_parameters':initial_fit.params,
              'n_starts':n_starts,'seed':seed,'numpy':np.__version__,'scipy':scipy.__version__}
    digest.update(json.dumps(settings,sort_keys=True).encode())
    return digest.hexdigest()


def expanding_arma_garch_forecasts(values, initial_fit, *, n_starts=2, seed=111,
                                    cache_path=None, progress=None):
    """Refit on values[:t] before forecasting values[t], with fixed model orders.

    Use the previous fit as one start and an interior variance reset as another.
    No target or later observation enters its own fit. Optional CSV checkpoints
    include all parameters, input/source/settings signature and a byte checksum.
    A valid saved record resumes/completes the same calculation, never replaces
    the algorithm with fixed-parameter forecasts. progress(done,total) is optional.
    """
    x=_series(values)
    training_size=len(initial_fit.values)
    if not 3<=training_size<len(x) or not np.array_equal(x[:training_size],initial_fit.values):
        raise ValueError('initial_fit must contain exactly the training prefix')
    if not isinstance(n_starts,int) or n_starts<1:
        raise ValueError('n_starts must be positive')
    signature=_refit_signature(x,initial_fit,n_starts,seed)
    path=Path(cache_path) if cache_path is not None else None
    metadata=path.with_suffix('.json') if path is not None else None
    rows=[]
    if path is not None and path.exists() and metadata.exists():
        info=json.loads(metadata.read_text())
        if (info.get('signature')==signature and
                info.get('csv_sha256')==hashlib.sha256(path.read_bytes()).hexdigest()):
            frame=pd.read_csv(path,float_precision='round_trip')
            targets=frame.target_index.to_numpy()
            if np.array_equal(targets,np.arange(training_size,training_size+len(frame))):
                rows=frame.to_dict('records')
    fit=initial_fit
    if rows:
        params=json.loads(rows[-1]['parameters'])
        p,q=len(fit.ar),len(fit.ma);nb,na=fit.volatility_order
        fitted_size=int(rows[-1]['target_index'])
        fit=JointARMAGARCHFit(params['mu'],np.array([params[f'ar[{i}]'] for i in range(1,p+1)]),
            np.array([params[f'ma[{i}]'] for i in range(1,q+1)]),params['omega'],
            np.array([params[f'alpha[{i}]'] for i in range(1,na+1)]),
            np.array([params[f'beta[{i}]'] for i in range(1,nb+1)]),params.get('nu'),
            x[:fitted_size].copy(),initial_fit.initial_variance,initial_fit.hold_back,
            rows[-1]['loglikelihood'],[])
    total=len(x)-training_size

    def checkpoint():
        if path is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            frame=pd.DataFrame(rows)
            temporary=path.with_suffix('.tmp')
            frame.to_csv(temporary,index=False,float_format='%.17g')
            temporary.replace(path)
            metadata.write_text(json.dumps({'signature':signature,'rows':len(rows),
                'complete':len(rows)==total,'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')

    for target in range(training_size+len(rows),len(x)):
        fit=fit_arma_garch(x[:target],fit.ar,fit.ma,fit.mu,
            distribution='normal' if fit.df is None else 't',hold_back=fit.hold_back,
            n_starts=n_starts,seed=seed+target,garch_order=fit.volatility_order,
            start_fit=fit,compute_inference=False)
        _,e,h=fit.filter()
        mean=fit.mu+sum(a*(x[target-j]-fit.mu) for j,a in enumerate(fit.ar,1))
        mean+=sum(b*e[-j] for j,b in enumerate(fit.ma,1))
        variance=fit.omega+sum(a*e[-j]**2 for j,a in enumerate(np.atleast_1d(fit.alpha),1))
        variance+=sum(b*h[-j] for j,b in enumerate(np.atleast_1d(fit.beta),1))
        sd=np.sqrt(variance)
        if fit.df is None:
            var,es=gaussian_var(-mean,sd,.05),gaussian_es(-mean,sd,.05)
        else:
            scale=sd*np.sqrt((fit.df-2)/fit.df)
            var,es=student_t_var(-mean,scale,fit.df,.05),student_t_es(-mean,scale,fit.df,.05)
        rows.append({'target_index':target,'mean':mean,'variance':variance,'df':fit.df,
                     'VaR':var,'ES':es,'loglikelihood':fit.loglikelihood,
                     'persistence':np.sum(fit.alpha)+np.sum(fit.beta),
                     'converged_starts':sum(record['converged'] for record in fit.starts),
                     'parameters':json.dumps(fit.params,sort_keys=True)})
        if progress is not None:progress(len(rows),total)
        if len(rows)%25==0:checkpoint()
    checkpoint()
    frame=pd.DataFrame(rows)
    assert len(frame)==total
    return frame
