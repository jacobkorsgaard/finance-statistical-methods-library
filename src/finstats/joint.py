"""Conditional joint ARMA–GARCH estimation and fixed-parameter prediction.

The mean uses an unconditional location mu and plus-MA coefficients. Presample
centered observations and innovations are zero; variance starts at the sample
variance of the first 75 observations. This is conditional, not exact likelihood.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, softmax
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate, unconstrain_stationary_univariate,
)
from ._validation import _finite_scalar


def _series(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("values must be a finite one-dimensional sample of length >=3")
    return x


def filter_arma_garch(values, mu, ar, ma, omega, alpha, beta, initial_variance):
    """Return conditional means, innovations and variances without look-ahead.

    initial_variance is supplied explicitly so future observations cannot change
    the initialization. Require stationary GARCH coefficients and invertible MA.
    """
    x = _series(values)
    mu = _finite_scalar(mu, "mu")
    ar, ma = np.asarray(ar, dtype=float), np.asarray(ma, dtype=float)
    if ar.ndim != 1 or ma.ndim != 1 or not np.isfinite(ar).all() or not np.isfinite(ma).all():
        raise ValueError("ar and ma must be finite coefficient vectors")
    from .linear import is_stationary_ar
    from .linear import is_invertible_ma
    if len(ar) and not is_stationary_ar(ar):
        raise ValueError("AR coefficients must be stationary")
    if len(ma) and not is_invertible_ma(ma):
        raise ValueError("MA coefficients must be invertible")
    omega = _finite_scalar(omega,"omega")
    initial_variance = _finite_scalar(initial_variance,"initial_variance")
    alpha,beta = np.atleast_1d(np.asarray(alpha,float)),np.atleast_1d(np.asarray(beta,float))
    if (alpha.ndim != 1 or beta.ndim != 1 or not len(alpha) or not len(beta)
            or not np.isfinite(alpha).all() or not np.isfinite(beta).all()):
        raise ValueError("alpha and beta must be nonempty finite coefficient vectors")
    if omega <= 0 or min(alpha.min(), beta.min()) < 0 or alpha.sum()+beta.sum() >= 1 or initial_variance <= 0:
        raise ValueError("require positive variance, omega>0 and alpha,beta>=0 with sum<1")
    return _filter(x, mu, ar, ma, omega, alpha, beta, initial_variance)


def _filter(x, mu, ar, ma, omega, alpha, beta, initial_variance):
    residuals = lfilter(np.r_[1., -ar], np.r_[1., ma], x-mu)
    alpha,beta = np.atleast_1d(alpha),np.atleast_1d(beta)
    inputs = omega+np.convolve(residuals**2,np.r_[0.,alpha])[:len(x)]
    # Missing presample squares and variances use the fixed initialization.
    for lag,coefficient in enumerate(alpha,1):
        inputs[1:min(lag,len(x))] += coefficient*initial_variance
    for lag,coefficient in enumerate(beta,1):
        inputs[1:min(lag,len(x))] += coefficient*initial_variance
    inputs[0] = initial_variance
    variance = lfilter([1.],np.r_[1.,-beta],inputs)
    return x-residuals, residuals, variance


def _wald_inference(loglikelihood_obs, parameters):
    """Nominal sandwich SE and two-sided Wald p-values for an interior fit.

    Singular/indefinite curvature is reported as unavailable, never repaired by
    a pseudoinverse. The caller must flag nonregular boundary fits separately.
    """
    parameters = np.asarray(parameters,float)
    try:
        hessian = approx_hess(parameters,lambda theta:-np.sum(loglikelihood_obs(theta)))
        hessian = (hessian+hessian.T)/2
        scores = approx_fprime(parameters,loglikelihood_obs,centered=True)
        if not np.isfinite(hessian).all() or not np.isfinite(scores).all():
            raise ValueError("nonfinite numerical derivatives")
        eigenvalues = np.linalg.eigvalsh(hessian)
        if eigenvalues[0] <= 0 or eigenvalues[-1]/eigenvalues[0] > 1e12:
            raise ValueError("singular or indefinite likelihood curvature")
        inverse = np.linalg.inv(hessian)
        covariance = inverse @ (scores.T@scores) @ inverse
        if np.any(np.diag(covariance)<=0) or not np.isfinite(covariance).all():
            raise ValueError("invalid sandwich covariance")
        standard_errors = np.sqrt(np.diag(covariance))
        pvalues = 2*norm.sf(np.abs(parameters/standard_errors))
        return standard_errors,pvalues,"Computed"
    except (ValueError,np.linalg.LinAlgError,FloatingPointError):
        missing=np.full(len(parameters),np.nan)
        return missing,missing.copy(),"Unavailable: unstable likelihood curvature"


@dataclass
class JointARMAGARCHFit:
    """Conditional MLE with recorded optimizer starts; no exact-MLE claim."""
    mu: float
    ar: np.ndarray
    ma: np.ndarray
    omega: float
    alpha: float | np.ndarray
    beta: float | np.ndarray
    df: float | None
    values: np.ndarray
    initial_variance: float
    hold_back: int
    loglikelihood: float
    starts: list
    inference: dict | None = None

    @property
    def params(self):
        return dict(mu=self.mu, **{f"ar[{i}]":v for i,v in enumerate(self.ar,1)},
                    **{f"ma[{i}]":v for i,v in enumerate(self.ma,1)},
                    omega=self.omega,
                    **{f"alpha[{i}]":v for i,v in enumerate(np.atleast_1d(self.alpha),1)},
                    **{f"beta[{i}]":v for i,v in enumerate(np.atleast_1d(self.beta),1)},
                    **({"nu":self.df} if self.df is not None else {}))

    @property
    def nobs(self):
        return len(self.values)-self.hold_back

    @property
    def aic(self):
        return -2*self.loglikelihood+2*len(self.params)

    @property
    def bic(self):
        return -2*self.loglikelihood+np.log(self.nobs)*len(self.params)

    @property
    def volatility_order(self):
        """(variance lags, squared-innovation lags), matching Notebook 09."""
        return len(np.atleast_1d(self.beta)),len(np.atleast_1d(self.alpha))

    def filter(self, values=None):
        """Filter a full history with the original initialization and fixed parameters."""
        return filter_arma_garch(self.values if values is None else values, self.mu,
            self.ar, self.ma, self.omega, self.alpha, self.beta, self.initial_variance)

    def forecast(self, horizon, simulations=20000, seed=1111):
        """Single-origin conditional means, innovation variances and predictive paths."""
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(simulations, (int, np.integer)) or simulations < 1:
            raise ValueError("simulations must be a positive integer")
        _, residuals, variance = self.filter()
        p, q = len(self.ar), len(self.ma)
        # Conditional expectations: unobserved future innovations are zero.
        xs = list(self.values); es = list(residuals)
        means = []
        for _ in range(horizon):
            m = self.mu+sum(a*(xs[-j]-self.mu) for j,a in enumerate(self.ar,1))
            m += sum(b*es[-j] for j,b in enumerate(self.ma,1))
            means.append(m); xs.append(m); es.append(0.)
        alpha,beta = np.atleast_1d(self.alpha),np.atleast_1d(self.beta)
        na,nb = len(alpha),len(beta)
        expected_variance = np.empty(horizon)
        squares = list(residuals**2); variance_history = list(variance)
        for t in range(horizon):
            h_next = self.omega+sum(a*squares[-j] for j,a in enumerate(alpha,1))
            h_next += sum(b*variance_history[-j] for j,b in enumerate(beta,1))
            expected_variance[t] = h_next
            squares.append(h_next); variance_history.append(h_next)
        rng = np.random.default_rng(seed)
        z = (rng.normal(size=(simulations,horizon)) if self.df is None else
             rng.standard_t(self.df,size=(simulations,horizon))*np.sqrt((self.df-2)/self.df))
        xs = np.tile(self.values[-p:] if p else [], (simulations,1))
        es = np.tile(residuals[-max(q,na):], (simulations,1))
        hs = np.tile(variance[-nb:],(simulations,1))
        paths = np.empty((simulations,horizon))
        for t in range(horizon):
            m = np.full(simulations,self.mu)
            for j,a in enumerate(self.ar,1): m += a*(xs[:,-j]-self.mu)
            for j,b in enumerate(self.ma,1): m += b*es[:,-j]
            h = np.full(simulations,self.omega)
            for j,a in enumerate(alpha,1): h += a*es[:,-j]**2
            for j,b in enumerate(beta,1): h += b*hs[:,-j]
            e = np.sqrt(h)*z[:,t]; paths[:,t] = m+e
            xs = np.column_stack((xs,paths[:,t])); es = np.column_stack((es,e))
            hs = np.column_stack((hs,h))
        return np.asarray(means), expected_variance, paths


def fit_arma_garch(values, ar_start, ma_start, mu_start, distribution="normal", *,
                   hold_back=2, n_starts=3, seed=111, garch_order=(1,1),
                   start_fit=None, compute_inference=True):
    """Fit a stationary/invertible ARMA–GARCH(p,q) by conditional joint MLE.

    garch_order uses (variance lags, squared-innovation lags), opposite to
    arch software argument names. Missing presample squares/variances use
    initial_variance; the first hold_back likelihood contributions are omitted.

    Gaussian or unit-variance Student-t density; parameter transformations enforce
    root and variance constraints. Multiple deterministic starts do not guarantee
    a global optimum. start_fit adds a warm optimizer start; compute_inference=False
    skips coefficient uncertainty for repeated forecasting fits. The returned fit records success and objectives for every
    start. Boundary fits should not be assigned a reliable long-run variance.
    """
    x = _series(values)
    ar_start, ma_start = np.asarray(ar_start, float), np.asarray(ma_start, float)
    p, q = len(ar_start), len(ma_start)
    if distribution not in ("normal", "t"):
        raise ValueError("distribution must be normal or t")
    if not isinstance(hold_back, int) or not max(p,q) <= hold_back < len(x)-2:
        raise ValueError("hold_back must cover mean lags and leave at least three observations")
    if not isinstance(n_starts, int) or n_starts < 1:
        raise ValueError("n_starts must be positive")
    if (len(garch_order)!=2 or any(not isinstance(v,int) or v<1 for v in garch_order)):
        raise ValueError("garch_order must contain positive variance and shock lag counts")
    nb,na = garch_order
    if max(na,nb)>hold_back:
        raise ValueError("hold_back must also cover variance and squared-innovation lags")
    initial_variance = float(np.var(x[:min(75,len(x))],ddof=1))
    if initial_variance <= 0:
        raise ValueError("nonconstant observations required")
    mu_start = _finite_scalar(mu_start,"mu_start")
    # Validate starts before mapping coefficients to unconstrained root parameters.
    filter_arma_garch(x,mu_start,ar_start,ma_start,.1,.1,.8,initial_variance)
    initial = np.r_[mu_start, unconstrain_stationary_univariate(ar_start) if p else [],
        unconstrain_stationary_univariate(-ma_start) if q else [],
        np.log(.04*np.var(x)), np.repeat(np.log((.08/na)/.04),na),
        np.repeat(np.log((.88/nb)/.04),nb),
        [np.log(6.)] if distribution=="t" else []]

    def decode(theta):
        ar = constrain_stationary_univariate(theta[1:1+p]) if p else np.array([])
        ma = -constrain_stationary_univariate(theta[1+p:1+p+q]) if q else np.array([])
        offset = 1+p+q
        omega = np.exp(theta[offset])
        weights = softmax(np.r_[theta[offset+1:offset+1+na+nb],0.])
        alpha,beta = weights[:na]*(1-1e-8),weights[na:na+nb]*(1-1e-8)
        df = 2+np.exp(theta[-1]) if distribution=="t" else None
        return theta[0],ar,ma,omega,alpha,beta,df

    def objective(theta):
        mu,ar,ma,omega,alpha,beta,df = decode(theta)
        _,e,h = _filter(x,mu,ar,ma,omega,alpha,beta,initial_variance)
        e,h = e[hold_back:],h[hold_back:]
        if not np.isfinite(e).all() or not np.isfinite(h).all() or np.any(h<=0):
            return 1e100
        if df is None:
            log_density = -.5*(np.log(2*np.pi)+np.log(h)+e**2/h)
        else:
            log_density = (gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(np.pi*(df-2))
                           -.5*np.log(h)-(df+1)/2*np.log1p(e**2/(h*(df-2))))
        return -float(np.sum(log_density))

    bounds = [(None,None)]+[(-20,20)]*(p+q)+[(-20,10)]+[(-15,15)]*(na+nb)
    if distribution=="t": bounds += [(-5,7)]
    warm = None
    if start_fit is not None:
        if (len(start_fit.ar)!=p or len(start_fit.ma)!=q or start_fit.volatility_order!=garch_order
                or (start_fit.df is not None)!=(distribution=="t")):
            raise ValueError("start_fit must have the same model orders and density")
        weights = np.r_[np.atleast_1d(start_fit.alpha),np.atleast_1d(start_fit.beta)]
        slack = max(1-weights.sum(),1e-8)
        warm = np.r_[start_fit.mu,
            unconstrain_stationary_univariate(start_fit.ar) if p else [],
            unconstrain_stationary_univariate(-start_fit.ma) if q else [],
            np.log(start_fit.omega),np.log(np.maximum(weights,1e-12)/slack),
            [np.log(start_fit.df-2)] if distribution=="t" else []]
        for i,(lower,upper) in enumerate(bounds):
            if lower is not None: warm[i]=max(warm[i],lower)
            if upper is not None: warm[i]=min(warm[i],upper)
    generator = np.random.default_rng(seed)
    results = []
    for k in range(n_starts):
        if warm is not None:
            # Warm solution and an interior variance reset avoid inheriting only a boundary solution.
            start = warm.copy() if k==0 else initial.copy() if k==1 else initial+generator.normal(0,.35,len(initial))
        else:
            start = initial.copy() if k==0 else initial+generator.normal(0,.35,len(initial))
        result = minimize(objective,start,method="L-BFGS-B",bounds=bounds,
                          options={"maxiter":1500,"ftol":1e-10,"gtol":1e-5,"maxls":40})
        results.append(result)
    successful = [r for r in results if r.success and np.isfinite(r.fun)]
    if not successful:
        raise RuntimeError("No joint optimization start converged")
    best = min(successful,key=lambda r:r.fun)
    mu,ar,ma,omega,alpha,beta,df = decode(best.x)
    fit = JointARMAGARCHFit(mu,ar,ma,omega,alpha,beta,df,x.copy(),initial_variance,
        hold_back,-best.fun,[{"converged":bool(r.success),"loglikelihood":-float(r.fun),
                            "message":str(r.message)} for r in results])
    if not compute_inference:
        return fit
    names = list(fit.params)
    natural_parameters = np.array(list(fit.params.values()))

    def natural_loglikelihood_obs(theta):
        offset = 1+p+q
        mu,ar,ma = theta[0],theta[1:1+p],theta[1+p:offset]
        omega = theta[offset]
        alpha = theta[offset+1:offset+1+na]
        beta = theta[offset+1+na:offset+1+na+nb]
        _,e,h = _filter(x,mu,ar,ma,omega,alpha,beta,initial_variance)
        e,h = e[hold_back:],h[hold_back:]
        if not np.isfinite(h).all() or np.any(h<=0):
            return np.full(len(h),np.nan)
        if distribution=="normal":
            return -.5*(np.log(2*np.pi)+np.log(h)+e**2/h)
        df = theta[-1]
        if df<=2: return np.full(len(h),np.nan)
        return (gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(np.pi*(df-2))
                -.5*np.log(h)-(df+1)/2*np.log1p(e**2/(h*(df-2))))

    se,pvalues,status = _wald_inference(natural_loglikelihood_obs,natural_parameters)
    boundary = alpha.sum()+beta.sum()>=1-1e-4 or min(alpha.min(),beta.min())<=1e-6
    if boundary and status=="Computed": status="Nominal only: near a variance-parameter boundary"
    fit.inference = {name:{"standard_error":error,"p_value":pvalue,"status":status}
                     for name,error,pvalue in zip(names,se,pvalues)}
    # df=0 is outside the Student-t parameter space, so it is not a useful null.
    if "nu" in fit.inference: fit.inference["nu"]["p_value"]=np.nan
    return fit
