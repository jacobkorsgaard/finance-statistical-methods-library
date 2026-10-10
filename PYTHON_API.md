# Python API guide

This repository is an educational reference with a focused set of reusable implementations in `finstats`. The notebooks supply the mathematical explanations and empirical applications; NumPy, SciPy, statsmodels, pmdarima and arch supply standard methods. This guide covers both, rather than implying that every demonstrated method has a custom implementation.

Install with `python -m pip install -e ".[dev]"`. Examples below can be run in order in that environment. Empirical notebooks use local data and need no downloads. See [README](README.md) for the reading sequence, tested environment and sources.

## 1. Conventions and providers

Observations occupy rows of multivariate arrays `(n, k)`. Univariate custom functions expect finite real one-dimensional observations; they reject invalid inputs rather than silently removing missing values. Third-party functions retain their own validation behavior.

Returns and losses must use consistent units: a decimal return of `-.02` means −2%, and its relative loss is `.02`. The empirical applications multiply returns by 100, so variances there have units of percentage points squared. For simple returns, loss is `L = -R`; negative log return is an approximation to this relative loss.

Sample variance uses `ddof=1`; Gaussian MLE variance uses `ddof=0`. Skewness and kurtosis in the notebooks use denominator-n central moments (`bias=True`); raw kurtosis uses `fisher=False`. Student-t scale is **not** its standard deviation. A multivariate scale matrix is **not generally** a covariance matrix.

| Task | Provider / interface |
| --- | --- |
| Elementary moments, returns, linear combinations, forecast errors | Explicit NumPy/SciPy calculations |
| Univariate densities, CDFs, quantiles and sampling | `scipy.stats` |
| Comparable univariate fit summaries | `finstats.fitting.fit_distribution`, delegating estimation to SciPy |
| Multivariate Gaussian evaluation/sampling | `scipy.stats.multivariate_normal` |
| Multivariate Student-t evaluation/sampling and custom estimation | `scipy.stats.multivariate_t`; `finstats.multivariate_fit` |
| Multivariate skew-normal and bivariate symmetric Laplace | `finstats.multivariate` |
| ECDF, inverse ECDF and QQ positions | `finstats.empirical` |
| Kernel density estimation | `scipy.stats.gaussian_kde` |
| Linear time-series estimation and automatic order search | statsmodels `ARIMA`; pmdarima `auto_arima` |
| Standard ARCH/GARCH estimation | `arch.arch_model` |
| Joint ARMA–GARCH conditional estimation | `finstats.joint` |
| Expanding-window joint refits | `finstats.rolling` |
| Explicit process simulations and variance recursions | `finstats.simulation`, `finstats.volatility`, `finstats.linear` |
| Dependence diagnostics and financial tail formulas | `finstats.diagnostics`, `finstats.risk` |

## 2. Returns, moments and standard inference

[Notebooks 00](notebooks/00_financial_returns.ipynb) and [03](notebooks/03_statistical_inference_and_mle.ipynb) show these calculations directly. Dividend-adjusted prices already include the adjustment: do not add dividends twice.

```python
import numpy as np
from scipy import stats

prices = np.array([100., 105., 101., 108.])
gross_returns = prices[1:] / prices[:-1]
returns = gross_returns - 1
log_returns = np.log(gross_returns)
losses = -returns
x = np.array([-.02, .01, .005, -.01, .03, -.015, .004, .007])
mean = x.mean()
variance = x.var(ddof=1)
standard_deviation = x.std(ddof=1)
skewness = stats.skew(x, bias=True)
raw_kurtosis = stats.kurtosis(x, fisher=False, bias=True)
```

For IID Gaussian observations with unknown variance, the mean interval uses Student-t and the variance interval uses chi-square. These are standard inference calculations, not separate project modules.

```python
n = len(x)
mean_interval = stats.t.interval(.95, n-1, loc=x.mean(), scale=stats.sem(x))
variance_interval = (n-1)*x.var(ddof=1) / stats.chi2.ppf([.975, .025], n-1)
mean_test = stats.ttest_1samp(x, popmean=0.)
sigma2_null = .02**2
variance_statistic = (n-1)*x.var(ddof=1) / sigma2_null
variance_pvalue = min(1., 2*min(stats.chi2.cdf(variance_statistic, n-1),
                              stats.chi2.sf(variance_statistic, n-1)))
```

For simulated estimator replications `estimates`, bias is `estimates.mean()-truth`, replication variance is `estimates.var(ddof=0)`, and MSE is `np.mean((estimates-truth)**2)`. Their exact finite-replication identity uses `ddof=0`. Standard error concerns an estimator's sampling variation, not the dispersion of individual observations. IID intervals and tests should not be applied automatically to dependent returns.

## 3. Univariate distributions and estimation

Use a frozen SciPy distribution's `pdf`, `logpdf`, `cdf`, `ppf`, `rvs`, `mean`, `var` and `stats`. `ppf` is the quantile function. [SciPy's statistical API](https://docs.scipy.org/doc/scipy/reference/stats.html) documents the underlying interfaces.

| Distribution | Frozen distribution | Parameter meaning |
| --- | --- | --- |
| Gaussian | `stats.norm(loc=mu, scale=sigma)` | `sigma > 0`, standard deviation |
| Student-t | `stats.t(df=nu, loc=mu, scale=s)` | `nu > 0`, `s > 0`; mean exists for `nu > 1`, variance `s² nu/(nu−2)` for `nu > 2` |
| Laplace | `stats.laplace(loc=mu, scale=b)` | `b > 0`; variance `2b²` |
| Lognormal | `stats.lognorm(s=sigma, loc=0, scale=np.exp(mu))` | `mu`, `sigma > 0` are the mean and SD of log(X), support X > 0 |
| Skew-normal | `stats.skewnorm(a=alpha, loc=xi, scale=omega)` | `omega > 0`; `xi` is location, not generally the mean; alpha=0 gives Gaussian |

```python
from finstats.fitting import fit_distribution

families = {
    "Gaussian": stats.norm(loc=0., scale=.02),
    "Student-t": stats.t(df=6, loc=0., scale=.02),
    "Laplace": stats.laplace(loc=0., scale=.02),
    "Lognormal": stats.lognorm(s=.2, loc=0, scale=np.exp(0.)),
    "Skew-normal": stats.skewnorm(a=3., loc=0., scale=.02),
}
for name, distribution in families.items():
    sample = distribution.rvs(size=300, random_state=123)
    generator = {"Gaussian": stats.norm, "Student-t": stats.t,
                 "Laplace": stats.laplace, "Lognormal": stats.lognorm,
                 "Skew-normal": stats.skewnorm}[name]
    result = fit_distribution(sample, generator)
    quantiles = result["distribution"].ppf([.05, .5, .95])
```

`fit_distribution(data, distribution)` accepts one of the five SciPy generators or the names `"norm"`, `"t"`, `"laplace"`, `"lognorm"`, `"skewnorm"`. It returns `params`, a frozen `distribution`, `log_likelihood`, `aic`, `bic`. Parameter labels are `mu,sigma` for Gaussian/lognormal; `mu,scale` for Laplace; `mu,scale,df` for t; `mu,scale,alpha` for skew-normal. Lognormal fitting fixes `loc=0`. The helper preserves the previous notebook fitting calls; it does not implement a new optimizer or report coefficient standard errors.

AIC is `-2*log_likelihood + 2*k`; BIC is `-2*log_likelihood + np.log(n)*k`. Here k includes fitted shape, location and scale parameters, with the fixed lognormal location excluded. Compare models on the same observations, likelihood definition and units. Numerical t/skew-normal fits can reach local optima; these summaries inherit SciPy's fitting behavior. See [Notebook 01](notebooks/01_univariate_distributions.ipynb) for moments and [05](notebooks/05_case_study_financial_distribution.ipynb) for model selection and out-of-sample comparison.

## 4. Multivariate distributions

[Notebook 02](notebooks/02_multivariate_distributions.ipynb) explains joint, marginal and conditional distributions and linear transformations. [Notebook 05](notebooks/05_case_study_financial_distribution.ipynb) fits joint financial models.

### Gaussian and Student-t

Gaussian location MLE is the column mean and covariance MLE is the centered cross-product divided by n. SciPy evaluates and samples the joint distribution. The custom Student-t estimator supplies the fit that SciPy's multivariate-t distribution interface does not supply.

```python
from finstats.multivariate_fit import multivariate_t_mle

rng = np.random.default_rng(123)
observations = stats.multivariate_t(loc=[0., 0.], shape=[[1., .2], [.2, .5]], df=6).rvs(
    size=200, random_state=rng)
mu_hat = observations.mean(axis=0)
cov_hat = np.cov(observations, rowvar=False, ddof=0)
gaussian_joint = stats.multivariate_normal(mean=mu_hat, cov=cov_hat)
t_fit = multivariate_t_mle(observations)
t_joint = stats.multivariate_t(loc=t_fit["location"], shape=t_fit["scale"], df=t_fit["df"])
t_covariance = t_fit["scale"] * t_fit["df"] / (t_fit["df"]-2)
```

`multivariate_t_mle(data, df_starts=(4.,10.,30.))` requires k≥2 and n>k, finite observations and nonsingular sample covariance. It returns `location`, `scale`, `df`, `log_likelihood`, `starts`. Cholesky parameterization enforces positive definiteness; multiple L-BFGS-B starts and standardization improve fitting. The finite-covariance fit constrains df to [2.01,200]; hitting the upper bound approximates the Gaussian limit. Inspect optimization diagnostics. The covariance is `df/(df-2)*scale`, not `scale` itself.

A marginal t retains the joint df and the corresponding location and scale entry. It need not match a separately fitted univariate t, because joint estimation optimizes the joint likelihood. Joint-model expectations are precisely the expectations of its marginals when these exist. A Gaussian marginal similarly uses its matching mean and covariance entry.

### Symmetric Laplace and skew-normal

| Function | Input / result |
| --- | --- |
| `bivariate_symmetric_laplace_pdf(x, location, covariance)` | Points `(...,2)`, location `(2,)`, SPD actual covariance `(2,2)`; density shape `(...)` |
| `multivariate_skew_normal_logpdf(x, location, scale, shape)` | Points `(...,k)`, location/shape `(k,)`, SPD scale `(k,k)`; log density |
| `multivariate_skew_normal_moments(location, scale, shape)` | Tuple of population mean and actual covariance |
| `multivariate_skew_normal_rvs(location, scale, shape, size, random_state)` | Samples `(size,k)` from the same parameterization |
| `multivariate_skew_normal_mle(data)` | Dict `location`, `scale`, `shape`, `log_likelihood`, `starts`, `shape_bound_hit` |

```python
from finstats.multivariate import (
    bivariate_symmetric_laplace_pdf, multivariate_skew_normal_logpdf,
    multivariate_skew_normal_moments, multivariate_skew_normal_rvs,
    multivariate_skew_normal_mle,
)

covariance = np.array([[1., .2], [.2, .5]])
# Laplace mixture: W~Exp(1), Z~N(0,covariance), independent.
w = rng.exponential(size=300)
laplace_sample = np.sqrt(w[:, None]) * rng.multivariate_normal([0., 0.], covariance, 300)
laplace_density = bivariate_symmetric_laplace_pdf([[1., 0.]], [0., 0.], covariance)
skew_sample = multivariate_skew_normal_rvs([0., 0.], covariance, [2., -1.], 200, 123)
skew_mean, skew_covariance = multivariate_skew_normal_moments([0., 0.], covariance, [2., -1.])
skew_log_density = multivariate_skew_normal_logpdf(skew_sample, [0., 0.], covariance, [2., -1.])
skew_fit = multivariate_skew_normal_mle(skew_sample)
```

The symmetric bivariate Laplace density has an integrable infinite density at its center. No joint Laplace estimator is provided; the implementation is a density, with mixture sampling shown explicitly. Its marginal Laplace scale is `sqrt(covariance[j,j]/2)`. This is one specified multivariate Laplace construction, not a universal extension of independent Laplace marginals.

Skew-normal `scale` describes the underlying Gaussian kernel, not the distribution's actual covariance. The estimator uses multiple starts, shape bounds [-30,30] and bounded standardized Cholesky diagonals; failed starts and shape-bound hits must be inspected. Shape zero recovers multivariate Gaussian. Marginal shape is not generally obtained by simply selecting one joint shape component.

There is no custom multivariate lognormal API. For that construction, sample a multivariate Gaussian and exponentiate componentwise; estimate Gaussian parameters from strictly positive log observations. This does not make lognormal suitable for signed returns.

Linear-combination moments use `intercept + weights @ means` and `weights @ covariance @ weights`. For Gaussian/t families, the matching joint distribution determines the transformed location and covariance/scale. Multivariate t transformations preserve the joint df. Empirical paired observations must remain paired.

## 5. Nonparametric methods and QQ comparisons

[Notebook 04](notebooks/04_nonparametric_methods.ipynb) shows empirical distributions, KDE and model-dependent QQ comparisons.

```python
from finstats.empirical import empirical_cdf, empirical_quantile, qq_data

ecdf = empirical_cdf(x)
probabilities = ecdf(np.array([-.01, 0., .01]))
q95 = empirical_quantile(x, .95)
# Equivalently: np.quantile(x, .95, method="inverted_cdf").
kde = stats.gaussian_kde(x)
density = kde(np.linspace(x.min(), x.max(), 100))
normal_fit = fit_distribution(x, stats.norm)
theoretical, ordered = qq_data(x, normal_fit["distribution"])
```

`empirical_cdf(x)` returns a callable using right-inclusive `X≤z` counts. `empirical_quantile(x,q)` returns the inverse ECDF, with q∈[0,1] and endpoints equal to sample extrema. It does not use NumPy's default interpolated quantile. `qq_data(x, distribution=stats.norm)` returns theoretical and ordered sample quantiles using positions `(i-.5)/n`. Pass a frozen fitted distribution to include its parameters. SciPy `probplot` uses different plotting positions, so it is not an interchangeable replacement for these notebook figures.

SciPy KDE's default bandwidth uses Scott's rule. `bw_method` is a dimensionless covariance factor/callback, not an absolute bandwidth in return units. A KDE needs adequate nonconstant observations. It is a smoothed density estimate, not the empirical distribution itself.

## 6. Linear time-series models, simulations and diagnostics

Use [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) for AR(p) `(p,0,0)`, MA(q) `(0,0,q)`, ARMA(p,q) `(p,0,q)` and ARIMA(p,i,q) `(p,i,q)`. The library calls differencing order `d`; the notebooks call it `i`. Choose trend/intercept treatment explicitly, especially after differencing. [Notebook 08](notebooks/08_linear_time_series_models.ipynb) covers identification, likelihood estimation, evaluation and forecasting.

```python
from finstats.simulation import simulate_ar1, simulate_ma, simulate_random_walk
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

series = simulate_ar1(.6, 1., 300, mu=2., rng=123)
ar_fit = ARIMA(series, order=(1,0,0), trend="c").fit()
mean_forecast = ar_fit.get_forecast(steps=10)
forecast_mean = np.asarray(mean_forecast.predicted_mean)
forecast_interval = np.asarray(mean_forecast.conf_int(alpha=.05))
automatic_fit = auto_arima(series, d=0, seasonal=False, start_p=0, start_q=0,
                          max_p=2, max_q=2, information_criterion="bic",
                          error_action="raise", suppress_warnings=True)
```

`auto_arima` supplies an additional candidate search using the chosen criterion; it does not prove the DGP or replace residual diagnostics. See [its documentation](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html). Under Gaussian innovations conditional AR regression can coincide with OLS; exact likelihood initialization, MA terms and joint volatility estimation need not have this equivalence.

| `finstats.linear` function | Meaning |
| --- | --- |
| `ar1_unconditional_variance(phi,sigma2)` | Innovation **variance** divided by `1-phi²`, requiring `abs(phi)<1` |
| `ar1_acf(phi,max_lag)` | Population ACF including lag zero |
| `ar1_forecast(last_value,phi,mu,horizon)` | Known-parameter conditional mean array |
| `ar_roots(phi_coefficients)`, `is_stationary_ar(...)` | Roots of `1−phi1 z−…`; stationary if all outside unit circle |
| `ma_roots(theta)`, `is_invertible_ma(...)` | Roots of `1+theta1 z+…`; invertible if all outside unit circle |
| `arma_roots(ar_coefficients,ma_coefficients)` | Tuple `(ar_roots,ma_roots)` |

Finite MA processes are stationary with finite-variance innovations regardless of invertibility; those are different conditions. Differencing is `np.diff(series,n=i)` and loses i observations.

### Reusable simulations

All simulators accept integer seeds or a NumPy generator via `rng`. Their exact initialization and RNG draw order are preserved from the notebooks; sampling length is not a burn-in argument. Check [source docstrings](src/finstats/simulation.py) for initialization.

| Function | Result |
| --- | --- |
| `simulate_ar1(phi,sigma,n,mu=0.,rng=None)` | n observations; `sigma` is innovation SD; stationary initial draw |
| `simulate_ma(theta,sigma,n,mu=0.,rng=None)` | n observations; presample innovations generated for lag terms |
| `simulate_random_walk(n,sigma=1.,initial=0.,rng=None)` | n+1 levels, including initial level |
| `simulate_arch1(omega,alpha,n,rng=None)` | `(innovations, conditional_variances)` |
| `simulate_garch11(omega,alpha,beta,n,rng=None,df=None)` | Same tuple; optional unit-variance t innovations require df>2 |
| `simulate_ar1_garch11(phi,omega,alpha,beta,n,mu=0.,rng=None,df=None)` | `(observations, innovations, conditional_variances)`; `df` is keyword-only |

```python
from finstats.simulation import simulate_ar1_garch11, simulate_arch1, simulate_garch11

arch_values, arch_variances = simulate_arch1(.1, .4, 300, rng=123)
garch_values, garch_variances = simulate_garch11(.05, .08, .88, 300, rng=123, df=6)
```

The combined simulator uses a 1,000-step variance burn-in and a further 1,000-step combined-process burn-in. Use it when testing aligned mean and variance recursions.

### Diagnostics

```python
from finstats.diagnostics import acf, ljung_box, mcleod_li, standardized_residuals
from statsmodels.tsa.stattools import pacf, adfuller, kpss
from statsmodels.stats.diagnostic import het_arch

residuals = np.asarray(ar_fit.resid)
correlations = acf(residuals, max_lag=10, adjusted=True)
partial_correlations, pacf_intervals = pacf(series, nlags=10, alpha=.05)
whiteness = ljung_box(residuals, lags=10, model_df=1)
squared_dependence = mcleod_li(residuals, lags=10)
arch_lm = het_arch(residuals, nlags=10, ddof=1)
adf_result = adfuller(series)
kpss_result = kpss(series, regression="c", nlags="auto")
```

`sample_autocovariance(x,lag,adjusted=True)` uses denominator `n−lag`; `False` uses n. `sample_autocorrelation` divides by lag-zero covariance. `acf(x,max_lag,adjusted=True)` returns lag 0 through max_lag and uses the statsmodels backend with explicit denominator convention. The approximate white-noise reference bounds used in the notebooks are `±1.96/sqrt(n)`; statsmodels' returned confidence intervals can follow a different assumption and are not automatically those plotted bounds.

`ljung_box(x,lags,model_df=0)` delegates to statsmodels with denominator-n ACF and returns `statistic`, `p_value`, `df`, `lags`, `nobs`. Degrees of freedom are `lags−model_df`; choose a valid model adjustment. `mcleod_li(residuals,lags)` applies Ljung–Box to squared residuals, with zero model adjustment. It assesses squared dependence under its reference assumptions, including adequate moments. ARCH LM is a separate regression-based test. Heavy tails can complicate both interpretations.

`standardized_residuals(residuals,conditional_volatility)` divides by **SD**, not variance, and checks aligned finite inputs and positive volatility. For combined models, examine both standardized residuals and their squares. ADF has a unit-root null; KPSS has a stationarity null determined by the regression specification. See [Notebook 07](notebooks/07_stationarity_and_nonstationarity.ipynb).

## 7. Conditional volatility and joint models

Use [arch's standard estimation APIs](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html) for stand-alone ARCH/GARCH. [Notebook 09](notebooks/09_conditional_volatility_models.ipynb) explains conditions, estimation and risk forecasts.

```python
from arch import arch_model

volatility_fit = arch_model(garch_values, mean="Zero", vol="GARCH", p=1, q=1,
                            dist="StudentsT", rescale=False).fit(disp="off")
volatility_forecast = volatility_fit.forecast(horizon=5, reindex=False)
innovation_variance_forecast = volatility_forecast.residual_variance.iloc[-1].to_numpy()
```

The notebooks/custom joint API order GARCH(p,q) as **variance lags, squared-innovation lags**. arch's `p` denotes squared-innovation lags and `q` variance lags: swap them when translating asymmetric orders. ARCH(q) in notebook notation uses arch `vol="ARCH", p=q`. arch's fitted Student-t errors are standardized to unit variance. Its `conditional_volatility` is SD; `residual_variance` forecasts innovations while `variance` can include mean-model propagation. Convergence flags and coefficient uncertainty require inspection; repeated starts can reduce local-optimum risk but do not guarantee a global optimum.

### Known-parameter recursions

`finstats.volatility` preserves explicit recursions for illustrating and checking estimates:

- `arch1_unconditional_variance(omega,alpha)` and `garch11_unconditional_variance(omega,alpha,beta)` require finite second moments (respectively alpha<1 and alpha+beta<1).
- `garch11_variance_path(residuals,omega,alpha,beta,initial_variance=None)` returns aligned innovation variances; default initialization is the unconditional variance. Explicit positive initialization also supports a finite nonstationary path.
- `garch11_forecast(last_residual,last_variance,omega,alpha,beta,horizon)` returns expected future innovation variances. The first step uses the last observed residual square, then expected future squares equal their forecast variances.

### Joint ARMA–GARCH estimator

`fit_arma_garch(values,ar_start,ma_start,mu_start,distribution="normal",*,hold_back=2,n_starts=3,seed=111,garch_order=(1,1),start_fit=None,compute_inference=True)` fits mean and volatility parameters **jointly** by conditional likelihood. The AR/MA start vector lengths fix their model orders. `distribution` is `"normal"` or `"t"`. `garch_order` follows notebook notation, not arch argument names. `start_fit` adds a warm start; skipping inference is useful during repeated refits.

```python
from finstats.joint import fit_arma_garch

joint_fit = fit_arma_garch(garch_values[:290], [.1], [.1], 0.,
                           distribution="t", n_starts=2, seed=123)
fitted_means, fitted_innovations, fitted_variances = joint_fit.filter()
means, innovation_variances, predictive_paths = joint_fit.forecast(
    horizon=5, simulations=1000, seed=123)
```

`JointARMAGARCHFit` exposes `params`, `nobs`, `aic`, `bic`, `loglikelihood`, `starts`, `inference`, `volatility_order` and the fitted parameters/history. `.filter(values=None)` returns conditional means, innovations and variances using fixed parameters and original initialization. `.forecast(...)` returns `(means, expected_innovation_variances, simulated_return_paths)` with paths `(simulations,horizon)`. At multiple steps innovation variance is not generally total return prediction-error variance; simulated paths include propagated mean uncertainty. Parameter-estimation uncertainty is not simulated.

`filter_arma_garch(values,mu,ar,ma,omega,alpha,beta,initial_variance)` offers the same causal recursion for supplied parameters. Scalars or lag vectors specify alpha/beta. Initial variance and hold-back treatment matter for likelihood comparisons. Parameter transformations enforce roots/variance constraints; the t density is standardized to unit variance. Inspect all start diagnostics and boundary flags in `inference`; robust numerical uncertainty near boundaries may be unavailable or only nominal. Multiple starts do not guarantee a global maximum. The custom estimator is conditional and should not be expected to match an external package with another likelihood initialization exactly. [Notebooks 10](notebooks/10_joint_mean_volatility_models.ipynb) and [11](notebooks/11_case_study_financial_time_series.ipynb) show its role.

### Expanding-window forecasting

```python
from finstats.rolling import expanding_arma_garch_forecasts

rolling = expanding_arma_garch_forecasts(garch_values, joint_fit, n_starts=2, seed=123)
# Each row forecasts its target_index after fitting only preceding observations.
```

`expanding_arma_garch_forecasts(values,initial_fit,*,n_starts=2,seed=111,cache_path=None,progress=None)` requires the initial fit's observations to match the complete series' training prefix exactly. It refits at every origin using fixed model orders, warm and interior starts. It returns a DataFrame with `target_index`, `mean`, `variance`, `df`, `VaR`, `ES`, `loglikelihood`, `persistence`, `converged_starts`, and serialized `parameters`. VaR/ES use 5% tail probability and loss sign conventions.

Optional CSV/JSON checkpoints record data/source/settings signatures and CSV checksums; valid incomplete records resume execution. Changing relevant inputs or estimator/rolling source invalidates the cache. `progress(done,total)` reports progress. Saved results do not replace the refitting algorithm. Notebook 11's 752 daily refits take substantially longer to regenerate than smaller examples.

## 8. Financial tail risk and evaluation

[Notebook 01](notebooks/01_univariate_distributions.ipynb) defines population risk; [04](notebooks/04_nonparametric_methods.ipynb) explains finite-sample estimates; [05](notebooks/05_case_study_financial_distribution.ipynb) and [11](notebooks/11_case_study_financial_time_series.ipynb) evaluate them out of sample.

`alpha` denotes **tail probability**, so alpha=.05 means the 95th loss quantile. All parametric helpers take **loss-distribution** location/scale. Gaussian helpers use loss SD. Student-t helpers use t scale and df: VaR needs df>0, ES requires df>1.

| `finstats.risk` function | Result |
| --- | --- |
| `historical_var(losses,alpha=.05)` | Inverse ECDF at 1−alpha |
| `historical_es(losses,alpha=.05)` | Mean of losses strictly above historical VaR |
| `gaussian_var(mu,sigma,alpha=.05)` | Gaussian loss quantile |
| `gaussian_es(mu,sigma,alpha=.05)` | Gaussian conditional upper-tail loss expectation |
| `student_t_var(location,scale,df,alpha=.05)` | Location/scale t loss quantile |
| `student_t_es(location,scale,df,alpha=.05)` | t upper-tail expectation |

```python
from finstats.risk import historical_var, historical_es, student_t_var, student_t_es

training_returns = stats.t(df=6, loc=.001, scale=.02).rvs(size=300, random_state=123)
training_losses = -training_returns
historical_threshold = historical_var(training_losses, .05)
historical_tail_mean = historical_es(training_losses, .05)
return_fit = fit_distribution(training_returns, stats.t)["params"]
t_threshold = student_t_var(-return_fit["mu"], return_fit["scale"], return_fit["df"], .05)
t_tail_mean = student_t_es(-return_fit["mu"], return_fit["scale"], return_fit["df"], .05)

# Conditional unit-variance t innovation model: return SD -> ordinary t scale.
conditional_scale = np.sqrt(fitted_variances[-1]) * np.sqrt((joint_fit.df-2)/joint_fit.df)
conditional_var = student_t_var(-fitted_means[-1], conditional_scale, joint_fit.df, .05)
conditional_es = student_t_es(-fitted_means[-1], conditional_scale, joint_fit.df, .05)
```

Historical ES here is a **strict-exceedance average**, not integrated-quantile ES with fractional boundary mass. It rejects samples with no strict exceedances. The removed legacy ES wrapper used inclusive `≥VaR` averaging; this distinction remains tested and is not silently standardized away.

For another continuous loss model (Laplace, lognormal or skew-normal), use `loss_model.ppf(1-alpha)` and, if the tail mean is finite, `loss_model.expect(lb=threshold)/alpha`. A distribution fitted to returns must first be converted to losses; for continuous returns VaR is `-return_model.ppf(alpha)` and ES is `-return_model.expect(ub=return_model.ppf(alpha))/alpha`. Lognormal is supported as a positive loss model, not as a model for signed returns.

Out-of-sample evaluation keeps forecasts separate from realizations. ME is `np.mean(actual-forecast)`; RMSE is `np.sqrt(np.mean((actual-forecast)**2))`; MAE is `np.mean(np.abs(actual-forecast))`. VaR evaluation compares the observed exceedance proportion to alpha. ES severity comparison averages forecast ES and actual losses over the **same exceedance days**. Different model thresholds select different days, so that descriptive comparison is not a common-sample ranking or formal ES backtest. Retain dates and periods when comparing static versus rolling models.

## 9. Namespace migration and verification

Version 0.2 consolidates all reusable project code under `finstats`. There are no compatibility adapters or empty package scaffolds. The distribution package's installation name remains `finance-statistical-methods-library`.

| Previous path | Current usage |
| --- | --- |
| `finstats.timeseries.ar/ma/arma` | `finstats.linear` for roots/moments/known-parameter forecasts; `finstats.simulation` for simulators |
| `finstats.timeseries.volatility/composite/nonstationary` | `finstats.volatility`, `finstats.simulation`; NumPy differencing |
| `finstats.timeseries.diagnostics/joint/rolling` | `finstats.diagnostics/joint/rolling` |
| `finstats.nonparametric` | `finstats.empirical` |
| `finstats.multivariate_skew_normal` | `finstats.multivariate` |
| `finance_statistical_methods.distributions.univariate` | SciPy generators; `finstats.fitting.fit_distribution` for summaries |
| `finstats.descriptive/inference/distributions` | Standard calculations described above and in notebooks |
| Legacy return, risk and plotting wrappers | Explicit return calculations, `finstats.risk`, Matplotlib/SciPy |

Numerical tests retain custom estimator/recursion checks, causality, RNG reproducibility, parameter constraints, analytical identities, data checksums and risk/quantile conventions. Removed-wrapper validation tests are replaced by relevant mathematical and current-interface checks. See [tests](tests/README.md). The restructuring is also checked by executing all notebooks and comparing saved numerical baselines; notebook exposition, examples and empirical inputs remain unchanged.
