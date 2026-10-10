# Python API guide

Use this guide to find and reuse the Python tools behind the notebooks. Standard methods come from established libraries; `finstats` provides selected estimators, simulations and helpers. The notebooks explain the mathematics, while source docstrings give full arguments and validation rules.

Install following the [README](README.md). The three examples below can be run in order.

## Finding the right tool

| Task | Tool | Notebook |
| --- | --- | --- |
| Returns, moments and inference | NumPy/SciPy | [00](notebooks/00_financial_returns.ipynb), [03](notebooks/03_statistical_inference_and_mle.ipynb) |
| Univariate models | `scipy.stats`, `finstats.fitting` | [01](notebooks/01_univariate_distributions.ipynb) |
| Multivariate models | SciPy and custom estimators | [02](notebooks/02_multivariate_distributions.ipynb) |
| Empirical methods | `finstats.empirical`, SciPy KDE | [04](notebooks/04_nonparametric_methods.ipynb) |
| Distributional financial risk | `finstats.risk` | [05](notebooks/05_case_study_financial_distribution.ipynb) |
| Processes and stationarity | Simulations, statsmodels tests | [06](notebooks/06_stochastic_processes.ipynb), [07](notebooks/07_stationarity_and_nonstationarity.ipynb) |
| Linear time-series models | statsmodels, pmdarima | [08](notebooks/08_linear_time_series_models.ipynb) |
| Conditional volatility | arch, `finstats.volatility` | [09](notebooks/09_conditional_volatility_models.ipynb) |
| Combined models and rolling forecasts | `finstats.joint`, `finstats.rolling` | [10](notebooks/10_joint_mean_volatility_models.ipynb), [11](notebooks/11_case_study_financial_time_series.ipynb) |

**Conventions:** Keep units consistent: decimal −0.02 means −2%; applications use percentage points. Simple-return losses are `-returns`. Sample variance uses `ddof=1`, Gaussian MLE variance `ddof=0`. Notebook skewness/kurtosis use `bias=True`, with `fisher=False` for raw kurtosis. Multivariate samples have observations in rows, `(n,k)`.

## Standard calculations

| Calculation | Interface |
| --- | --- |
| Simple / log returns | `prices[1:]/prices[:-1]-1` / `np.log(prices[1:]/prices[:-1])` |
| Mean / sample SD | `np.mean(x)` / `np.std(x,ddof=1)` |
| Skewness / raw kurtosis | `stats.skew(x,bias=True)` / `stats.kurtosis(x,fisher=False,bias=True)` |
| Covariance / correlation | `np.cov(X,rowvar=False,ddof=1)` / `np.corrcoef(X,rowvar=False)` |
| Linear-combination mean / variance | `weights @ means` / `weights @ covariance @ weights` |
| Mean standard error / test | `stats.sem(x)` / `stats.ttest_1samp(x,popmean=mu0)` |
| Gaussian mean / variance inference | `stats.t` / `stats.chi2`; worked examples in Notebook 03 |

Dividend-adjusted prices already account for dividends. IID inference requires additional care for dependent observations.

## Distributions and fitting

Import `from scipy import stats`. SciPy distributions provide `pdf`, `logpdf`, `cdf`, `ppf` (quantiles), `rvs` (sampling), `mean` and `var`.

| Model | Distribution | Parameter convention |
| --- | --- | --- |
| Gaussian | `stats.norm(loc=mu,scale=sigma)` | sigma is SD |
| Student-t | `stats.t(df=nu,loc=mu,scale=s)` | SD=`s*sqrt(nu/(nu-2))` for nu>2 |
| Laplace | `stats.laplace(loc=mu,scale=b)` | SD=`sqrt(2)*b` |
| Lognormal | `stats.lognorm(s=sigma,loc=0,scale=np.exp(mu))` | Parameters describe log(X); X>0 |
| Skew-normal | `stats.skewnorm(a=alpha,loc=xi,scale=omega)` | xi is location, not generally the mean |

[`fit_distribution(data,distribution)`](src/finstats/fitting.py) fits these families through SciPy. It accepts generators or `"norm"`, `"t"`, `"laplace"`, `"lognorm"`, `"skewnorm"`, returning `params`, a fitted `distribution`, `log_likelihood`, `aic`, `bic`. Lognormal fitting fixes `loc=0`; numerical fits can reach local optima.

```python
import numpy as np
from scipy import stats
from finstats.fitting import fit_distribution
from finstats.risk import historical_var, historical_es, student_t_var, student_t_es

returns = stats.t(df=6, loc=.001, scale=.02).rvs(size=300, random_state=123)
fit = fit_distribution(returns, "t")
quantiles = fit["distribution"].ppf([.05, .5, .95])
p = fit["params"]
empirical_risk = historical_var(-returns), historical_es(-returns)
fitted_risk = (student_t_var(-p["mu"], p["scale"], p["df"]),
               student_t_es(-p["mu"], p["scale"], p["df"]))
```

Parameter labels: `mu,sigma` for Gaussian/lognormal; `mu,scale` for Laplace; `mu,scale,df` for t; `mu,scale,alpha` for skew-normal. Compare AIC/BIC on the same observations and likelihood basis.

### Multivariate models

| Model | Available interface |
| --- | --- |
| Gaussian | `stats.multivariate_normal(mean=mu,cov=covariance)`; fit column means and covariance with `ddof=0` |
| Student-t | `stats.multivariate_t(loc=mu,shape=scale,df=nu)`; custom `multivariate_t_mle(X)` |
| Symmetric bivariate Laplace | `bivariate_symmetric_laplace_pdf(x,location,covariance)`; no joint estimator |
| Skew-normal | `multivariate_skew_normal_logpdf`, `_moments`, `_rvs`, `_mle` (full names share the prefix) |

[`multivariate_t_mle`](src/finstats/multivariate_fit.py) returns location, scale, df, likelihood and optimizer-start diagnostics. It needs nonsingular samples with n>k≥2 and constrains df to [2.01,200]. Actual covariance is `scale*df/(df-2)`.

[`finstats.multivariate`](src/finstats/multivariate.py) documents the Laplace/skew-normal interfaces. Laplace's matrix is actual covariance; its Gaussian–exponential mixture has an integrable infinite density at the center. Skew-normal scale describes the Gaussian kernel; use its moments function for actual mean/covariance and inspect MLE start/boundary diagnostics.

Joint-model marginals need not match separate univariate fits. Multivariate lognormal has no custom API; it can be constructed by exponentiating Gaussian samples. Lognormal is unsuitable for signed returns.

## Empirical methods

| Interface | Purpose |
| --- | --- |
| `empirical_cdf(x)` | Callable ECDF counting observations ≤ evaluation points |
| `empirical_quantile(x,q)` | Inverse ECDF; NumPy equivalent uses `method="inverted_cdf"` |
| `qq_data(x,distribution)` | Theoretical/ordered sample quantiles at `(i-.5)/n` |
| `stats.gaussian_kde(x)` | Smoothed density; default Scott bandwidth |

See [`finstats.empirical`](src/finstats/empirical.py). Pass a fitted distribution for QQ comparisons. NumPy's default quantiles and SciPy `probplot` positions differ from these conventions.

## Time-series models

| Task | Interface |
| --- | --- |
| AR, MA, ARMA, ARIMA | `statsmodels.tsa.arima.model.ARIMA(x,order=(p,d,q)).fit()` |
| Automatic candidate search | `pmdarima.auto_arima(x,seasonal=False,information_criterion="bic")` |
| Mean forecasts / differencing | Fitted `.get_forecast(steps=h)` / `np.diff(x,n=d)` |
| Stationarity / identification | statsmodels `adfuller`, `kpss`, `acf`, `pacf` |
| ARCH/GARCH | `arch.arch_model(...).fit()` |
| ARCH LM test | `statsmodels.stats.diagnostic.het_arch` |

AR, MA and ARMA use orders `(p,0,0)`, `(0,0,q)` and `(p,0,q)`. Notebook differencing order i is library d. Set trends explicitly; automatic searches complement residual diagnostics.

**GARCH ordering:** Notebooks/custom joint models use `(variance lags,squared-innovation lags)`. arch uses p for squared-innovation lags and q for variance lags. Swap asymmetric orders. Its `conditional_volatility` is SD; forecast `residual_variance` is innovation variance.

### Custom helpers

| Module | Functions / purpose |
| --- | --- |
| [simulation](src/finstats/simulation.py) | `simulate_ar1`, `simulate_ma`, `simulate_random_walk`, `simulate_arch1`, `simulate_garch11`, `simulate_ar1_garch11` |
| [linear](src/finstats/linear.py) | AR(1) moments/forecasts and AR/MA roots, stationarity and invertibility checks |
| [volatility](src/finstats/volatility.py) | ARCH/GARCH unconditional variance, GARCH(1,1) variance paths and forecasts |
| [diagnostics](src/finstats/diagnostics.py) | Sample autocovariance/autocorrelation, `acf`, `ljung_box`, `mcleod_li`, `standardized_residuals` |

Simulators accept seeds/generators via `rng`; volatility outputs are variances. Random walks include the initial level. Optional t innovations have unit variance and require df>2. Source docstrings specify initialization and return order.

Custom ACF defaults to denominator n−lag; Ljung–Box uses denominator n and df=`lags-model_df`. McLeod–Li applies it to squared residuals with zero deduction and approximate calibration. Standardize residuals by **SD**, then inspect both residuals and their squares.

### Joint estimation and rolling forecasts

```python
from finstats.simulation import simulate_garch11
from finstats.joint import fit_arma_garch
from finstats.rolling import expanding_arma_garch_forecasts

a, h = simulate_garch11(.05, .08, .88, 300, rng=123, df=6)
joint_fit = fit_arma_garch(a[:290], [.1], [.1], 0., distribution="t", n_starts=2)
means, residuals, variances = joint_fit.filter()
future_means, future_innovation_variances, paths = joint_fit.forecast(
    horizon=5, simulations=1000, seed=123)
rolling = expanding_arma_garch_forecasts(a, joint_fit, n_starts=2)
```

[`fit_arma_garch`](src/finstats/joint.py) takes observations, starting AR/MA vectors and starting mean; vector lengths determine orders. Options include `distribution`, `garch_order`, `hold_back`, `n_starts`, `seed`, `start_fit`, `compute_inference`. Results expose parameters, likelihood, AIC/BIC, starts and inference diagnostics. Inspect convergence and boundaries; multiple starts do not guarantee a global optimum. Multi-step innovation variance differs from total return forecast-error variance. Simulated paths exclude parameter-estimation uncertainty.

[`expanding_arma_garch_forecasts`](src/finstats/rolling.py) requires the initial fit to match the training prefix, then refits before every target. It returns mean, variance, df, 5% VaR/ES and estimation diagnostics. Optional `cache_path` stores validated checkpoints; `progress(done,total)` reports progress. Changed inputs/settings/estimator source invalidate cached records.

## Risk and evaluation

[`finstats.risk`](src/finstats/risk.py) provides `historical_var/es(losses,alpha=.05)`, `gaussian_var/es(mu,sigma,alpha=.05)` and `student_t_var/es(location,scale,df,alpha=.05)` (each pair denotes two functions). Parametric inputs describe **losses**; alpha=.05 means the 95th loss quantile. Gaussian sigma is SD; t scale is not SD.

Historical ES averages losses **strictly above** inverse-ECDF VaR and needs an exceedance. It differs from integrated-quantile ES with fractional boundary mass. Student-t ES requires df>1. Conditional unit-variance t models need scale=`sd*sqrt((df-2)/df)` and loss location equal to minus the return mean.

For other continuous return models, VaR is `-distribution.ppf(alpha)` and ES is `-distribution.expect(ub=distribution.ppf(alpha))/alpha`, when finite. For continuous loss models, use the upper quantile/tail instead.

```python
from finstats.empirical import qq_data

theoretical, observed = qq_data(returns, fit["distribution"])
actual = a[290:]
forecast = rolling["mean"].to_numpy()
me = np.mean(actual - forecast)
rmse = np.sqrt(np.mean((actual - forecast)**2))
mae = np.mean(np.abs(actual - forecast))
exceedance_frequency = np.mean(-actual > rolling["VaR"].to_numpy())
```

Compare VaR exceedance frequency with alpha. Compare actual tail losses and predicted ES on the same exceedance days; different thresholds select different days, so this alone is not an overall ranking or formal ES backtest.

Further details: [SciPy](https://docs.scipy.org/doc/scipy/reference/stats.html), [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html), [pmdarima](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html), [arch](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html), and [project tests](tests/README.md).
