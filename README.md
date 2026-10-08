# Finance Statistical Methods Library

Probability and statistical methods for finance, developed through mathematical explanations, reproducible simulations and empirical applications. The notebooks are supported by tested Python functions.

**First complete draft:** Notebooks 00–11 form a connected progression from financial returns and probability distributions to inference, temporal models and empirical evaluation. Both applications are implemented, with saved notebook outputs and tested reusable Python functions. A final editorial and mathematical review remains before declaring a stable release.

## Notebooks

| Notebook | Topic |
| --- | --- |
| [00](notebooks/00_financial_returns_as_distributions.ipynb) | From prices and returns to probability models and downside-risk questions |
| [01](notebooks/01_univariate_distributions.ipynb) | Univariate distributions, moments and population tail properties |
| [02](notebooks/02_multivariate_distributions.ipynb) | Multivariate random variables through bivariate illustrations |
| [03](notebooks/03_statistical_inference_and_mle.ipynb) | Estimator properties, frequentist inference, MLE and model comparison |
| [04](notebooks/04_nonparametric_distributions.ipynb) | Empirical distributions, QQ plots and finite-sample tail estimates |
| [05](notebooks/05_example_application_of_distributions.ipynb) | Example: Application of Distribution Models to Equity and Bond Returns |
| [06](notebooks/06_stochastic_processes_and_time_series.ipynb) | Stochastic processes, realized time series and population/sample properties |
| [07](notebooks/07_stationary_and_nonstationary_processes.ipynb) | Stationarity, ergodicity, stationary noise and stochastic trends |
| [08](notebooks/08_linear_time_series_models.ipynb) | Linear time-series models: AR, MA, ARMA/ARIMA, estimation, diagnostics and forecasting |
| [09](notebooks/09_conditional_volatility_models.ipynb) | Conditional volatility: ARCH/GARCH, estimation, diagnostics and forecasting |
| [10](notebooks/10_combined_mean_and_volatility_models.ipynb) | Combining conditional means and volatility; joint AR–GARCH example |
| [11](notebooks/11_empirical_financial_time_series_case_study.ipynb) | Example: time-series models for S&P 500 returns |

Start with Notebook 00: calculate returns, distinguish an uncertain return from its observed realizations, and introduce expected return, dispersion and downside loss. The same SPY observations appear as a realized history and a pooled histogram. A separate hypothetical loss illustration introduces VaR and ES without assuming the market distribution is known.

Notebooks 01–03 develop probability distributions and inference. Expectation and variance remain general distributional properties. Notebook 01 also derives population tail quantiles and expectations; Notebook 04 obtains their empirical counterparts and examines sparse-tail sampling variability. These population and empirical perspectives prepare the financial applications without treating a finite sample as the population distribution.

Notebook 05 applies these tools to USD equity exposure (SPY) and broad investment-grade bond exposure (AGG). Models are estimated and selected on 2017–2022, then evaluated against 2023–2025 observations without refitting. Separate fits and joint-model marginals are compared with the empirical baseline. No description performs best on every evaluated measure in this period. Conditional distributions and linear combinations remain deferred.

Notebook 06 introduces stochastic processes, realized time series, and their population and sample properties. Notebook 07 establishes stationarity and ergodicity. Notebook 08 develops conditional-mean models, 09 develops conditional variance, and 10 combines them. Notebook 11 uses the same adjusted SPY returns and 2017–2022 / 2023–2025 split as Notebook 05. It compares four ARMA mean models, extends the selected mean to joint ARMA–GARCH estimation, and compares Notebook 05’s fixed univariate models with daily expanding-window parameter refits out of sample. Saved refit records are validated against data, estimation code and settings. Out-of-sample discrepancies alone do not establish temporal dependence. The capstone separates mean-model selection, conditional-volatility estimation and model evaluation; residual and squared-residual checks, McLeod–Li tests and QQ comparisons support the staged specification choices.

The empirical conclusion is deliberately measured: better in-sample fit and changing conditional volatility do not guarantee better future forecasts. In Notebook 11, rolling refits give slightly worse mean errors than the fixed univariate Student-t model. The rolling 5% VaR exceedance frequency is 7.05%, compared with 3.32% for the fixed Student-t and a 5% target. Conditional ES is close to average losses on its own exceedance days, but differing thresholds select different days, so this descriptive check does not establish an overall winner.

All observations come from reproducible simulations or fixed local snapshots. Notebook 08 uses simulated processes, and 11 uses the saved adjusted SPY snapshot. No notebook requires execution-time downloads.

## Setup and verification

Use Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Select this environment as the notebook kernel. Run notebooks from the repository root or `notebooks/`.

To execute one notebook and save its outputs:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_univariate_distributions.ipynb
```

The full test suite currently contains 458 passing tests. Notebook 11 has been executed top to bottom after the latest revisions.

Notebook 11 verifies and reuses the saved 752 expanding-window joint-model refits. Changing the estimation inputs or implementation invalidates these records and triggers recalculation, which takes longer than the other examples; allow a longer notebook execution timeout when regenerating them.

All empirical data are fixed local snapshots, so notebook execution requires no network access. [Data documentation](data/README.md) records sources, windows, units, currencies, preprocessing and checksums.

## Repository structure

| Location | Contents |
| --- | --- |
| `notebooks/` | Exposition, simulations and applications, with saved outputs |
| `src/` | Reusable statistical functions, including preserved compatible APIs |
| `tests/` | Analytical, numerical and data-integrity checks |
| `data/` | Documented reproducible data snapshots |
| `artifacts/notebook11/` | Verified rolling-refit records and calculation documentation |

## Acknowledgement

Thank you to Professor Stefano Tonellato at Università Ca’ Foscari Venezia for teaching *Statistical Models and Methods for Finance* and providing the foundations that inspired this project. The Python implementation and additional explanations are my own work. Any errors or interpretations are my responsibility.

A repository license remains to be selected, as recorded in [LICENSE](LICENSE).
