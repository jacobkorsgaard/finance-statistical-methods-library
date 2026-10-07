# Finance Statistical Methods Library

Probability and statistical methods for finance, developed through mathematical explanations, reproducible simulations and empirical applications. The notebooks are supported by tested Python functions.

**Early draft:** all nine notebooks execute, and the library includes tested statistical functions. The mathematical exposition, course fidelity and cross-notebook consistency remain under review.

## Notebooks

| Notebook | Topic |
| --- | --- |
| [00](notebooks/00_financial_returns_as_distributions.ipynb) | From prices and returns to probability models and downside-risk questions |
| [01](notebooks/01_univariate_distributions.ipynb) | Univariate distributions, moments and population tail properties |
| [02](notebooks/02_multivariate_distributions.ipynb) | Multivariate random variables through bivariate illustrations |
| [03](notebooks/03_statistical_inference_and_mle.ipynb) | Estimator properties, frequentist inference, MLE and model comparison |
| [04](notebooks/04_nonparametric_distributions.ipynb) | Empirical distributions, QQ plots and finite-sample tail estimates |
| [05](notebooks/05_example_application_of_distributions.ipynb) | Example: Application of Distribution Models to Equity and Bond Returns |
| [06](notebooks/06_stochastic_processes_and_time_series_foundations.ipynb) | From temporal ordering to stochastic-process foundations |
| [07](notebooks/07_linear_time_series_and_conditional_volatility.ipynb) | Linear time series and conditional volatility |
| [08](notebooks/08_empirical_financial_time_series_case_study.ipynb) | Empirical financial time-series case study |

Start with Notebook 00: calculate returns, distinguish an uncertain return from its observed realizations, and introduce expected return, dispersion and downside loss. The same SPY observations appear as a realized history and a pooled histogram. A separate hypothetical loss illustration introduces VaR and ES without assuming the market distribution is known.

Notebooks 01–03 develop probability distributions and inference. Expectation and variance remain general distributional properties. Notebook 01 also derives population tail quantiles and expectations; Notebook 04 obtains their empirical counterparts and examines sparse-tail sampling variability. The former standalone returns/risk notebook has been distributed across 00, 01 and 04, preserving its derivations and simulations.

Notebook 05 applies these tools to USD equity exposure (SPY) and broad investment-grade bond exposure (AGG). Models are estimated and selected on 2017–2022, then evaluated against 2023–2025 observations without refitting. Separate fits and joint-model marginals are compared with the empirical baseline. No description performs best on every evaluated measure in this period. Conditional distributions and linear combinations remain deferred.

Notebook 06 returns to temporal ordering. Reordering identical observations preserves their empirical distribution and historical VaR/ES but changes their placement in time. The stochastic-process foundations lead to conditional-mean and conditional-variance models in 07, then their empirical application in 08. Out-of-sample discrepancies alone do not establish temporal dependence.

All observations come from reproducible simulations or fixed local snapshots. Notebook 07 retains the course inflation example, and 08 retains the existing S&P 500 capstone. No notebook requires execution-time downloads.

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

All empirical data are fixed local snapshots, so notebook execution requires no network access. [Data documentation](data/README.md) records sources, windows, units, currencies, preprocessing and checksums.

## Repository structure

| Location | Contents |
| --- | --- |
| `notebooks/` | Exposition, simulations and applications, with saved outputs |
| `src/` | Reusable statistical functions, including preserved compatible APIs |
| `tests/` | Analytical, numerical and data-integrity checks |
| `data/` | Documented reproducible data snapshots |

## Acknowledgement

Thank you to Professor Stefano Tonellato at Università Ca’ Foscari Venezia for teaching *Statistical Models and Methods for Finance* and providing the foundations that inspired this project. The Python implementation and additional explanations are my own work. Any errors or interpretations are my responsibility.

A repository license remains to be selected, as recorded in [LICENSE](LICENSE).
