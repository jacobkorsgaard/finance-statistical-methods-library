# Finance Statistical Methods Library

Probability and statistical methods for finance, developed through mathematical explanations, reproducible simulations and empirical applications. The notebooks are supported by tested Python functions.

**Early draft:** all nine notebooks execute, and the full suite contains 383 passing tests. The mathematical exposition, course fidelity and cross-notebook consistency remain under review.

## Notebooks

| Notebook | Topic |
| --- | --- |
| [01](notebooks/01_random_variables_and_distributions.ipynb) | Continuous univariate distributions and theoretical/empirical moments |
| [02](notebooks/02_multivariate_random_variables.ipynb) | Multivariate random variables through bivariate illustrations |
| [03](notebooks/03_statistical_inference_and_mle.ipynb) | Frequentist inference and MLE across parametric models |
| [04](notebooks/04_nonparametric_distributions.ipynb) | Empirical distributions, density estimates and QQ plots |
| [05](notebooks/05_returns_losses_and_financial_risk.ipynb) | Returns, losses and financial risk |
| [06](notebooks/06_empirical_distributions_and_risk.ipynb) | Empirical distributions and risk in US/German markets |
| [07](notebooks/07_stochastic_processes_and_time_series_foundations.ipynb) | Stochastic processes and time-series foundations |
| [08](notebooks/08_linear_time_series_and_conditional_volatility.ipynb) | Linear time series and conditional volatility |
| [09](notebooks/09_empirical_financial_time_series_case_study.ipynb) | Empirical financial time-series case study |

Read them in order. Notebook 02 introduces random vectors, marginal and conditional distributions, then population and sample covariance and correlation. Section 5 illustrates linear transformations of Gaussian, Student-t and Laplace distributions. Section 6 compares bivariate density surfaces for Gaussian, Student-t, symmetric Laplace and skew-normal distributions. Portfolio applications remain in the later finance notebook.

Notebooks 01–05 and 07 use simulated observations. Notebook 06 uses US/German market and FX snapshots, 08 includes the Mishkin inflation example, and 09 applies the time-series workflow to S&P 500 returns.

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
python -m nbconvert --to notebook --execute --inplace notebooks/01_random_variables_and_distributions.ipynb
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
