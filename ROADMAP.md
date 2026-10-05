# Library roadmap

## Purpose and current status

Develop an applied statistical library whose notebooks explain the mathematics before computation, distinguish population properties from finite observations, and connect results to financial interpretation. All nine notebooks are implemented and executable as an early draft. Review remains provisional.

Current user instructions govern revisions. The [notebook specifications](NOTEBOOK_SPECS/) define allowed scope, [AGENTS.md](AGENTS.md) consolidates development rules, and Tonellato's slides determine the primary course exposition. Corrections and deliberate departures are retained in [docs/SOURCE_NOTES.md](docs/SOURCE_NOTES.md).

## Curriculum

| Notebook | Scope and progression |
| --- | --- |
| 01 | Variables/realizations, brief discrete/PMF distinction, continuous PDF/CDF, quantiles, expectation/variance, moments, Gaussian/Student-t/Laplace/skew-normal families and finite samples. No standalone transformations or inference. |
| 02 | Vectors, joint/marginal/conditional distributions, covariance/correlation, covariance matrix, linear combinations/transformations, Gaussian vectors and financial mean–variance interpretation. No optimization. |
| 03 | Models, samples, estimators/estimates, bias, variance/SE, MSE, consistency, Gaussian sampling laws, intervals, tests, likelihood and Gaussian MLE. No Bernoulli example or model selection. |
| 04 | ECDF, empirical quantiles, KDE and QQ plots against known simulated populations. Sample-size and bandwidth effects, without market data or risk measures. |
| 05 | Simulated simple/log returns and losses, VaR/ES, population risk and finite-sample historical/parametric estimates. Preserve the course rare-tail Student-t example. |
| 06 | S&P 500/DAX price-index distributions, parametric fitting, AIC/BIC, tails and risk, followed by a common-currency fixed-weight multivariate application. No conditional time-series model. |
| 07 | Processes, finite-dimensional distributions, realized histories and sample moments, trends/random walks/drift, stationarity, ergodicity and noise assumptions. Foundations only. |
| 08 | Conditional means: AR(1), AR/MA/ARMA, integration/ARIMA, estimation/diagnostics and forecasting. Conditional variances: ARCH/GARCH, standardized diagnostics and variance forecasts. |
| 09 | One S&P 500 financial capstone, from return construction and stationarity assessment through parsimonious mean/variance fitting, diagnostics and holdout forecasts. No substantial new theory. |

Retain the nine-notebook numbering. Linear transformations belong in 02. Time-series foundations, modeling and the capstone remain separate in 07–09. Do not add advanced model classes merely for breadth.

## Implementation conventions

Use a definition, key result, useful illustration where needed and concise interpretation. Preserve course terminology, notation, parameterizations and R examples in Python. Use fixed seeds, purposeful figures and explicit assumptions, units and denominator conventions. Avoid duplicate plots, unnecessary derivations and thin wrappers.

Reusable computations live in `src/finstats/`. Preserve useful `src/finance_statistical_methods/` APIs and reuse their fitting/return functions where appropriate. No package-wide rewrite is required. The tested time-series modules support both conditional-mean and conditional-variance exposition.

Data remain fixed local snapshots with provenance and checksums. Tests cover reusable numerical functions, domain restrictions, analytical identities and agreement with mature libraries. Revised notebooks must execute in fresh kernels with saved outputs.

## Review sequence

1. Review 01–03 together for notation, mathematical depth, pacing and course fidelity.
2. Review the separation and applications in 04–06.
3. Review the foundations → modeling → capstone sequence in 07–09.
4. Resolve cross-notebook inconsistencies and choose a repository license before a final release.

Notebook 01 remains the provisional writing/figure benchmark. Review checkpoints do not authorize adding later content beyond the current specification.
