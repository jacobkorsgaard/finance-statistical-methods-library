# Notebook 09 — Empirical Financial Time-Series Case Study

The user confirmed retaining current numbering. This is the requested conceptual Notebook 07. It remains a capstone, with no substantial new theory. Reuse 04 distributional descriptions, 07 process/realization/stationarity/ergodicity concepts and 08 mean/variance models. Tonellato's financial workflow is the primary course reference.

## Required workflow

1. Data and empirical question: one S&P 500 price-index series, fixed local source/window and provenance.
2. Data construction: percentage-point log returns, unique sorted dates, no filling or look-ahead. Fixed training cutoff and holdout.
3. Exploratory distribution analysis: sample summaries and QQ/tail descriptions learned earlier.
4. Time-series dependence: training return ACF/PACF and squared-return ACF.
5. Stationarity assessment: training-only ADF with stated deterministic terms and lag selection, interpreted as evidence concerning a unit-root null rather than proof of stationarity or ergodicity.
6. Parsimonious conditional-mean model: existing small constant/AR(1)/AR(2) candidate set, AIC/BIC on the same observations.
7. Mean-model diagnostics: fitted residual dependence and Ljung–Box.
8. ARCH effects: squared residuals and McLeod–Li screen before imposing GARCH.
9. GARCH(1,1): jointly estimated mean/variance, units, persistence and finite-long-run-moment restrictions.
10. Innovation distribution: Gaussian versus standardized Student-t, same observations and model orders.
11. Final diagnostics: standardized residuals, squared standardized residuals, ACF and assumed-innovation quantiles. Omit nominal squared-portmanteau p-values when the fitted innovations lack a finite fourth moment.
12. Out-of-sample mean forecasts: frozen training parameters, sequential one-step forecasts updated only after revealing each observation, simple zero/training-mean benchmarks, MSE/RMSE/MAE.
13. Out-of-sample volatility forecasts: conditional innovation variance evaluated against a noisy squared forecast-error proxy, fixed training-variance benchmark and units. Do not equate the proxy with observed latent volatility.
14. Final interpretation: report actual diagnostics and holdout results, including model limitations and failures.
15. Repository synthesis: explicitly connect 04, 07 and 08.

## Data and scope

Preserve the S&P 500 closing-price snapshot from 2017-02-01 through 2018-11-16. Training returns end 2018-11-02, with ten holdout observations from 2018-11-05. Provider differences prevent claims of identical slide estimates. A short holdout is illustrative, not general proof of forecast superiority.

[COURSE] Mean/variance workflow and financial interpretation. [EXTENSION] Existing fixed holdout, candidate restriction and noisy-variance-proxy evaluation. Add no new model classes, advanced theory, optimization or model race. Reuse all existing tested APIs. Preserve meaningful existing empirical code and simplify duplicate plots where possible. Execute offline in a fresh kernel.
