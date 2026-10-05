# Notebook 08 — Linear Time Series and Conditional Volatility

The user confirmed retaining current numbering. This is the requested conceptual Notebook 06, following 07 foundations and preceding 09 capstone. Mean and variance models share one notebook but remain distinct statistical targets. Tonellato is the primary course source. No scope expansion or general optimization tutorial.

## Required main sections

1. Linear Stochastic Processes.
2. Non-stationarity, Differencing and ARIMA.
3. Estimation and Model Diagnostics.
4. Forecasting.
5. Conditional Heteroskedastic Processes.
6. Volatility Estimation, Diagnostics and Forecasting.

## Course exposition and implementation

- [COURSE] Linear filters, innovations and lagged dependence. Keep the course's Wold motivation brief, with the purely nondeterministic-component qualification.
- [COURSE/DERIVATION] AR(1) is central: centered equation, conditional mean under a zero-conditional-mean innovation assumption, unconditional mean/variance, |phi|<1 causal stationarity, infinite-MA representation, gamma_j and rho_j. Recursion intercept differs from unconditional mean. Retain the course phi=.7, n=300, innovation variance one and stationary Gaussian initialization. [SIMULATION] A single path/ACF comparison with phi=0,.7,-.7 and seed 510 demonstrates persistence.
- [COURSE] Generalize AR(p), MA(q), ARMA(p,q), AR/MA lag polynomials, stationarity and invertibility roots, population ACF/PACF patterns and identification intuition. Do not duplicate demonstrations for each model.
- [COURSE] Connect the random-walk foundations to unit roots, first/repeated differences, I(d), lag notation and ARIMA(p,d,q). Explain an ADF unit-root diagnostic sufficiently for the capstone, including deterministic terms and nonstandard reference distribution. No extensive test simulation.
- [COURSE] Disturbances versus residuals, likelihood connection to 03, exact Gaussian state-space versus conditional likelihood, nested AR/MA/ARMA/ARIMA software orders. AIC/BIC parameter counting and comparable samples. Residual ACF and Ljung–Box, with denominator and degrees-of-freedom conventions explicit.
- [COURSE/DERIVATION] Forecasting uses information set F_t and MMSE conditional expectation. AR(1) one/h-step formulas, forecast-error variance and Gaussian prediction intervals. Distinguish forecast MSE from estimator MSE. State holdout MSE/RMSE/MAE and units.
- [COURSE/EXTENSION] Preserve the Mishkin pai1 inflation series, course differencing motivation and small ARIMA candidate set. Preserve the fixed 24-month holdout and last-level benchmark as existing applied extensions. One ACF comparison and one forecast/interval figure, without a large horse race or repeated residual-path plots.
- [COURSE] Volatility clustering and levels versus squares; conditional versus unconditional variance. ARCH(1) a_t=sigma_t nu_t and sigma_t²=omega+alpha a_(t-1)², conditional mean zero, variance equation and omega/(1-alpha). Squared recursion and moment-dependent squared ACF.
- [COURSE/SIMULATION] Retain course ARCH omega=2, alpha=.8, n=300, seed 123 with Gaussian shocks. One four-panel figure shows levels, conditional SD and dependence in levels/squares. The squared population correlation is undefined without finite fourth moments, so this example's squared ACF is descriptive. Do not overlay alpha^j as a population target.
- [COURSE] Generalize ARCH(q), course GARCH(p,q) and practical GARCH(1,1), omega/alpha/beta roles, alpha+beta persistence and long-run variance omega/(1-alpha-beta), finite-variance stationarity under usual assumptions. Explain squared ARMA representation concisely.
- [COURSE] Course p counts variance lags, q shock lags. The arch package reverses these names. State the mapping explicitly, even though (1,1) is unchanged.
- [SIMULATION/EXTENSION] Demonstrate mature-package variance estimation on one known Gaussian GARCH population instead of repeating S&P 500 empirical analysis before the capstone. State seed, population parameters, simulation units and burn-in. Compare Gaussian and standardized Student-t innovations on the same sample.
- [COURSE] Standardized residuals and their squared ACF, Ljung–Box/McLeod–Li screening and finite-fourth-moment limitations. No exact post-estimation specification-test claim.
- [COURSE/DERIVATION] One/multistep conditional-variance forecasts and mean reversion. Reuse tested recursion and compare with package residual-variance forecasts. Conditional variance, unconditional variance and a realized squared observation remain different objects.

## Reuse and acceptance

Reuse existing finstats.timeseries modules, statsmodels and arch. No new wrappers or duplicate statistical functions. No EGARCH, GJR, stochastic volatility, HAR, realized-volatility models or multivariate GARCH. Execute in a fresh kernel and conclude with the coherent empirical workflow in 09.
