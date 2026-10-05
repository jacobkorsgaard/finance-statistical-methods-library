# Notebook 07 — Stochastic Processes and Time Series Foundations

The user requested a foundations → modeling → capstone structure and subsequently confirmed retaining the current nine-notebook numbering. This is therefore 07, corresponding to the requested conceptual Notebook 05. Existing 01–06 stay unchanged. The Tonellato slides govern course exposition within this scope. Reader-facing source labels are omitted in accordance with the writing convention, with provenance recorded here and in the source notes.

## Required main sections

1. Stochastic Processes.
2. Realized Time Series.
3. Non-stationary Stochastic Processes.
4. Stationary Stochastic Processes.

## Content

- [COURSE] Define {Y_t}, t in Z, histories and finite-dimensional distributions. Marginals alone do not characterize dependence. Define population mean, variance, autocovariance and autocorrelation functions with course notation.
- [COURSE] A time series is one partial realization. Define sample mean, variance, sample autocovariance with n-j denominator and sample autocorrelation. Distinguish the lag-zero n denominator from adjusted n-1 sample variance. Ergodicity is needed to learn population properties from one history.
- [SIMULATION] One figure shows several histories from one iid Gaussian process, highlighting one as the observed history. Fixed seed 501. One concise numerical sample/population comparison, without estimator-distribution or AR-model demonstrations.
- [COURSE] Deterministic linear trend, random walk and random walk with drift. Derive time-dependent means, variance and random-walk covariance using shared innovations. Explain trend stationarity versus stochastic trends without formal integration orders or modeling differencing.
- [SIMULATION] One coordinated path comparison, retaining the course length 300, starting value/intercept 10, deterministic slope .1 and shock SD .5. Drift .1 is the required concise extension. Fixed seed 506.
- [COURSE] Strict and weak/second-order/covariance stationarity, finite-second-moment qualification and ergodicity. State the course's moment-ergodicity interpretation with a concise qualification rather than claiming it characterizes general ergodicity.
- [COURSE] White noise is uncorrelated; IID white noise additionally independent/identically distributed; Gaussian white noise iid normal. Marginal normality is not joint Gaussianity. Define Gaussian processes through all finite joint laws and connect to 02.

## Boundary and acceptance

Foundations only. No Wold/AR/MA/ARMA/ARIMA, unit-root test, integration order, differencing as a modeling procedure, estimation or forecasting demonstration. These belong in 08. End with the question of constructing explicit dependence models. Reuse descriptive and sample-dependence functions; preserve all tested APIs. Prefer the two purposeful figures over repeated path/noise/ergodicity plots. Execute in a fresh kernel.
