# Notebook 05 — Returns, Losses, and Financial Risk

This specification implements the approved split. Match the revised foundations and use iid simulations from known distributions, not empirical market data.

## Section order

1. Returns and losses.
2. Value at Risk and Expected Shortfall.
3. Parametric population risk, with Gaussian, Student-t, and equal-variance comparison subsections.
4. Historical risk estimates.
5. Sampling variability in risk estimates.
6. Historical and fitted parametric risk.

## Required material

- [COURSE] Simple and log returns, relation log(1+R), percentage versus decimal units, generic L=-R, continuous loss density/CDF under negation. Distinguish monetary loss -V0*R for a simple return from V0*(1-exp(r)) for a log return. Reuse existing price-return implementations.
- [COURSE] Alpha means upper-tail probability. Define VaR=F_L^-1(1-alpha), not maximum possible loss. Define continuous ES as conditional strict tail mean and density integral. State integrated-quantile equivalence for a continuous distribution and boundary-mass distinction for atoms.
- [SIMULATION] Gaussian density/tail illustration at loss mean zero, SD two percentage points, alpha=.05. Distinguish threshold and conditional tail average.
- [DERIVATION] Gaussian VaR and ES, using phi'(u)=-u phi(u). Student-t location/scale VaR and ES using the standard tail integral, with finite ES only for nu>1. Explain nu=2 finite ES/infinite variance and scale versus SD.
- [SIMULATION] Gaussian, Student-t5, Laplace and skew-normal losses with mean zero and unit variance. Compare population 5%/1% VaR/ES and survival curves. Use defining integrals for Laplace/skew-normal rather than adding unnecessary wrappers.
- [COURSE] Historical inverse-ECDF VaR and strict-exceedance ES. State tail-count and ties convention, difference from fractional-mass integrated empirical ES, and undefined empty tails.
- [SIMULATION] Preserve course L=5+2*T_nu rare-tail experiment with (nu,n)=(2,250),(20,250),(2,2500), alpha=.005, seed 415. Compare population and estimated VaR/ES, show tail resolution, and print strict tail counts.
- [SIMULATION] Historical risk estimator distributions at n=250,2500, B=500, unit-variance Student-t5, alpha=.01, seed 505. Compare estimates with fixed true values, not individual loss distributions.
- [SIMULATION] Nested samples n=250,2500 from Student-t5, seed 506. Compare true, historical, fitted Gaussian and fitted Student-t risk on the same observations. Reuse fitting code from Notebook 03. Do not use future observations in a smaller-sample fit.

## Reuse and acceptance

Reuse `finstats.risk`, `finstats.nonparametric`, and preserved return/fitting APIs. Do not change historical ES convention or remove the older inclusive-tail API. Existing risk-domain, tail-integral, and fitting tests must pass. Execute cleanly with fixed seeds and no empirical data or network. No time-series risk model or portfolio optimization.
