# Notebook 02 — Multivariate Random Variables

This contract incorporates the latest course-fidelity review, replacing the earlier multivariate-family extension catalogue. The Tonellato slides govern course exposition. Follow docs/WRITING_STYLE.md and record provenance outside reader-facing prose.

## Required section order

1. From random variables to random vectors and joint distributions.
2. Marginal distributions.
3. Conditional distributions, with the course uniform-band example.
4. Covariance and correlation.
5. Mean vector and covariance matrix.
6. Linear combinations and transformations.
7. Multivariate Gaussian distribution and conditioning.
8. Financial mean–variance interpretation.
9. Population versus finite sample.

## Content

- [COURSE] Random vectors and realizations, joint CDF/PDF, marginal integration and conditional density ratio. Preserve the uniform band f(x,y)=1/(2c) on 0<=x<=1, x-c<=y<=x+c. Show marginal and conditional formulas, conditional mean x and variance c²/3. One coordinated figure at c=1, x=.25 replaces repeated band examples.
- [COURSE] Covariance, correlation, units and finite positive variances. Compute the uniform-band means, variances, covariance and correlation. Zero covariance need not imply independence, but does for jointly Gaussian components.
- [COURSE] Mean vector, covariance matrix, positive semidefiniteness and sample counterpart.
- [COURSE/DERIVATION] E[c0+c'X]=c0+c'mu and Var[c0+c'X]=c'Sigma c. Show the two-variable covariance term and one-variable Y=a+bX as a special case. Include E[AX+b] and Cov[AX+b]. This is the home of linear material removed from 01.
- [COURSE] Gaussian joint density, marginal/conditional/linear-transformation properties. Use standard marginals with rho=-.95,0,.95 in a single contour comparison. Conditional comparison uses rho=.8 and x1=-3,0,3, with variance .36, matching the course parameters.
- [FINANCE] Fixed-weight simple portfolio returns, Rp=w'R, E[Rp]=w'mu, Var[Rp]=w'Sigma w. Keep mean–variance central. A compact two-asset table illustrates correlation's effect on variance without another sampling experiment.
- [EXTENSION/SIMULATION] Preserve a concise iid Gaussian repeated-sample illustration of sample mean, variance, covariance and correlation. Use one covariance figure plus a numerical summary. Replication percentiles are not confidence intervals. Independence is across vectors, not necessarily within each vector.

## Exclusions and reuse

No extra multivariate t/Laplace/skew-normal family catalogue, sine/cosine simulation, inference, PCA, optimization, efficient frontiers, CAPM or factor models. Reuse tested finstats.multivariate functions and preserve all useful earlier APIs. No new wrappers are needed. Execute in a fresh kernel.
