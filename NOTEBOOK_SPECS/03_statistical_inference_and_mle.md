# Notebook 03 — Statistical Inference and MLE

This contract incorporates the latest course-fidelity review. The subsequent user review excludes Bernoulli examples. Model selection and a general numerical optimization demonstration also remain excluded. The Tonellato slides govern the course sequence and examples. Follow docs/WRITING_STYLE.md and retain provenance outside reader-facing prose.

## Required section order

1. Statistical models and inferential problems.
2. Random samples, estimators and estimates.
3. Bias.
4. Variance and standard error.
5. Mean-squared error.
6. Consistency.
7. Sampling distributions under the Gaussian model.
8. Confidence intervals.
9. Hypothesis testing.
10. Likelihood.
11. Maximum likelihood estimation, including Gaussian, concise numerical verification and Fisher information.

## Content

- [COURSE] Statistical model as a family, population parameter fixed, iid sample, estimator random, realized estimate fixed. Point estimation, intervals and tests answer different questions.
- [COURSE] Bias, sampling variance/SE, MSE decomposition and consistency, in that order. Keep MSE as an organizing criterion without a separate shrinkage demonstration. MSE consistency implies consistency, but not conversely.
- [COURSE] Sample mean, adjusted sample variance, exact Gaussian/chi-squared/Student-t laws. [SIMULATION] One sample-mean repeated-sampling figure with bias/variance/MSE table illustrates these concepts. Distinguish observation variance from sampling variance.
- [COURSE] Known-variance Gaussian interval, unknown-variance Student-t mean interval and chi-squared variance interval. Random interval versus realized interval and repeated-sampling coverage.
- [COURSE] Null/alternative, test statistic, critical region, Type I/II error and p-value. Preserve the return test n=100, sigma=.02, observed mean=.005, null mean=0, z=2.5, p approximately .0124. No invented data are needed for that supplied summary example.
- [COURSE] Iid likelihood/log-likelihood with fixed observed data. Use the Gaussian example in the following MLE section.
- [COURSE] Gaussian theta=(mu,sigma²), log-likelihood, partial derivatives, sample mean and denominator-n variance MLE, Hessian at the maximum. One log-likelihood contour replaces duplicate likelihood and log-likelihood surfaces. Preserve the size-ten N(5,4) sample and reuse it for exact intervals.
- [SIMULATION] One numerical Gaussian maximization checked against the analytical solution, without local-optimum demonstrations or a numerical-methods tutorial.
- [COURSE/DERIVATION] Score, expected/observed information, inverse information standard errors and regular MLE asymptotics. Make scaling precise: I_n=n I_1 and sqrt(n)(theta_hat-theta) converges to N(0,I_1^-1). The unscaled Gaussian law is a large-n approximation. Mention conditions and equivariance concisely.

## Exclusions and reuse

No AIC/BIC or model-selection section, Cauchy multi-start demonstration, integrated Monte Carlo synthesis, time-series likelihood, bootstrap, Bayesian inference or regression. Reuse tested finstats.inference functions, and descriptive functions. Preserve the tested Bernoulli helper for compatibility, without importing or using it in this notebook. Preserve legacy fitters/information criteria for later notebooks. Run the full suite and execute in a fresh kernel.
