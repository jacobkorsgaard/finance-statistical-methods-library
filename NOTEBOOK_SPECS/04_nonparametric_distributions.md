# Notebook 04 — Non-parametric Distributions

This specification implements the approved split of the former combined Notebook 04. Follow the reviewed writing and figure conventions of Notebooks 01–03. Use first-person singular for illustration choices, ordinary prose introductions, spaced subsections, mathematics before package calls, and fixed seeds. Source labels appear in development documents, not reader-facing notebook prose.

## Purpose and scope

Estimate a distribution without specifying a parametric family. Use iid simulations from known populations only. Do not use S&P 500, DAX, other empirical financial data, VaR/ES, fitted financial distributions, or time-series models.

## Section order

1. From a population to an observed sample.
2. Empirical cumulative distribution.
3. Empirical quantiles, with a quantile-precision subsection.
4. Histograms and kernel density estimation.
5. Quantile–quantile plots.
6. What the methods reveal.

## Required material

- [COURSE] Population distribution versus finite observed sample, using Gaussian and unit-variance Student-t5 examples.
- [COURSE] ECDF Fhat(x)=n^-1 sum I(Xi<=x), ties and right continuity. Distinguish a continuous population from its atomic empirical distribution.
- [DERIVATION] Under iid sampling, E[Fhat(x)]=F(x), Var[Fhat(x)]=F(x)(1-F(x))/n. Chebyshev yields pointwise consistency. Bounded indicators allow this even when the population mean is undefined.
- [SIMULATION] Preserve the course Cauchy comparison n=5,100,1000, using nested samples, seed 0, common central axes, and full ranges printed. Independently verify Fhat(0) mean/variance with 2,000 samples of size 100, seed 404.
- [COURSE] Inverse-ECDF order-statistic quantile ceil(nq), not an interpolated NumPy quantile. State endpoint conventions and quantile consistency/regularity conditions. Retain the correctly scaled sqrt(n) asymptotic quantile law.
- [SIMULATION] Cauchy 75th percentile comparison using the same prefixes. Additional Gaussian/Student-t quantile precision at n=250, B=500, seed 405, with center/tail summaries and a 99th-percentile error figure.
- [COURSE] Histogram bin dependence and KDE formula with normalized kernel. Explain bandwidth versus kernel and avoid claiming KDE reveals unobserved extreme outcomes.
- [SIMULATION] Retain the course bandwidth multipliers .1,1,100, applying them to a known two-group Gaussian-mixture population, n=250, seed 406. Compare KDE with the true density and report bandwidth. Document SciPy Scott versus R default difference.
- [COURSE] QQ plotting probabilities (i-.5)/n, reference quantiles, location/scale line interpretation, tail variability, descriptive rather than testing status.
- [SIMULATION] Gaussian and unit-variance Student-t5 n=1000, seed 408. Compare Gaussian references, then a correctly specified Student-t reference.

## Reuse and acceptance

Reuse `finstats.nonparametric` ECDF, quantile, and QQ data. Use SciPy KDE directly after its mathematics, with no thin wrapper. Preserve existing tests. All figures have explicit statistical purposes and population references. Execute cleanly from top to bottom without external data or network. No reader-facing exercise or learning-objective labels.
