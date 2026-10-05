# Notebook 01 — Random Variables and Distributions

This contract incorporates the latest course-fidelity review and supersedes earlier transformation and continuous-only instructions. The Tonellato slides govern course terminology, notation, parameterization and examples. Follow docs/WRITING_STYLE.md. Provenance labels belong in the contract and source notes rather than reader-facing notebook prose.

## Purpose and section order

Univariate foundations before estimation, with a continuous focus and a brief discrete/PMF distinction.

1. Random variables and probability distributions, including the CDF and discrete PMF.
2. Probability density and cumulative distribution.
3. Quantiles.
4. Expectation and variance.
5. Moments, skewness and kurtosis.
6. Examples of univariate distributions: Gaussian, Student-t, Laplace and standard skewed normal.
7. Population versus finite sample, with concise financial interpretation.

## Content

- [COURSE] X versus realized x, support, CDF, interval probabilities, discrete versus continuous, PMF/PDF normalization. Use the course partial-derivative notation for the PDF, with a differentiability qualification.
- [COURSE] Quantiles, generalized inverse, median and quartiles. One coordinated Gaussian density/CDF illustration at q=.75 is sufficient.
- [COURSE] Expectation in both cases, variance, standard deviation, raw/central moments, skewness, raw/excess kurtosis and finite-moment requirements. [DERIVATION] Briefly expand the variance identity.
- [COURSE] Retain the four existing families and their formulas. Preserve Gaussian mean/variance, Student-t real degrees of freedom and scale-versus-SD distinction, Laplace scale and standard skew-normal alpha. Use the course Gaussian (0,1) and (3,.25), Student-t nu=1,5,10,30 and Gaussian limit, Laplace(0,1) versus N(0,2), skew-normal alpha=-3,0,3. Consolidate density comparisons rather than repeat PDF/CDF/tail plots.
- [SIMULATION] One fixed-seed Gaussian finite-sample illustration is enough to distinguish known population quantities from realized summaries. State iid sampling and all sample-statistic conventions. An unknown financial population remains a modeling assumption.

## Exclusions and reuse

No standalone linear or nonlinear transformations, affine formulas or location-scale construction in 01. Family density parameterizations may retain location and scale parameters without deriving transformations. Linear combinations and transformations belong in 02. No fitting, confidence intervals, tests, ECDF/KDE, risk measures or time-series material.

Reuse finstats.descriptive and the tested Student-t population moment helpers. Preserve useful existing transformation APIs and tests even when removed from this notebook. Avoid numerical-library wrappers. Execute from top to bottom in a fresh kernel.
