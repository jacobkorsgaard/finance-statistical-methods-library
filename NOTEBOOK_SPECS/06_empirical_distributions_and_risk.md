# Notebook 06 — Empirical Distributions and Financial Risk

This specification implements the approved empirical part of the split. Match the revised 01–03 conventions. S&P 500 and Germany's DAX are the core examples. Reuse earlier definitions and derivations rather than repeat the full theory.

## Section order

1. Markets, data, and return conventions.
2. Return observations and sample summaries.
3. Non-parametric descriptions.
4. Parametric fitting and model comparison, with center/tail and QQ subsections.
5. Historical and parametric loss risk.
6. Aligning markets and currency.
7. Joint returns and dependence.
8. A fixed-weight portfolio.
9. Interpretation and limits.

## Required material

- [EXTENSION] Use dated S&P 500 and DAX **price** indices, with comparable exclusion of reinvested dividends. Avoid silently comparing S&P price and DAX performance indices. Use a fixed 2017–2025 window. Retain original course Ecdat datasets and earlier time-series snapshots separately.
- [EXTENSION] Add USD-per-EUR data for a USD perspective in the joint application. Use dated local snapshots, source URLs, checksums, provider timezone conventions, sample counts and missing-row handling. No runtime downloads, interpolations, winsorization, or forward fills.
- [COURSE] Introduce price-derived returns and distinguish simple/log returns. Use simple percentage returns throughout this application so fixed-weight aggregation is exact. Record this deliberate change from the original course log-return example.
- [COURSE] Sample moments, history, histogram/KDE, ECDF and Gaussian shape reference. Unknown population remains distinct from its observed summaries. State iid as a working assumption rather than an established data property.
- [COURSE] Gaussian, Student-t, Laplace and skew-normal likelihood fits, with parameters, implied means/SDs, likelihood, parameter counts, AIC/BIC. Reuse mature fitting APIs. Same data and units within each candidate comparison. No claim the best family is true or the global numerical optimum is guaranteed.
- [EXTENSION] Compare centers and observed negative-tail probabilities with all four fitted families. Add fitted Gaussian and lowest-AIC QQ comparisons. In-sample marginal fit is not a conditional forecast or tail-specific validation.
- [COURSE] Historical and parametric loss VaR/ES at alpha=.05,.01, with percentage units and historical strict tail counts. Reflect locations and skew-normal shape correctly under negation. Use tested Gaussian/Student-t formulas and numerical upper-tail integrals for Laplace/skew-normal.
- [EXTENSION] Intersect price dates first, then compute returns over matching endpoints. Explain why joining independently calculated daily returns can mismatch starting dates. Report union missing counts and multiday common intervals.
- [DERIVATION] DAX USD level = DAX EUR level * USD/EUR. Verify simple-return compounding including the currency cross term. State that US, German and FX closes are date-aligned but not simultaneous.
- [COURSE] Sample mean vector, covariance/correlation and a joint scatter with a moment-matched Gaussian reference. Dependence is estimated, not known. Covariance ellipses need not describe joint extremes.
- [FINANCE] Equal weights chosen in advance, rebalanced at each common interval, USD price-only benchmark, no dividends/costs/financing. Derive/use Rp=wᵀR, sample mean and variance identities. Distinguish constant weights from constant shares and from portfolio log-return aggregation.
- [FINANCE] Compare component and portfolio historical risk on identical common intervals. Contrast portfolio historical loss tails with the moment-matched multivariate Gaussian implication. No optimized weights, PCA, trading-strategy claims, or conditional/time-series model.

## Reuse and acceptance

Reuse descriptive, nonparametric, multivariate, risk, returns, and fitting functions. Keep calendar/currency choices visible rather than hide them in a new generic framework. Test all data snapshots against their manifest and validate conversion identities. Execute cleanly offline. Interpret actual outcomes without asserting an improvement or chosen family in advance.
