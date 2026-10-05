# Source conventions and clarifications

The primary statistical source is Stefano Tonellato's *Statistical Models and Methods for Finance*. Bibliography is in [REFERENCES.md](../REFERENCES.md), and the current scope is in [NOTEBOOK_SPECS/](../NOTEBOOK_SPECS/). The slides and supporting books are local ignored development references. Notebooks present the exposition without source tags or exact slide citations.

These notes retain the substantive source decisions from earlier implementation reports. They describe current conventions rather than the history of individual edits.

## Probability and inference

- [COURSE] CDFs characterize distributions; continuous PDFs represent probability as area. Partial-derivative notation follows the course, with the differentiability qualification stated.
- [DERIVATION] For Y=a+bX, variance is b² Var(X). A slide instead labels the transformation Y=aX+b but assigns variance to the intercept coefficient. The notebook's coefficient and variance are consistent. All standalone linear transformations now belong in 02.
- [DERIVATION] The K-dimensional Gaussian normalization is (2*pi)^(K/2)*sqrt(det(Sigma)); a course display repeats the bivariate constant. The quadratic form uses x-mu. This agrees with Ruppert and Matteson, Section 7.5, equation (7.15). Gaussian contour covariances are constructed consistently from the stated correlations and marginal SDs.
- [COURSE] The uniform band has density 1/(2c) on 0<=x<=1, x-c<=y<=x+c. Conditional moments are E[Y|X=x]=x and Var(Y|X=x)=c²/3. This resolves the swapped-variable conditional-variance caption in some course figures.
- [DERIVATION] Raw kurtosis has Gaussian value 3, excess kurtosis value 0. The general lower bound is raw kurtosis >=1, including equality. Sample shape statistics use empirical central moments with denominator n, without bias correction, and are distinguished from sample variance using n-1. Course raw/excess labels are not propagated ambiguously.
- [DERIVATION] Finite Student-t moments require degrees of freedom above the moment order. Symmetry alone does not supply a finite mean or skewness. Student-t scale is not its standard deviation; skew-normal location/scale are not generally its mean/SD.
- [DERIVATION] Full-sample Fisher information is I_n=n I_1. The precise regular MLE limit is sqrt(n)(theta_hat-theta) converging to N(0,I_1^-1), and the unscaled covariance I_n^-1 is a large-sample approximation. This agrees with Ruppert and Matteson, Section 5.10, equations (5.18)–(5.25). Gaussian exact intervals remain distinct from asymptotic intervals.
- [EXTENSION] The four existing univariate families are covered selectively. No extra multivariate family catalogue is included. Bernoulli likelihood/MLE is omitted from 03 at the user's request, although its tested library helper remains available. Model selection appears in later fitting/modeling notebooks.

## Empirical distributions and financial risk

- [COURSE/SIMULATION] Retain the Cauchy ECDF/quantile comparison, bandwidth sensitivity and Student-t rare-loss example. Population simulations use reproducible Python seeds rather than identical R random streams.
- [DERIVATION] Quantile asymptotics use the properly scaled sampling law. Continuous ES density and quantile-integral expressions agree under the stated assumptions; empirical atoms require an explicit boundary convention.
- [COURSE] The historical risk notebook uses strict loss exceedance. The older inclusive historical-ES API is preserved separately, with its own convention, rather than silently changing it.
- [EXTENSION] The course's observed-return bandwidth demonstration becomes a known Gaussian-mixture illustration in 04. Bandwidth multipliers are retained, but SciPy Scott bandwidth differs from the R default.
- [EXTENSION] The US/German application adds a longer fixed window, FX conversion and a fixed-weight common-currency portfolio. It uses simple percentage returns so linear portfolio aggregation is exact. Price dates are intersected before returns are computed. Asynchronous market/FX closes, dividend exclusion and index-price units are documented in [data/README.md](../data/README.md).
- [FINANCE] Historical or fitted VaR/ES are finite-sample estimates, not observed population truth. A pooled iid working likelihood does not establish that real returns are iid or that their distribution is stable.

## Processes, mean models and volatility

- [COURSE] Process moments use the course sample autocovariance denominator n-j. Production Ljung–Box calculations use denominator n and their stated fitted-model degrees-of-freedom adjustment.
- [DERIVATION] Wold's linear representation applies to the purely nondeterministic component, with a separate perfectly predictable component where needed. Innovations are orthogonal prediction errors, not necessarily independent shocks.
- [DERIVATION] The course's ergodicity statements are interpreted as moment convergence. Stationarity alone does not guarantee ergodicity. Gaussian marginal laws alone do not establish a Gaussian process. White noise, IID noise and Gaussian white noise are distinct assumptions.
- [DERIVATION] AR(1) conditional-expectation formulas require zero-conditional-mean shocks. Uncorrelated white noise by itself is insufficient. Recursion intercept and unconditional mean differ.
- [COURSE/SIMULATION] Retain the course AR(1) phi=.7, n=300, innovation variance one and stationary Gaussian initialization; deterministic trend/random-walk n=300, level/intercept 10 and shock SD .5; and ARCH omega=2, alpha=.8, n=300, seed 123. Plots are consolidated and Python paths are not claimed to equal R paths.
- [EXTENSION] The requested foundations/modeling/capstone separation is implemented as 07–09 because the user retained the nine-notebook numbering. Drift comparisons, compact AR persistence views, the existing Mishkin holdout and one simulated GARCH fitting example provide focused illustrations without repeating the financial capstone.
- [DERIVATION] The Gaussian ARCH alpha=.8 example has finite return variance but infinite fourth moment because 3*alpha²>=1. Its squared sample ACF is descriptive; no undefined population squared correlation is overlaid.
- [COURSE] In course GARCH(p,q), p counts variance lags and q squared-shock lags. The arch package reverses these argument names. The mapping is explicit, and GARCH(1,1) is unaffected. Student-t innovations are standardized to unit variance.
- [DERIVATION] Conditional-variance recursions, forecasts and long-run variance have separate meanings. Alpha+beta<1 supplies the usual finite unconditional variance under unit-variance shocks; it does not guarantee every higher moment. Standardized squared-residual diagnostic p-values are approximate after fitting and omitted when the fitted innovation fourth moment is not finite.
- [EXTENSION] The capstone retains a parsimonious AR mean and symmetric Student-t innovations rather than the course's more elaborate ARMA/skewed-t fits. Provider differences, a fixed training cutoff and a short holdout preclude exact replication of course coefficients. ADF assesses a specified unit-root null rather than proving stationarity or ergodicity.

The capstone reports its actual limits: near-unit fitted persistence, heavy-tailed innovations and a ten-observation holdout that does not favor the fitted forecasts over simple benchmarks. No finite long-run variance or general forecasting superiority is inferred from those results.
