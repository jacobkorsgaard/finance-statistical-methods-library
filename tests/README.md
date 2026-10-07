# Tests

The original four test modules remain unchanged and cover the preserved package.
`test_descriptive.py` and `test_distributions.py` cover the Notebook 01 API.
`test_multivariate.py` covers the five functions used in Notebook 02, including analytical identities, library agreement, affine transformations, shape/denominator conventions, and invalid inputs.

`test_inference.py` covers all ten Notebook 03 functions: replication diagnostics, standard errors, exact Gaussian intervals, elementary likelihoods, Gaussian MLEs, and scalar squared Wald statistics. Checks include hand calculations, SciPy agreement, boundary/domain cases, and fixed-seed interval coverage.

`test_nonparametric_risk.py` covers empirical distributions, quantiles, QQ data and risk formulas.
`test_timeseries.py` covers sample dependence, AR/MA/ARMA root conventions, random walks, differencing and forecast errors.
`test_volatility.py` covers ARCH/GARCH simulations, variance recursions, moment restrictions and forecasting.


`test_data_snapshots.py` verifies every fixed CSV against its recorded SHA-256, schema and row count. It checks the US/German price-index and FX metadata, date ordering, positive levels, fixed window, and the shared price calendar used in Notebook 05. These checks are offline. Statistical functions used by 00–05 retain their existing descriptive, nonparametric, risk, return, multivariate, and distribution-fit tests.

`test_multivariate_laplace.py` checks the bivariate symmetric Laplace density against its Gaussian–exponential mixture, radial mass and second moment, univariate marginal and affine Jacobian, plus vectorization, singular-center and input-domain behavior.
