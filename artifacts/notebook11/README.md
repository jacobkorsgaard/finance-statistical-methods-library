# Notebook 11 rolling-refit calculation

`rolling_spy_forecasts.csv` records one expanding-window joint ARMA(2,2)–GARCH(1,1) Student-t refit for each of 752 SPY target returns in 2023–2025. The initial estimation period is 2017–2022. Every refit uses only returns before the target. Two optimizer starts are checked; all mean, volatility and density parameters are reestimated. Coefficient standard errors are omitted for these repeated prediction fits.

`target_index` indexes the full simple-return series constructed from `data/spy_adjusted_prices_2017_2025.csv`. The notebook attaches its return dates. The record includes the one-step mean, innovation variance, 5% loss VaR/ES, likelihood, persistence, convergence count and all estimated parameters.

The JSON file verifies the record checksum and a signature covering data, initial parameters, NumPy/SciPy versions, estimation source and settings. Notebook 11 resumes an incomplete matching calculation or reuses a complete matching one. Changes invalidate it automatically. This is a saved computation, not an alternative fixed-parameter forecasting procedure. The algorithm can also run without a saved record by setting `cache_path=None`.
