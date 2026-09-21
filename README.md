# Finance Statistical Methods Library

A Python library for learning and implementing statistical methods used in finance.

The project is organized around two main statistical object families—distributions and time-series models—with cross-cutting estimation and simulation methods, and risk as an application layer.

## Planned scope

- Returns: simple, gross, and log returns.
- Distributions
  - Parametric: Normal, Student-t, Lognormal, Laplace, Skew-normal.
  - Nonparametric: empirical distribution, ECDF, empirical quantiles, KDE.
  - Diagnostics: Q-Q plots, goodness-of-fit, normality diagnostics.
- Time series
  - Foundations: white noise, random walks, stationarity.
  - Linear: AR, MA, ARMA, ARIMA, SARIMA.
  - Volatility: ARCH, GARCH.
  - Composite: ARIMA-GARCH.
  - Diagnostics: ACF/PACF, stationarity/invertibility, residual checks, ARCH effects.
- Estimation: likelihood, MLE, AIC/BIC.
- Risk: volatility, VaR, Expected Shortfall.
- Simulation: Monte Carlo and bootstrap utilities.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## Package layout

The repository uses the standard `src` layout:

```text
src/
└── finance_statistical_methods/
```

`src` is the source container; `finance_statistical_methods` is the actual importable Python package.

Example:

```python
from finance_statistical_methods.returns import simple_returns
```
