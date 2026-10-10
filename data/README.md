# Data snapshots

The repository contains two fixed market snapshots: SPY equity exposure and AGG bond exposure. Notebook 00 uses SPY to introduce returns; Notebook 05 uses both series for distributional analysis; Notebook 11 uses SPY for time-series modelling. The other notebooks use simulations. Execution requires no network requests.

## Equity and bond application (Notebook 05)

The fixed window is 2017-01-01 through 2025-12-31. I use [SPY, the S&P 500 ETF proxy](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy), and [AGG, the broad US investment-grade bond ETF](https://www.ishares.com/us/products/239458/ishares-core-us-aggregate-bond-etf). These are fund exposures, not index levels or bond yields. Both trade in USD. Fund expenses and tracking differences are embedded in observed fund prices.

| File | Series | Columns | Price dates |
|---|---|---|---|
| `spy_adjusted_prices_2017_2025.csv` | SPY | Date, Close, AdjustedClose | 2,262 |
| `agg_adjusted_prices_2017_2025.csv` | AGG | Date, Close, AdjustedClose | 2,262 |

Yahoo Finance chart snapshots were retrieved on 2026-10-06. The [provider describes adjusted close](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html) as adjusted for splits and dividend distributions. Notebook 05 uses AdjustedClose ratios as distribution-adjusted **total-return proxies**, not exact official index total returns. Yahoo's adjustment multipliers need not reproduce an investor's exact cash-distribution reinvestment convention. Close is retained for auditability and is not used for risk calculation.

Timestamps are converted to America/New_York before extracting dates. Both snapshots span 2017-01-03 to 2025-12-31, have identical calendars, and contain no missing closing/adjusted-closing rows. Values are retained at the downloaded floating-point precision. No interpolation, forward fill, winsorization or truncation is applied. Price dates are intersected before returns are calculated, giving 2,261 common return intervals. Weekends and holidays produce multiday calendar intervals. All returns, scales and risk figures are in percentage points, and covariance entries in squared percentage points.

The current application does not construct weighted combinations. Investor taxes are excluded. Fixed snapshots ensure offline execution, even if the provider later revises adjusted prices. Query URLs, currencies, timezones, row counts and SHA-256 checksums are recorded in [manifest.json](manifest.json).

## Training and out-of-sample evaluation

Univariate and joint equity–bond models use 1,509 common return observations with endpoints through 2022-12-30 for training, then 752 endpoints from 2023-01-03 through 2025-12-31 for out-of-sample evaluation. The first out-of-sample return uses the 2022-12-30 adjusted price as its starting level. The date boundary is chosen in advance of this revision; no return is discarded or independently realigned. Fitting and BIC selection use only the training data, and parameters and historical/fitted thresholds remain fixed during evaluation. Notebook 11 uses the same SPY series and date boundary, with expanding-window parameter refits at every out-of-sample origin. The full window was already inspected, so this is a retrospective out-of-sample assessment rather than a previously unseen live test. The joint fit uses the same training pairs and evaluates its implied marginals on the same out-of-sample pairs. Snapshots and their checksums are unchanged.
