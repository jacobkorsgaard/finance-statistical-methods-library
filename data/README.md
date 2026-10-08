# Data snapshots

Notebooks 01–04 use synthetic observations. Notebook 00 uses SPY to motivate financial returns; Notebook 05 uses the paired local market snapshots; Notebooks 06–10 primarily use simulations. The earlier inflation dataset remains available for reuse, and Notebook 11 preserves the earlier capstone dataset. Execution requires no network requests.

## Equity and bond application (Notebook 05)

The fixed window is 2017-01-01 through 2025-12-31. I use [SPY, the S&P 500 ETF proxy](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy), and [AGG, the broad US investment-grade bond ETF](https://www.ishares.com/us/products/239458/ishares-core-us-aggregate-bond-etf). These are fund exposures, not index levels or bond yields. Both trade in USD. Fund expenses and tracking differences are embedded in observed fund prices.

| File | Series | Columns | Price dates |
|---|---|---|---|
| `spy_adjusted_prices_2017_2025.csv` | SPY | Date, Close, AdjustedClose | 2,262 |
| `agg_adjusted_prices_2017_2025.csv` | AGG | Date, Close, AdjustedClose | 2,262 |

Yahoo Finance chart snapshots were retrieved on 2026-10-06. The [provider describes adjusted close](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html) as adjusted for splits and dividend distributions. Notebook 05 uses AdjustedClose ratios as distribution-adjusted **total-return proxies**, not exact official index total returns. Yahoo's adjustment multipliers need not reproduce an investor's exact cash-distribution reinvestment convention. Close is retained for auditability and is not used for risk calculation.

Timestamps are converted to America/New_York before extracting dates. Both snapshots span 2017-01-03 to 2025-12-31, have identical calendars, and contain no missing closing/adjusted-closing rows. Values are retained at the downloaded floating-point precision. No interpolation, forward fill, winsorization or truncation is applied. Price dates are intersected before returns are calculated, giving 2,261 common return intervals. Weekends and holidays produce multiday calendar intervals. All returns, scales and risk figures are in percentage points, and covariance entries in squared percentage points.

The current application does not construct weighted combinations. Investor taxes are excluded. Fixed snapshots ensure offline execution, even if the provider later revises adjusted prices. Query URLs, currencies, timezones, row counts and SHA-256 checksums are recorded in [manifest.json](manifest.json).

## Preserved US/German and FX snapshots

The former US/German market-application inputs remain available for reuse. They are **not used by the current equity–bond application**. The old data conventions and alignment are retained below as provenance, rather than instructions for the current notebook.

The fixed window is 2017-01-01 through 2025-12-31. Yahoo Finance's chart endpoint supplies daily **Close** values, not adjusted closes. The exact query URLs, downloaded series names, currencies, date ranges, row counts, removed null-close counts, and SHA-256 checksums are recorded in [manifest.json](manifest.json). These snapshots were retrieved on 2026-10-04.

| File | Series | Units | Retained prices | First / last observed date |
|---|---|---|---:|---|
| `sp500_prices_2017_2025.csv` | S&P 500, `^GSPC` | USD price-index points | 2,262 | 2017-01-03 / 2025-12-31 |
| `dax_price_prices_2017_2025.csv` | DAX price index, `^GDAXIP`, provider name DAX K | EUR price-index points | 2,280 | 2017-01-03 / 2025-12-30 |
| `eurusd_2017_2025.csv` | `EURUSD=X` | USD per EUR | 2,341 | 2017-01-02 / 2025-12-31 |

The German series is the **price** version. STOXX identifies DAXK as EUR price return, which excludes reinvested dividends. The commonly quoted performance version would introduce a different dividend basis from the S&P 500 price index. See the [official DAX price-index description](https://stoxx.com/index/DAXK/) and [DAX calculation guide](https://www.stoxx.com/document/News/2024/May/DAX_Equity_Index_Calculation_Guide_20240529.pdf).

Download processing converts each timestamp into the provider's exchange timezone before extracting the date: America/New_York for S&P 500, Europe/Berlin for DAX, and Europe/London for EUR/USD. Rows with null closes are removed explicitly: zero US rows, five DAX rows, and eight FX rows. The remaining values are saved with ten decimal places. No forward fill, interpolation, adjusted-close substitution, or winsorization is applied. The CSV snapshots are fixed inputs rather than a live-data feed. Future provider downloads may revise historical values, which would require a separately documented snapshot update.

For the local-market analysis, each index uses its own observed trading dates and simple returns in percentage points. The joint application intersects **price dates before calculating returns**, retaining 2,217 shared levels and 2,216 paired return intervals. This keeps start and end dates identical across the US, German, and FX series around market holidays. An interval can span several calendar days.

German exposure is converted to USD using DAX EUR level × USD per EUR. The portfolio uses simple returns and equal USD weights rebalanced at each shared interval. It is a price-only benchmark approximation, excluding dividends, costs, financing, and investable-fund tracking differences. US, German, and FX closes are date-aligned but not simultaneous. The calculation does not claim synchronous intraday dependence or execution at a common closing time.

## Preserved earlier snapshots

| File | Content and conventions | Use |
|---|---|---|
| `ecdat_sp500.csv` | 2,783 daily decimal log returns, January 1981–April 1991, without exact dates or prices | Preserved course reference |
| `ecdat_capm.csv` | 516 monthly observations, January 1960–December 2002. `rfood` is food-industry excess return in percentage points | Preserved course reference |
| `ecdat_mishkin.csv` | 491 monthly observations, February 1950–December 1990. `pai1` is one-month inflation in annualized percentage units | Preserved inflation course reference |
| `sp500_prices_2017_2018.csv` | 454 Yahoo S&P 500 closes, February 1, 2017–November 16, 2018 | Preserved capstone example in Notebook 11 |

The Ecdat CSV exports retain their original row-number column. They come from the public Rdatasets export of the R package Ecdat. Dataset documentation: [SP500](https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/SP500.html), [Capm](https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Capm.html), and [Mishkin](https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mishkin.html). Exact CSV sources and checksums appear in the manifest. Monthly dates in the inflation example are reconstructed from its documented regular sequence. Exact daily dates are not invented for the old SP500 return-only series.

The earlier short-window price snapshot is deliberately retained rather than replaced with the new long-window data. The time-series examples keep their original training/holdout boundary. Dataset integrity checks are in `tests/test_data_snapshots.py`.


## Notebook 05 estimation/evaluation boundary

Univariate and joint equity–bond models use 1,509 common return observations with endpoints through 2022-12-30 for training, then 752 endpoints from 2023-01-03 through 2025-12-31 for out-of-sample evaluation. The first out-of-sample return uses the 2022-12-30 adjusted price as its starting level. The date boundary is chosen in advance of this revision; no return is discarded or independently realigned. Fitting and BIC selection use only the training data, and parameters and historical/fitted thresholds remain fixed during evaluation. The earlier full window was already inspected, so this is a retrospective out-of-sample assessment rather than a previously unseen live test. The joint fit uses the same training pairs and evaluates its implied marginals on the same out-of-sample pairs. Snapshots and their checksums are unchanged.
