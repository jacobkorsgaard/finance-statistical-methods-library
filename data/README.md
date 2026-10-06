# Data snapshots

Notebooks 01–05 use synthetic observations. Notebook 06 reads the local market snapshots below and performs no network requests. The time-series notebooks in 07–09 retain their earlier datasets.

## US and German market application

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
| `ecdat_mishkin.csv` | 491 monthly observations, February 1950–December 1990. `pai1` is one-month inflation in annualized percentage units | Inflation example in Notebook 08 |
| `sp500_prices_2017_2018.csv` | 454 Yahoo S&P 500 closes, February 1, 2017–November 16, 2018 | Preserved capstone example in Notebook 09 |

The Ecdat CSV exports retain their original row-number column. They come from the public Rdatasets export of the R package Ecdat. Dataset documentation: [SP500](https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/SP500.html), [Capm](https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Capm.html), and [Mishkin](https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mishkin.html). Exact CSV sources and checksums appear in the manifest. Monthly dates in the inflation example are reconstructed from its documented regular sequence. Exact daily dates are not invented for the old SP500 return-only series.

The earlier short-window price snapshot is deliberately retained rather than replaced with the new long-window data. The time-series examples keep their original training/holdout boundary. Dataset integrity checks are in `tests/test_data_snapshots.py`.
