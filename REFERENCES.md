# References and Further Reading

## Primary course source

**Tonellato, Stefano.** *Statistical Models and Methods for Finance*.
Department of Economics, Università Ca’ Foscari Venezia. Course slide deck.

The initial version of this repository is intentionally built close to Professor Tonellato's course structure, notation, examples, simulations, and empirical demonstrations. Where course material is reproduced computationally, the objective is to translate the statistical reasoning and examples into Python while preserving the original pedagogical structure as closely as practical.

The supplied slide deck is stored locally in:

```text
references/tonellato_statistical_models_and_methods_for_finance_slides.pdf
```

This local PDF is ignored by git and is not included in the public repository.

---

## Core textbook

**Ruppert, David.** *Statistics and Data Analysis for Financial Engineering with R Examples*. Springer.
The course slides list Ruppert (2011 or 2015 editions) as the principal textbook.

This is the main recommendation for readers who want a fuller treatment of the probability, statistics, simulation, and financial-data concepts developed in the repository.

---

## Statistical inference references

**Casella, George, and Roger L. Berger.** *Statistical Inference*. 2nd ed. 2002.
Recommended in the course slides as an advanced reference for statistical inference.

Useful for deeper study of:

- estimators and sampling distributions;
- sufficiency and likelihood;
- point estimation;
- confidence intervals;
- hypothesis testing;
- asymptotic theory.

**Wasserman, Larry.** *All of Statistics: A Concise Course in Statistical Inference*. Springer, 2013.
Recommended in the course slides as an advanced but compact reference.

Useful for:

- probability foundations;
- estimation;
- hypothesis testing;
- non-parametric methods;
- asymptotic statistics.

**Leekley, R. M.** *Applied Statistics for Business and Economics*. 2010.
Listed in the course slides as a more basic reference for Part I.

---

## Time-series references

**Cryer, Jonathan D., and Kung-Sik Chan.** *Time Series Analysis: With Applications in R*. Springer, 2008.

Useful for:

- stochastic processes;
- stationarity;
- AR, MA, and ARMA models;
- forecasting;
- model diagnostics.

**Tsay, Ruey S.** *Analysis of Financial Time Series*. 3rd ed. Wiley, 2010.

Especially useful for:

- financial return dynamics;
- volatility modelling;
- ARCH/GARCH;
- financial time-series diagnostics;
- empirical applications.

---

## Repository referencing principle

The repository should distinguish clearly between:

- **course-derived material**, based directly on Professor Tonellato's slides;
- **textbook-supported elaboration**, used for clarification and further reading;
- **our own derivations**, which follow from the course material;
- **our own simulations/extensions**, which are included to make the statistical behavior clearer.

The slide deck remains the primary source for v1. Supporting books should deepen or clarify the material rather than silently replace the course's terminology, notation, examples, or sequence.

---

## Copyright and source files

Do not commit or redistribute copyrighted textbooks unless the repository owner has explicit permission to do so.

If legally obtained personal copies of supporting books are used during development, keep them outside the public repository or in a local ignored folder such as:

```text
references/local_books/
```

and add that path to `.gitignore`.

The public repository should normally contain bibliographic references rather than full textbook files.


## Secondary writing reference

Korsgaard, Jacob, and Axel Ulveman (2026). *Firm Frictions, Adjustment Dynamics, and Cross-Sectional Returns*. Master’s thesis, Copenhagen Business School. The local PDF in `references/` informs prose style, economic intuition, and treatment of uncertainty. It is not a replacement for the course’s statistical content, and notebook wording is not copied from it.


## Market application data

The new empirical distribution/risk notebook uses fixed Yahoo Finance snapshots for S&P 500 (`^GSPC`), DAX price (`^GDAXIP`, DAX K), and USD per EUR (`EURUSD=X`), with provenance in [data/README.md](data/README.md) and [data/manifest.json](data/manifest.json). The dividend-basis distinction is checked against the [official STOXX DAXK description](https://stoxx.com/index/DAXK/) and [DAX calculation guide](https://www.stoxx.com/document/News/2024/May/DAX_Equity_Index_Calculation_Guide_20240529.pdf). These market/currency additions are approved applications beyond the original combined course example.
