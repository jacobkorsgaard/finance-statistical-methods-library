# Repository development instructions

## Governing scope

Current user instructions override earlier repository guidance. ROADMAP.md and NOTEBOOK_SPECS/ define the allowed curriculum. Read the relevant specification, existing notebook/code/tests and corresponding Tonellato slides before editing. Refactor useful existing work rather than rebuilding or duplicating it. Keep specifications synchronized with approved changes.

The supplied Tonellato slides are the primary reference for course material: preserve terminology, notation, conceptual order, examples, parameterizations and interpretation. Translate R demonstrations into Python instead of substituting unrelated examples. Supporting books clarify rather than replace the course. Document necessary corrections and deliberate departures in docs/SOURCE_NOTES.md. Keep [COURSE], [DERIVATION], [SIMULATION], [FINANCE] and [EXTENSION] classifications in specifications and source notes, not notebook prose.

## Writing and mathematical exposition

Follow docs/WRITING_STYLE.md. Notebook 01 is the provisional reference for tone, mathematical depth, pacing and figures. Use the locally supplied thesis as a secondary style reference without copying its wording.

Prefer definition → key mathematical result → one useful illustration if needed → concise interpretation. Do not add content, derivations or figures merely for completeness. Explain mathematics before package calls. Use first-person singular for author choices, direct prose and spaced Markdown subsections. Introduce illustrations naturally with “For illustration, I use…”, without exercise or learning-objective labels.

Preserve the distinction between population parameters, estimators, realized estimates and finite observations. State assumptions and moment-existence restrictions. Use the course's derivative notation and partial derivatives for likelihood gradients. Make parameterizations, denominator conventions, units and information sets explicit.

## Current notebook boundaries

- 01: univariate foundations, brief discrete/PMF distinction and four course families. No standalone transformations or inference.
- 02: vectors, dependence, linear transformations/combinations and financial mean–variance interpretation. No portfolio optimization or multivariate-family catalogue.
- 03: estimator properties, Gaussian sampling/inference, generic likelihood and Gaussian MLE. Omit Bernoulli examples and model selection.
- 04–06: non-parametric simulations, simulated returns/losses/risk, then US/German empirical distributions and risk.
- 07: process foundations only. No AR/MA models, formal integration/differencing, unit-root tests or forecasting demonstration.
- 08: conditional-mean models and forecasting, followed by ARCH/GARCH, standardized diagnostics and variance forecasting.
- 09: one financial capstone applying earlier theory. No substantial new theory or advanced model classes.

Retain the nine-notebook numbering. Honor explicit review checkpoints. Do not expand later notebooks during a scoped revision.

## Code, simulations and validation

Reuse existing finstats functions and mature NumPy/SciPy/statsmodels/arch implementations. Preserve useful finance_statistical_methods APIs. Do not create thin wrappers or generic frameworks without pedagogical or reuse value. Reusable numerical functions need clear parameterization, concise docstrings, domain validation and meaningful tests.

Use fixed numpy.random.Generator seeds. State DGP, parameters, sample size and purpose before each simulation. Every figure should answer a statistical question, with readable labels, clean axes and no decorative or redundant plots.

Run `.venv/bin/python -m pytest` after numerical changes. Execute revised notebooks top to bottom in fresh kernels and retain their outputs for review. Check local Markdown links after documentation cleanup. Avoid tests of notebook prose. Report material limitations and actual diagnostic/holdout failures without inventing successful empirical results.

Data sources, units, calendars and checksums belong in data/README.md and data/manifest.json. Keep local slides, thesis and textbooks ignored. Do not commit environments, caches or temporary inspection scripts. Do not choose a license on the owner's behalf.
