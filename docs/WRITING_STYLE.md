# Writing conventions

Write academically precise, restrained and direct prose. This is an applied statistical library, so retain useful mathematics without turning each notebook into a catalogue of possible proofs or demonstrations.

## Tone

Use the local thesis, *Firm Frictions, Adjustment Dynamics, and Cross-Sectional Returns* (Korsgaard and Ulveman, 2026), as a secondary reference for sentence construction, economic intuition and treatment of uncertainty. Do not copy wording. Prefer a shorter, slightly less formal version of that voice.

Use familiar words and active verbs. Each paragraph develops one idea, with the main point early. Avoid generic introductions, inflated claims, mechanical summaries and prose that merely narrates code. Minimize semicolons. Use “I” for the author's choices and direct mathematical statements elsewhere. Avoid forced questions/answers, flattery and decorative headings.

## Exposition

Definitions precede applications, and mathematics precedes library calls. Usually give a definition, key result, one illustration where helpful, and concise interpretation. Include a derivation only when it materially improves understanding. State and interpret larger results rather than reproducing long proofs.

Keep assumptions distinct from results. Preserve course notation, including partial derivatives where used in the slides. Distinguish random variables from realizations, population moments from sample summaries, estimators from estimates and observation variance from sampling variance. Qualify conclusions where the model, finite sample or moment assumptions limit them.

State iid sampling when used. For vector samples it concerns independence across vectors, not necessarily between components. Temporal ordering matters for time series. A single realized history is not the process itself.

## Illustrations and figures

Use ordinary prose such as “For illustration, I use…”. State the purpose, distribution, parameters, sample size and seed before simulation. Do not use learning-objective labels, exercises or numerical-illustration prompts.

Use one strong figure rather than repeated plots making the same point. A figure must answer a statistical question. Prefer clear axes, restrained colors, informative titles and legends only where useful. Coordinate panels when comparison is the point, while avoiding dense grids of unrelated concepts. No visualization is required when the mathematics suffices.

Use Markdown subsection headings in their own cells with blank lines before text. Avoid crowded inline labels and unnecessary transition sections. Financial interpretation should follow naturally from the mathematics and state the actual relationship rather than claiming generic usefulness.

## Empirical interpretation

Describe what fitted models and diagnostic results support, not what they prove. In-sample fit differs from forecasting ability, and a short holdout cannot establish general superiority. Report failures and uncertainty plainly. Keep return units, currency exposure, sample windows and conditioning information explicit.

Follow the current notebook specifications and [development instructions](../AGENTS.md) for scope and review conventions.
