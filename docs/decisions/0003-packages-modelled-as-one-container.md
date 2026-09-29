# 3. The packages are one container, not five siblings of the pipeline stages

Date: 2026-08-31

## Status

Accepted

## Context

`packages/company_cleanse`, `company_tokenize`, `company_vectorize`, `company_classify` and `company_perturbation` were each modelled as a container of the `CompanyBlocker` system, sitting as flat siblings of the pipeline-stage containers (`orchestration`, `analysis`, `training`, `blocking`, `validation`).

That put two different kinds of thing at the same level. The pipeline stages are this repository's own workflow; the packages are standalone libraries the stages call into. Reading the Container view, the package layer had no identity of its own, and the system boundary implied the packages were internal parts of a pipeline rather than separately publishable deliverables.

It also obscured a real property of the design. The packages carry no dependency on this repo's dataframes, `system_uri`, or truth model, specifically so they can be consumed from outside. This platform's own orchestration is one caller among possible others, adding only its data plumbing around the calls. Modelled as flat siblings, that symmetry was invisible.

Promoting them to separate software systems was considered and rejected: they are not independently running systems, and it would have fragmented the Container view.

## Decision

A single `Packages` container holds the five packages as components. It sits at the same level as the pipeline-stage containers, so the Container view shows one package layer rather than five loose boxes. The `PackagesComponents` view shows the five individually.

Callers reach into the package layer the same way regardless of origin: `orchestration.cleansing -> packages.cleanseRules` has the same shape as `externalConsumers -> packages.cleanseRules`. Both the coarse container-level edges and the specific component-level ones are kept, because the fine ones record which component actually calls and the coarse ones give the Container view a line to draw.

Packages are named for what they do rather than the data they hold: `company_cleanse` is the Company Cleanse Engine that applies rules, not a rules engine that stores them.

## Consequences

The Container view distinguishes this repository's workflow from the libraries it depends on, and the External Python Consumers system can be drawn reaching the package layer directly without implying it reaches into the pipeline.

Component-level references gained a path segment (`pipeline.packages.cleanseRules`), so re-parenting cost a pass over every reference in `model.dsl` and every view include in `views.dsl`.

Structurizr allows no component nesting below this, so a package's internals cannot be modelled as components of that package within this model. Each package's internal structure stays in its own README and, where drawn, in a behavioural diagram.
