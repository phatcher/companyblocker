# 1. Structurizr owns structural diagrams; PlantUML owns behavioural ones

Date: 2026-08-31

## Status

Accepted

## Context

Structural diagrams existed twice. `docs/structurizr/model.dsl` described the C4 model, and a parallel hand-authored set (`docs/plantuml/c4-l1.puml`, `c4-l2.puml`, `c4-l3.puml`, `c4-l3-*.puml`) described the same systems, containers, components and relationships using the C4-PlantUML macros. Nothing linked the two. Every architectural fact had to be entered in both, and no tooling flagged divergence.

They did diverge. A note asserting that nothing in `src/validation` called `company_perturbation` survived two changes after that dependency became real, because it lived in prose in both models rather than in anything checkable. Several other edges were missing from one copy and present in the other.

The duplication was not deliberate. `docs/structurizr.md` had always described the intended workflow as authoring once in Structurizr and exporting to PlantUML if a portable rendering were needed; the hand-authored C4 PlantUML files were a stand-in that was never replaced by a real export.

PlantUML cannot simply be dropped, though: Structurizr's DSL models structure only. It has no sequence diagram with `alt`/`loop` blocks, no activity or decision-flow diagram, and no class diagram. Its dynamic view is a numbered ordering over relationships already in the static model. The repo genuinely needs those behavioural diagrams.

## Decision

`docs/structurizr/{workspace,model,views}.dsl` is the sole source of truth for structural diagrams: System Context, Container and Component views. `workspace.json` is generated output and is never hand-edited.

`docs/plantuml/*.puml` holds behavioural diagrams only: sequence, activity and decision flow.

The hand-authored C4 PlantUML set is retired. No second structural model is authored by hand. If a portable PlantUML or Mermaid rendering of the structure is wanted, it is exported from Structurizr.

## Consequences

There is one place to change a structural fact, and the diagram a reader sees is generated from it.

Rendering the structure now requires Structurizr tooling rather than any PlantUML-capable viewer. That cost is mitigated by `tooling/export_diagrams.py`, which exports every view to PNG via the Structurizr CLI and PlantUML, both in Docker.

The split is permanent rather than a migration step: the behavioural diagrams stay in PlantUML because Structurizr cannot express them. `docs/diagrams/README.md` records which family to use when, and the `diagrams` skill routes between them.
