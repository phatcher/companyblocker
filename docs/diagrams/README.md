# Diagrams

Three diagram families, split by what each is for.

| Family | For | Lives in | Notation/tooling |
| --- | --- | --- | --- |
| Structural | System structure: what the parts are and how they relate | `docs/structurizr/*.dsl` | C4, authored in the Structurizr DSL |
| Behavioural | Behaviour over time: call sequences, activity, decision flow | `docs/plantuml/*.puml` | UML, authored in PlantUML |
| Illustrative | Explaining an idea: data-lineage pictures, design-discussion visuals | `docs/diagrams/*.html` | Hand-authored HTML/SVG, per [`style-guide.md`](style-guide.md) |

All three share one visual identity, the palette and type defined in [`style-guide.md`](style-guide.md), applied through each toolchain's own mechanism, so diagrams from different families read as one project.

## Structural diagrams: C4 via Structurizr

`docs/structurizr/{workspace,model,views}.dsl` is the **sole source of truth** for System Context, Container, and Component views. `workspace.json` is generated output and is never hand-edited.

**Never hand-author a second C4 model.** A hand-authored PlantUML C4 set duplicating these structural facts drifted silently, since nothing kept the two in sync. Where a portable PlantUML or Mermaid rendering of the structure is wanted, export it from Structurizr rather than redrawing it.

### Running Structurizr Lite

```
cd docs/structurizr/.structurizr && docker compose up
```

The workspace UI is then at http://localhost:8090, and Structurizr's own validation warnings are listed at http://localhost:8090/workspace/inspections. Always render a DSL change before treating it as done: a syntax error takes the whole workspace down rather than one view.

## Behavioural diagrams: UML via PlantUML

`docs/plantuml/*.puml` holds sequence, activity, and decision-flow diagrams: the things the Structurizr DSL genuinely cannot express. It has no `alt`/`loop` sequence blocks and no activity or class diagrams, and its "dynamic view" is only a numbered ordering over relationships that already exist in the static model.

Shared skinparams live in `docs/plantuml/_style.puml` and are pulled in with `!include _style.puml`, so the palette is defined once rather than per diagram.

## Illustrative diagrams

Published HTML/SVG explainers, built to the palette, type pairing, and component conventions in [`style-guide.md`](style-guide.md). `resilience-ladder.html` is the reference implementation.

## Which one am I drawing?

- Showing what the parts of the system are and how they connect: **structural**, C4.
- Showing what happens, in what order, or which branch is taken: **behavioural**, UML.
- Explaining a mechanism or an idea to a reader, where the visual itself is the argument: **illustrative**.

If a picture is really making a structural claim, it belongs in the C4 model, not in a bespoke illustration that will drift away from it.
