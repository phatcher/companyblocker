# Diagrams

Three diagram families, split by what each is for.

| Family | For | Lives in | Notation/tooling |
| --- | --- | --- | --- |
| Structural | System structure: what the parts are and how they relate | `docs/structurizr/*.dsl` | C4, authored in the Structurizr DSL |
| Behavioural | Behaviour over time: call sequences, activity, decision flow | `docs/plantuml/*.puml` | UML, authored in PlantUML |
| Illustrative | Explaining an idea: data-lineage pictures, design-discussion visuals | `docs/diagrams/*.html` | Hand-authored HTML/SVG, per [`style-guide.md`](style-guide.md) |

All three share one visual identity, described in [`style-guide.md`](style-guide.md). The palette and fonts are defined once, in [`house-style.css`](house-style.css); `tooling/sync_palette.py` writes them into the generated `docs/plantuml/_palette.puml` and `docs/structurizr/_palette.dsl`, which are never edited by hand.

`tooling/export_diagrams.py` renders every diagram of all three families to `docs/diagrams/png/`, syncing the palette first; `--only html|structurizr|plantuml` narrows it to one family.

## Structural diagrams: C4 via Structurizr

`docs/structurizr/{workspace,model,views}.dsl` is the **sole source of truth** for System Context, Container, and Component views. `workspace.json` is generated output and is never hand-edited.

**Never hand-author a second C4 model**: nothing would keep it in sync with the DSL. Where a portable PlantUML or Mermaid rendering of the structure is wanted, export it from Structurizr rather than redrawing it.

### Running Structurizr Lite

```
uv run python tooling/start_structurizr.py
```

This runs the Docker Compose file under `docs/structurizr/.structurizr/` and opens the workspace UI at http://localhost:8090; it also takes `stop`, `restart`, `status` and `logs`. Structurizr's own validation warnings are listed at http://localhost:8090/workspace/inspections. Always render a DSL change before treating it as done: a syntax error takes the whole workspace down rather than one view.

## Behavioural diagrams: UML via PlantUML

`docs/plantuml/*.puml` holds sequence, activity, and decision-flow diagrams: the things the Structurizr DSL genuinely cannot express. It has no `alt`/`loop` sequence blocks and no activity or class diagrams, and its "dynamic view" is only a numbered ordering over relationships that already exist in the static model.

Shared skinparams live in `docs/plantuml/_style.puml`, which reads the generated `_palette.puml`, and are pulled in with `!include _style.puml`.

## Illustrative diagrams

Published HTML/SVG explainers, built to the palette, type pairing, and component conventions in [`style-guide.md`](style-guide.md). `resilience-ladder.html` is the reference implementation.

## Which one am I drawing?

- Showing what the parts of the system are and how they connect: **structural**, C4.
- Showing what happens, in what order, or which branch is taken: **behavioural**, UML.
- Explaining a mechanism or an idea to a reader, where the visual itself is the argument: **illustrative**.

If a picture is really making a structural claim, it belongs in the C4 model, not in a bespoke illustration that will drift away from it.
